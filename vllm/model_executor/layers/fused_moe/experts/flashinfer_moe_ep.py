# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer MoE-EP megakernels as a modular-kernel experts implementation.

The megakernel routes on vLLM's top-k output and then dispatches, computes and
combines in one launch, so it is paired with a pass-through prepare/finalize.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.flashinfer_moe_ep import (
    FlashInferMoeEp,
    FlashInferMoeEpEpilogue,
    FlashInferMoeEpWeights,
    apply_topk_in_fc1,
    supports_current_device,
    validate_flashinfer_moe_ep_layer,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
    kNvfp4Static,
)
from vllm.model_executor.utils import replace_parameter

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

# Layer parameters the kernel reads. After ``process_weights_after_loading`` they
# alias the kernel-resident tensors, so EPLB permutes exactly what the kernel uses.
KERNEL_WEIGHT_NAMES = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
EPILOGUE_PARAM_NAMES = (
    "flashinfer_moe_ep_fc1_alpha",
    "flashinfer_moe_ep_fc2_alpha",
    "flashinfer_moe_ep_fc1_norm_const",
)
INPUT_NORM_CONST_ATTR = "flashinfer_moe_ep_input_norm_const"


class FlashInferMoeEpPrepareAndFinalize(MoEPrepareAndFinalizeNoDPEPModular):
    """Pass-through stages: the megakernel dispatches, combines and reduces."""

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def output_is_reduced(self) -> bool:
        return True

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        return a1, None, None, None, None


class FlashInferMoeEpExperts(mk.FusedMoEExpertsModular):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ) -> None:
        super().__init__(moe_config, quant_config)
        self._kernel: FlashInferMoeEp | None = None

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return supports_current_device()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return weight_key in (kNvfp4Static, kMxfp4Static)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation is MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return (
            moe_parallel_config.use_ep
            and not moe_parallel_config.use_batched_activation_format
        )

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return (0,), (0,), (M, K)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        validate_flashinfer_moe_ep_layer(layer)  # type: ignore[arg-type]
        kernel = FlashInferMoeEp(
            self.moe_config,
            FlashInferMoeEpWeights(
                w13=layer.w13_weight,
                w2=layer.w2_weight,
                w13_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
            ),
            _epilogue_from_layer(layer),
            apply_topk_in_fc1=apply_topk_in_fc1(
                self.moe_config,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
            ),
        )
        kernel.warmup()
        for name, tensor in zip(KERNEL_WEIGHT_NAMES, kernel.kernel_weights()):
            replace_parameter(layer, name, tensor)
        self._kernel = kernel

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        if self._kernel is None:
            raise RuntimeError("FlashInfer MoE-EP weights have not been processed")
        output.copy_(self._kernel(hidden_states, topk_ids, topk_weights))


def _epilogue_from_layer(layer: torch.nn.Module) -> FlashInferMoeEpEpilogue:
    fc1_alpha, fc2_alpha, fc1_norm_const = (
        getattr(layer, name, None) for name in EPILOGUE_PARAM_NAMES
    )
    return FlashInferMoeEpEpilogue(
        input_norm_const=getattr(layer, INPUT_NORM_CONST_ATTR, 1.0),
        fc1_alpha=fc1_alpha,
        fc2_alpha=fc2_alpha,
        fc1_norm_const=fc1_norm_const,
    )


def _set_parameter(layer: torch.nn.Module, name: str, tensor: torch.Tensor) -> None:
    if hasattr(layer, name):
        replace_parameter(layer, name, tensor)
    else:
        layer.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))


def make_flashinfer_moe_ep_kernel(
    layer: RoutedExperts,
    moe: FusedMoEConfig,
    weights: FlashInferMoeEpWeights,
    epilogue: FlashInferMoeEpEpilogue | None,
    quant_config: FusedMoEQuantConfig,
) -> mk.FusedMoEKernel:
    """Build the megakernel for ``layer`` at load time.

    The canonical weights and the per-expert epilogue vectors become layer
    parameters first, so ``RoutedExperts.get_expert_weights`` (EPLB) and the
    kernel agree on the tensors.
    """
    assert weights.w13_scale is not None and weights.w2_scale is not None
    kernel_inputs = (weights.w13, weights.w13_scale, weights.w2, weights.w2_scale)
    for name, tensor in zip(KERNEL_WEIGHT_NAMES, kernel_inputs):
        _set_parameter(layer, name, tensor)
    epilogue = epilogue or FlashInferMoeEpEpilogue()
    vectors = (epilogue.fc1_alpha, epilogue.fc2_alpha, epilogue.fc1_norm_const)
    for name, vector in zip(EPILOGUE_PARAM_NAMES, vectors):
        if vector is not None:
            _set_parameter(layer, name, vector)
    setattr(layer, INPUT_NORM_CONST_ATTR, epilogue.input_norm_const)

    kernel = mk.FusedMoEKernel(
        FlashInferMoeEpPrepareAndFinalize(),
        FlashInferMoeEpExperts(moe, quant_config),
    )
    kernel.fused_experts.process_weights_after_loading(layer)
    return kernel
