# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.config.kernel import (
    FLASHINFER_MOE_EP_BACKENDS,
    FLASHINFER_MOE_EP_CUTEDSL,
    FLASHINFER_MOE_EP_DEEP_GEMM,
    MEGA_MOE_BACKENDS,
    NATIVE_MEGA_MOE_BACKENDS,
    validate_flashinfer_moe_ep_model,
)
from vllm.model_executor.layers.fused_moe import flashinfer_moe_ep as fi_ep
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_moe_ep import (
    FlashInferMoeEpExperts,
    FlashInferMoeEpPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)


def test_flashinfer_backends_are_megakernels_outside_the_native_model_path():
    assert FLASHINFER_MOE_EP_BACKENDS <= MEGA_MOE_BACKENDS
    assert FLASHINFER_MOE_EP_BACKENDS.isdisjoint(NATIVE_MEGA_MOE_BACKENDS)


def test_only_deep_gemm_backend_is_dsv4_specific():
    validate_flashinfer_moe_ep_model(
        FLASHINFER_MOE_EP_CUTEDSL,
        ["MixtralForCausalLM"],
    )
    with pytest.raises(ValueError, match="only supported for DeepSeek-V4"):
        validate_flashinfer_moe_ep_model(
            FLASHINFER_MOE_EP_DEEP_GEMM,
            ["MixtralForCausalLM"],
        )
    validate_flashinfer_moe_ep_model(
        FLASHINFER_MOE_EP_DEEP_GEMM,
        ["DeepseekV4ForCausalLM"],
    )


@pytest.mark.parametrize(
    "architectures",
    [["KimiK3ForConditionalGeneration"], ["MixtralForCausalLM"]],
)
def test_native_deep_gemm_mega_moe_not_arch_gated(architectures):
    """Models validate native deep_gemm mega constraints at construction time."""
    validate_flashinfer_moe_ep_model("deep_gemm_mega_moe", architectures)


def test_non_fi_backend_ignores_architectures():
    validate_flashinfer_moe_ep_model("auto", ["MixtralForCausalLM"])


@pytest.mark.parametrize("moe_backend", sorted(MEGA_MOE_BACKENDS))
def test_token_sharding_backends_enable_dsv4_sequence_parallel(moe_backend: str):
    from vllm.models.deepseek_v4.nvidia.model import _use_sequence_parallel

    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            enable_expert_parallel=True,
            tensor_parallel_size=8,
            data_parallel_size=1,
        ),
        kernel_config=SimpleNamespace(moe_backend=moe_backend),
    )
    assert _use_sequence_parallel(vllm_config)


def test_dsv4_requests_pre_fc2_router_weight_placement():
    dsv4 = SimpleNamespace(routing_method=RoutingMethodType.DeepseekV4)
    default = SimpleNamespace(routing_method=RoutingMethodType.Default)
    assert fi_ep.apply_topk_in_fc1(dsv4)
    assert not fi_ep.apply_topk_in_fc1(default)


def test_backend_specs_preserve_weight_format_contracts():
    cutedsl = fi_ep.flashinfer_moe_ep_backend_spec(FLASHINFER_MOE_EP_CUTEDSL)
    assert cutedsl.kernel == "cutedsl"
    assert cutedsl.weight_formats == frozenset({"nvfp4", "mxfp4"})

    deep_gemm = fi_ep.flashinfer_moe_ep_backend_spec(FLASHINFER_MOE_EP_DEEP_GEMM)
    assert deep_gemm.kernel == "deep_gemm"
    assert deep_gemm.weight_formats == frozenset({"mxfp4"})


def test_shared_backend_validation_requires_expert_parallel(monkeypatch):
    config = SimpleNamespace(
        weight_transfer_config=None,
        parallel_config=SimpleNamespace(enable_dbo=False, enable_eplb=False),
    )
    monkeypatch.setattr(fi_ep, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(
        fi_ep.current_platform,
        "get_device_capability",
        lambda: None,
    )
    moe = SimpleNamespace(
        moe_backend=FLASHINFER_MOE_EP_CUTEDSL,
        moe_parallel_config=SimpleNamespace(use_ep=False),
        is_lora_enabled=False,
        skip_final_all_reduce=False,
        in_dtype=torch.bfloat16,
        activation=MoEActivation.SILU,
        has_bias=False,
        swiglu_alpha=None,
        swiglu_beta=None,
        routing_method=RoutingMethodType.DeepseekV4,
    )

    with pytest.raises(ValueError, match="expert parallel disabled"):
        fi_ep.validate_flashinfer_moe_ep_config(moe, "nvfp4")

    moe.moe_parallel_config.use_ep = True
    fi_ep.validate_flashinfer_moe_ep_config(moe, "nvfp4")


def test_megakernel_is_a_modular_kernel_with_pass_through_stages():
    """The megakernel plugs into ``FusedMoEKernel`` like any other experts impl.

    Routing stays outside (top-k ids in), dispatch and combine happen inside,
    so prepare passes tokens through untouched and finalize has nothing to do.
    """
    moe = make_dummy_moe_config(
        num_experts=8, experts_per_token=2, hidden_dim=256, intermediate_size=128
    )
    quant_config = FusedMoEQuantConfig.make("nvfp4", weight_dtype="nvfp4")
    kernel = mk.FusedMoEKernel(
        FlashInferMoeEpPrepareAndFinalize(),
        FlashInferMoeEpExperts(moe, quant_config),
    )

    assert not kernel.is_monolithic
    assert kernel.prepare_finalize.output_is_reduced()
    assert kernel.prepare_finalize.topk_indices_dtype() is torch.int32
    assert kernel.fused_experts.expects_unquantized_inputs
    assert isinstance(
        kernel.fused_experts.finalize_weight_and_reduce_impl(), TopKWeightAndReduceNoOP
    )
    assert kernel.fused_experts.workspace_shapes(
        16, 256, 256, 2, 8, 8, None, MoEActivation.SILU
    ) == ((0,), (0,), (16, 256))

    tokens = torch.zeros(4, 256, dtype=torch.bfloat16)
    prepared = kernel.prepare_finalize.prepare(
        tokens,
        torch.ones(4, 2),
        torch.zeros(4, 2, dtype=torch.int32),
        8,
        None,
        False,
        quant_config,
    )
    assert prepared[0] is tokens and all(item is None for item in prepared[1:])
