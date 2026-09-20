# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import types
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from vllm.config import ModelConfig, VllmConfig
from vllm.config.compilation import CompilationMode
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    RoutedExpertsManager,
    bind_routed_experts_capturer,
    get_routed_experts_attn_gid,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

pytestmark = pytest.mark.cpu_test

_REC_MODULE = "vllm.model_executor.layers.fused_moe.routed_experts_capturer"


def _capturer_with_buffer(
    *,
    max_tokens: int = 8,
    num_layers: int = 4,
    num_experts_per_tok: int = 2,
    dp_rank: int = 0,
    tp_size: int = 1,
) -> RoutedExpertsCapturer:
    # Bypass __init__ so the test can use a CPU buffer and skip the
    # VllmConfig dependency. The CUDA device-tensor allocation in the
    # real constructor is not what we are exercising here.
    c = RoutedExpertsCapturer.__new__(RoutedExpertsCapturer)
    c.dp_rank = dp_rank
    c.tp_size = tp_size
    c.device_buffer = torch.full(
        (max_tokens, num_layers, num_experts_per_tok),
        -1,
        dtype=torch.int32,
    )
    return c


class DummyRouter(BaseRouter):
    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.FUSED_TOPK

    def _compute_routing(
        self, hidden_states, router_logits, indices_type, *, input_ids=None
    ):
        topk_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
        return topk_weights, topk_ids

    def _apply_eplb_mapping(self, topk_ids: torch.Tensor) -> torch.Tensor:
        # Make mapping observable without requiring CUDA EPLB path.
        return topk_ids + 10


def _make_router(eplb_state: EplbLayerState | None = None) -> DummyRouter:
    return DummyRouter(
        top_k=2,
        global_num_experts=16,
        eplb_state=eplb_state,
    )


def _make_modular_routed_experts():
    return types.SimpleNamespace(
        quant_method=types.SimpleNamespace(is_monolithic=False),
    )


def _make_model_config(hf_config):
    num_experts_per_token = ModelArchConfigConvertorBase(
        hf_config, hf_config
    ).get_num_experts_per_token()
    model_config = SimpleNamespace(
        hf_text_config=hf_config,
        model_arch_config=SimpleNamespace(
            num_experts_per_token=num_experts_per_token,
        ),
    )
    model_config.get_num_experts = lambda: hf_config.num_experts
    model_config.get_num_experts_per_tok = lambda: (
        ModelConfig.get_num_experts_per_tok(model_config)
    )
    model_config.get_total_num_hidden_layers = lambda: hf_config.num_hidden_layers
    return model_config


def test_routed_experts_manager_uses_gemma4_top_k_experts():
    hf_config = SimpleNamespace(
        num_experts=8,
        top_k_experts=2,
        num_hidden_layers=3,
    )
    vllm_config = SimpleNamespace(model_config=_make_model_config(hf_config))
    kv_cache_spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], kv_cache_spec)],
    )

    manager = RoutedExpertsManager(vllm_config, kv_cache_config)

    assert manager.routed_experts_by_slot.shape == (8, 3, 2)


def test_routed_experts_manager_uses_kimi_k3_experts_per_token():
    hf_config = SimpleNamespace(
        num_experts=8,
        num_experts_per_token=2,
        num_hidden_layers=3,
    )
    vllm_config = SimpleNamespace(model_config=_make_model_config(hf_config))
    kv_cache_spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], kv_cache_spec)],
    )

    manager = RoutedExpertsManager(vllm_config, kv_cache_config)

    assert manager.routed_experts_by_slot.shape == (8, 3, 2)


def test_base_router_capture_pre_eplb_mapping():
    router = _make_router()
    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    topk_weights, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert topk_weights.shape == topk_ids.shape
    assert len(captured) == 1
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_base_router_capture_with_eplb_enabled():
    eplb_state = EplbLayerState()
    eplb_state.expert_load_view = torch.zeros(32, dtype=torch.int64)
    eplb_state.logical_to_physical_map = torch.arange(32).view(32, 1)
    eplb_state.logical_replica_count = torch.ones(32, dtype=torch.int64)
    eplb_state.should_record_tensor = torch.ones((), dtype=torch.bool)
    eplb_state.num_unpadded_tokens_tensors = [torch.tensor(0, dtype=torch.int32)]
    router = _make_router(eplb_state=eplb_state)

    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    _, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert len(captured) == 1
    # Capture should see logical ids pre-EPLB mapping.
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    # Our DummyRouter mapping adds +10.
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_public_binding_only_visits_target_model(monkeypatch):
    class DummyFusedMoE:
        def __init__(self, layer_id):
            self.layer_id = layer_id
            self.router = _make_router()
            self._quant_method = _make_modular_routed_experts().quant_method

    target_module = DummyFusedMoE(layer_id=7)
    draft_module = DummyFusedMoE(layer_id=0)

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)
    calls = []
    capturer = types.SimpleNamespace(capture=lambda *args: calls.append(args))

    bind_routed_experts_capturer(
        types.SimpleNamespace(modules=lambda: [target_module]), capturer
    )

    assert target_module.router.capture_fn is not None
    assert draft_module.router.capture_fn is None
    topk_ids = torch.tensor([[5, 6]])
    target_module.router.capture_fn(topk_ids)
    assert calls == [(7, topk_ids)]


def test_public_binding_rejects_monolithic_without_replay_support(monkeypatch):
    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 3
            self.router = _make_router()
            # Use a concrete monolithic expert and override its capability
            # instead of instantiating the abstract base class directly.
            from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
                CPUExpertsFp8,
            )

            fused_experts = CPUExpertsFp8.__new__(CPUExpertsFp8)
            self.routed_experts = types.SimpleNamespace(
                quant_method=types.SimpleNamespace(
                    is_monolithic=True,
                    moe_kernel=types.SimpleNamespace(
                        impl=types.SimpleNamespace(fused_experts=fused_experts)
                    ),
                )
            )
            self._quant_method = self.routed_experts.quant_method
            self._quant_method.moe_kernel.impl.fused_experts = fused_experts
            fused_experts.supports_routing_replay_capture = lambda: False

    class DummyCapturer:
        def capture(self, layer_id, topk_ids):
            pass

    dummy_module = DummyFusedMoE()
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)

    with pytest.raises(ValueError, match="monolithic MoE kernel"):
        bind_routed_experts_capturer(
            types.SimpleNamespace(modules=lambda: [dummy_module]), DummyCapturer()
        )


def test_routed_experts_capturer_single_dp_no_metadata():
    """dp_metadata is None: capture writes the full topk_ids rows."""
    capturer = _capturer_with_buffer(dp_rank=0)
    topk = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    ctx = SimpleNamespace(dp_metadata=None, additional_kwargs={})
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk)
    assert capturer.device_buffer[3, 0, 0].item() == -1


def test_routed_experts_capturer_honors_microbatch_offset():
    capturer = _capturer_with_buffer(dp_rank=0)
    topk = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=None,
        additional_kwargs={"routed_experts_token_offset": 4},
    )
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert torch.equal(capturer.device_buffer[4:7, 0, :], topk)
    assert (capturer.device_buffer[:4, 0, :] == -1).all()


def test_routed_experts_snapshot_compacts_staged_rows():
    capturer = _capturer_with_buffer(max_tokens=8, num_layers=1)
    capturer.attn_gid = 0
    capturer.device_buffer.copy_(torch.arange(16).reshape(8, 1, 2))
    slots = torch.arange(8).reshape(1, 8)
    rows = torch.tensor([0, 1, 4, 5])
    result = capturer.get_routed_experts(slots, 4, rows)
    expected_routing = capturer.device_buffer[rows].clone()
    torch.testing.assert_close(result.routing_data, expected_routing)
    torch.testing.assert_close(result.slot_mapping, slots[0, :4])
    capturer.device_buffer.fill_(-1)
    torch.testing.assert_close(result.routing_data, expected_routing)


def test_routed_staging_end_to_end_preserves_logical_token_mapping():
    """Diagnose the complete logical -> physical -> logical row mapping."""
    from vllm.v1.worker.gpu.ubatch_utils import (
        compact_staged_rows,
        restore_staged_inputs,
        stage_decode_tokens,
    )
    from vllm.v1.worker.ubatch_utils import UBatchSlice

    num_tokens = 63
    padded = 128
    logical_ids = torch.arange(num_tokens, dtype=torch.int32) + 1000
    logical_positions = torch.arange(num_tokens, dtype=torch.int64) + 2000
    logical_blocks = torch.arange(num_tokens * 2, dtype=torch.int32).reshape(
        num_tokens, 2
    )
    logical_slots = torch.arange(num_tokens, dtype=torch.int64).unsqueeze(0) + 3000
    batch = SimpleNamespace(
        num_tokens=num_tokens,
        num_tokens_after_padding=padded,
        num_reqs=num_tokens,
        has_prefill=False,
        num_draft_tokens=0,
        num_scheduled_tokens=np.ones(num_tokens, dtype=np.int32),
        input_ids=torch.cat(
            (
                logical_ids,
                torch.full((padded - num_tokens,), -1, dtype=logical_ids.dtype),
            )
        ),
        positions=torch.cat(
            (logical_positions, torch.full((padded - num_tokens,), -1))
        ),
        is_padding=torch.zeros(padded, dtype=torch.bool),
    )
    blocks = torch.cat(
        (
            logical_blocks,
            torch.full((padded - num_tokens, 2), -1, dtype=logical_blocks.dtype),
        ),
        dim=0,
    )
    slots = torch.cat((logical_slots, torch.full((1, padded - num_tokens), -1)), dim=1)
    slices = [
        UBatchSlice(slice(0, 64), slice(0, 64)),
        UBatchSlice(slice(64, 128), slice(64, 128)),
    ]

    rows = stage_decode_tokens(batch, (blocks,), slots, slices)
    assert rows.tolist() == [*range(32), *range(64, 95)]
    torch.testing.assert_close(batch.input_ids[rows], logical_ids)

    capturer = _capturer_with_buffer(max_tokens=padded, num_layers=1)
    capturer.attn_gid = 0
    expected_routes = torch.stack(
        (logical_ids.remainder(17), logical_positions.to(torch.int32).remainder(19)),
        dim=1,
    )
    for microbatch in slices:
        token_slice = microbatch.token_slice
        physical_routes = torch.stack(
            (
                batch.input_ids[token_slice].remainder(17),
                batch.positions[token_slice].to(torch.int32).remainder(19),
            ),
            dim=1,
        )
        ctx = SimpleNamespace(
            dp_metadata=None,
            additional_kwargs={
                "routed_experts_token_offset": token_slice.start,
            },
        )
        with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
            capturer.capture(layer_id=0, topk_ids=physical_routes)

    physical_output = (
        batch.input_ids.to(torch.int64) * 10 + batch.positions
    ).unsqueeze(1)
    compact_staged_rows(physical_output, rows)
    restore_staged_inputs(batch, (blocks,), slots, rows)
    routed = capturer.get_routed_experts(slots, num_tokens, rows)

    torch.testing.assert_close(
        physical_output[:num_tokens, 0],
        logical_ids.to(torch.int64) * 10 + logical_positions,
    )
    torch.testing.assert_close(routed.routing_data[:, 0, :], expected_routes)
    torch.testing.assert_close(routed.slot_mapping, logical_slots[0])
    torch.testing.assert_close(batch.input_ids[:num_tokens], logical_ids)
    torch.testing.assert_close(batch.positions[:num_tokens], logical_positions)
    torch.testing.assert_close(blocks[:num_tokens], logical_blocks)


def test_routed_experts_capturer_dp_naive_concatenated_all_ranks():
    """N == sum(num_tokens_dp): slice this rank's segment from concatenated topk."""
    capturer = _capturer_with_buffer(dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp),
        additional_kwargs={},
    )
    # Concatenated order: rank0 rows then rank1 rows.
    topk = torch.tensor(
        [[0, 1], [2, 3], [10, 11], [12, 13], [14, 15]], dtype=torch.int32
    )
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    want = topk[2:5]
    assert torch.equal(capturer.device_buffer[:3, 0, :], want)


def test_routed_experts_capturer_dp_modular_local_tokens():
    """N == token_num_per_dp: topk is already local to this DP rank."""
    capturer = _capturer_with_buffer(dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp),
        additional_kwargs={},
    )
    topk = torch.tensor([[10, 11], [12, 13], [14, 15]], dtype=torch.int32)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk)


def test_routed_experts_capturer_dp_unexpected_batch_raises():
    """Mismatch between topk batch dim and DP layout: fail fast."""
    capturer = _capturer_with_buffer(dp_rank=0)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp),
        additional_kwargs={},
    )
    # total=5, local=2: n=1 matches neither naive (5) nor modular (2).
    topk = torch.tensor([[1, 2]], dtype=torch.int32)
    with (
        patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx),
        pytest.raises(AssertionError, match="unexpected topk_ids batch dim"),
    ):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert capturer.device_buffer[0, 0, 0].item() == -1


def test_routed_experts_attention_group_is_shared_and_fail_closed():
    """Both sides key routing data by this gid, so it must skip non-full-attention
    groups rather than defaulting to 0, and fail closed when none exists."""
    common = dict(num_kv_heads=1, head_size=1, dtype=torch.float32)
    config = SimpleNamespace(
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["swa_layer"],
                SlidingWindowMLASpec(block_size=4, sliding_window=8, **common),
            ),
            KVCacheGroupSpec(["full_layer"], FullAttentionSpec(block_size=4, **common)),
        ]
    )
    assert get_routed_experts_attn_gid(config) == 1

    with pytest.raises(ValueError, match="requires a full-attention KV cache group"):
        get_routed_experts_attn_gid(SimpleNamespace(kv_cache_groups=[]))


def test_routed_experts_attention_group_unwraps_uniform_type_specs():
    """DeepSeek-V4-shaped groups wrap their specs in ``UniformTypeKVCacheSpecs``.

    The wrapper is not a ``FullAttentionSpec``, so a bare isinstance check finds
    no group and fails closed on every worker. Unwrap semantics themselves are
    covered by ``test_is_full_attention_spec_*`` in tests/v1/core.
    """
    common = dict(num_kv_heads=1, head_size=1, dtype=torch.float32)
    swa_spec = SlidingWindowMLASpec(block_size=4, sliding_window=8, **common)
    mla_spec = MLAAttentionSpec(block_size=4, **common)
    config = SimpleNamespace(
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["swa_layer"],
                UniformTypeKVCacheSpecs(
                    block_size=4, kv_cache_specs={"swa_layer": swa_spec}
                ),
            ),
            KVCacheGroupSpec(
                ["mla_layer"],
                UniformTypeKVCacheSpecs(
                    block_size=4, kv_cache_specs={"mla_layer": mla_spec}
                ),
            ),
        ]
    )

    assert get_routed_experts_attn_gid(config) == 1

    swa_only = SimpleNamespace(kv_cache_groups=[config.kv_cache_groups[0]])
    with pytest.raises(ValueError, match="requires a full-attention KV cache group"):
        get_routed_experts_attn_gid(swa_only)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mrv2_async_output_returns_existing_routed_experts_field():
    from vllm.v1.outputs import ModelRunnerOutput, RoutedExpertsTensors
    from vllm.v1.worker.gpu.async_utils import AsyncOutput
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    routed_experts = RoutedExpertsTensors(
        routing_data=torch.arange(6, dtype=torch.int32, device="cuda").reshape(3, 1, 2),
        slot_mapping=torch.tensor([11, 12, 13], device="cuda"),
    )
    num_sampled = torch.tensor([1], dtype=torch.int32, device="cuda")
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.tensor([[1]], device="cuda"),
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=num_sampled,
        num_rejected=torch.tensor([0], dtype=torch.int32, device="cuda"),
    )
    output = AsyncOutput(
        model_runner_output=ModelRunnerOutput(req_ids=["req"], req_id_to_index={}),
        sampler_output=sampler_output,
        num_sampled_tokens=num_sampled,
        main_stream=torch.cuda.current_stream(),
        copy_stream=torch.cuda.Stream(),
        routed_experts=routed_experts,
    ).get_output()

    assert output.routed_experts is not None
    assert output.routed_experts.routing_data[:, 0, 0].tolist() == [0, 2, 4]
    assert output.routed_experts.slot_mapping.tolist() == [11, 12, 13]


@pytest.mark.parametrize("rank", [0, 1])
def test_all_tp_ranks_initialize_capture(monkeypatch, rank):
    pytest.importorskip("vllm.vllm_flash_attn", exc_type=ImportError)
    import vllm.v1.worker.gpu.model_runner as model_runner

    capturer = Mock()
    constructor = Mock(return_value=capturer)
    bind = Mock()
    monkeypatch.setattr(model_runner, "RoutedExpertsCapturer", constructor)
    monkeypatch.setattr(model_runner, "bind_routed_experts_capturer", bind)

    runner = model_runner.GPUModelRunner.__new__(model_runner.GPUModelRunner)
    runner.max_num_tokens = 32
    runner.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(rank=rank))
    runner.kv_cache_config = SimpleNamespace()
    runner.model = Mock()

    runner.init_routed_experts_capturer()

    constructor.assert_called_once_with(
        max_num_batched_tokens=32,
        vllm_config=runner.vllm_config,
        kv_cache_config=runner.kv_cache_config,
    )
    bind.assert_called_once_with(runner.model, capturer)
    assert runner.routed_experts_capturer is capturer


def test_v2_model_runner_accepts_routed_experts(monkeypatch):
    monkeypatch.setattr("importlib.metadata.entry_points", lambda **_: ())
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            enable_return_routed_experts=True,
            use_mla=False,
            logits_processors=None,
            enable_prompt_embeds=False,
        ),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
            distributed_executor_backend=None,
            pipeline_parallel_size=1,
            enable_dbo=False,
            use_ubatching=False,
            enable_elastic_ep=False,
        ),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.NONE,
            pass_config=SimpleNamespace(enable_sp=False),
        ),
        cache_config=SimpleNamespace(
            kv_sharing_fast_prefill=False,
            mamba_cache_mode="none",
        ),
        ec_transfer_config=None,
    )

    unsupported = VllmConfig._get_v2_model_runner_unsupported_features(config)

    assert "routed experts capture" not in unsupported
