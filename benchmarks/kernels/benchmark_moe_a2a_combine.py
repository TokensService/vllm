#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark the FlashInfer one-sided MoE all-to-all combine leg in BF16 vs FP8.

Measures the effect of VLLM_FLASHINFER_MOE_A2A_LOW_PRECISION_COMBINE, which
transmits expert outputs as fp8_e4m3 instead of bf16, halving the bytes that
cross NVLink on the combine leg.

The one-sided kernel requires a fresh dispatch before every combine, so combine
cannot be timed alone. This times dispatch-only and dispatch+combine separately
and reports the difference as the combine cost. Dispatch is identical in both
configurations, so the delta isolates the combine leg.

Runs bf16 fully, then fp8. If you suspect clock or thermal drift, run it twice
and compare -- the configurations are independent.

Usage:
    torchrun --nproc-per-node=8 benchmarks/kernels/benchmark_moe_a2a_combine.py

    # DeepSeek-R1 shapes at decode-like token counts
    torchrun --nproc-per-node=8 benchmarks/kernels/benchmark_moe_a2a_combine.py \
        --hidden-size 7168 --top-k 8 --num-experts 256 \
        --tokens-per-rank 8 16 32 64 128
"""

import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument(
        "--tokens-per-rank",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128],
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    return parser.parse_args()


def init_distributed() -> tuple[int, int, int]:
    """Set up vLLM's DP-based parallel state, mirroring the mnnvl tests."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.parallel import ParallelConfig
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = int(os.environ["MASTER_PORT"])

    torch.accelerator.set_device_index(local_rank)

    vllm_config = VllmConfig()
    vllm_config.parallel_config = ParallelConfig(
        data_parallel_size=world_size,
        data_parallel_rank=rank,
        # All ranks must agree on this port, and it must not collide with the
        # rendezvous port torchrun already owns.
        _data_parallel_master_port_list=[master_port + 1],
    )
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,  # tp * pp = 1; each rank is its own DP rank
            rank=0,
            distributed_init_method=f"tcp://{master_addr}:{master_port}",
            local_rank=local_rank,
        )
        ensure_model_parallel_initialized(1, 1)

    return rank, world_size, local_rank


def make_manager(low_precision: bool, args):
    """Build a one-sided manager with the fp8-combine flag set or cleared."""
    from vllm.distributed.device_communicators.all2all import (
        FlashInferNVLinkOneSidedManager,
    )
    from vllm.distributed.parallel_state import get_dp_group

    # envs.__getattr__ is uncached by default, so initialize() picks this up.
    os.environ["VLLM_FLASHINFER_MOE_A2A_LOW_PRECISION_COMBINE"] = (
        "1" if low_precision else "0"
    )

    manager = FlashInferNVLinkOneSidedManager(get_dp_group().cpu_group)
    manager.initialize(
        max_num_tokens=max(args.tokens_per_rank),
        top_k=args.top_k,
        num_experts=args.num_experts,
        hidden_size=args.hidden_size,
        # nvfp4 activations plus their fp8 block scales, matching the payload
        # layout the mnnvl tests exercise.
        x_bytes_per_token=args.hidden_size // 2,
        x_sf_bytes_per_token=args.hidden_size // 16,
    )
    assert manager.low_precision_combine == low_precision, (
        f"requested low_precision={low_precision} but the manager resolved to "
        f"{manager.low_precision_combine}; the installed FlashInfer build "
        "likely has no `use_low_precision` parameter on MoeAlltoAll.combine()"
    )
    return manager


def make_inputs(args, tokens: int, world_size: int, rank: int, device):
    torch.manual_seed(rank + 42)
    hidden = args.hidden_size
    x = torch.randint(0, 256, (tokens, hidden // 2), device=device, dtype=torch.uint8)
    x_sf = torch.randint(
        0, 256, (tokens, hidden // 16), device=device, dtype=torch.uint8
    )
    topk_ids = torch.randint(
        0, args.num_experts, (tokens, args.top_k), device=device, dtype=torch.int32
    )
    topk_weights = torch.rand(tokens, args.top_k, device=device, dtype=torch.float32)
    expert_output = torch.ones(
        world_size, tokens, hidden, device=device, dtype=torch.bfloat16
    )
    return [x, x_sf, topk_ids, topk_weights], topk_ids, expert_output


def check_correctness(
    manager, payloads, topk_ids, expert_output, args, tokens, world_size, device
) -> None:
    """Combine of all-ones expert output must equal the distinct-rank count.

    Integers 1..top_k are exactly representable in fp8_e4m3, so this holds for
    both the bf16 and fp8 transports. A workspace/dtype mismatch corrupts the
    buffer silently, so this must pass before any timing is trusted.
    """
    manager.moe_alltoall.dispatch(
        token_selected_experts=topk_ids,
        input_payloads=payloads,
        runtime_max_tokens_per_rank=tokens,
    )
    output = torch.empty(tokens, args.hidden_size, device=device, dtype=torch.bfloat16)
    manager.combine_into(
        payload=expert_output,
        runtime_max_tokens_per_rank=tokens,
        output=output,
    )
    experts_per_rank = args.num_experts // world_size
    expert_ranks = topk_ids // experts_per_rank
    num_distinct = torch.tensor(
        [len(set(row.tolist())) for row in expert_ranks],
        device=device,
        dtype=torch.bfloat16,
    ).unsqueeze(1)
    torch.testing.assert_close(output, num_distinct.expand_as(output))


def time_op(fn, warmup: int, iters: int, group) -> float:
    """Median over iterations of the per-iteration max across ranks.

    A rank-local time is not a distributed latency: a collective is only done
    when its slowest participant is, so each sample is MAX-reduced across ranks
    before the median. Collectives are scoped to the DP group because the
    default process group here is per-rank and would make them no-ops.
    """
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()

    samples = []
    for _ in range(iters):
        dist.barrier(group=group)
        torch.accelerator.synchronize()
        start = time.perf_counter()
        fn()
        torch.accelerator.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)  # microseconds

    local = torch.tensor(samples, device="cuda", dtype=torch.float64)
    dist.all_reduce(local, op=dist.ReduceOp.MAX, group=group)
    return statistics.median(local.tolist())


def benchmark_tokens(manager, args, tokens, world_size, rank, device, group):
    """Return (dispatch_us, dispatch_plus_combine_us) for one token count."""
    payloads, topk_ids, expert_output = make_inputs(
        args, tokens, world_size, rank, device
    )
    check_correctness(
        manager, payloads, topk_ids, expert_output, args, tokens, world_size, device
    )
    output = torch.empty(tokens, args.hidden_size, device=device, dtype=torch.bfloat16)

    def dispatch_only():
        manager.moe_alltoall.dispatch(
            token_selected_experts=topk_ids,
            input_payloads=payloads,
            runtime_max_tokens_per_rank=tokens,
        )

    def dispatch_and_combine():
        manager.moe_alltoall.dispatch(
            token_selected_experts=topk_ids,
            input_payloads=payloads,
            runtime_max_tokens_per_rank=tokens,
        )
        manager.combine_into(
            payload=expert_output,
            runtime_max_tokens_per_rank=tokens,
            output=output,
        )

    dispatch_us = time_op(dispatch_only, args.warmup, args.iters, group)
    pair_us = time_op(dispatch_and_combine, args.warmup, args.iters, group)
    return dispatch_us, pair_us


def main() -> None:
    args = parse_args()

    from vllm.distributed.parallel_state import get_dp_group

    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    group = get_dp_group().device_group

    results: dict[tuple[bool, int], tuple[float, float]] = {}
    for low_precision in (False, True):
        manager = make_manager(low_precision, args)
        try:
            for tokens in args.tokens_per_rank:
                results[low_precision, tokens] = benchmark_tokens(
                    manager, args, tokens, world_size, rank, device, group
                )
                dist.barrier(group=group)
        finally:
            manager.cleanup()
        dist.barrier(group=group)

    if rank != 0:
        return

    import flashinfer

    gpu_name = torch.get_device_module().get_device_name(0)
    print(f"\nworld_size={world_size} gpu={gpu_name}")
    print(f"flashinfer={flashinfer.__version__} torch={torch.__version__}")
    print(
        f"hidden={args.hidden_size} top_k={args.top_k} "
        f"experts={args.num_experts} warmup={args.warmup} iters={args.iters}\n"
    )
    header = (
        f"{'tokens':>7} {'bf16 comb':>11} {'fp8 comb':>11} "
        f"{'speedup':>8} {'bf16 GB/s':>10} {'fp8 GB/s':>10}"
    )
    print(header)
    print("-" * len(header))
    for tokens in args.tokens_per_rank:
        bf16 = results[False, tokens][1] - results[False, tokens][0]
        fp8 = results[True, tokens][1] - results[True, tokens][0]
        payload_elems = world_size * tokens * args.hidden_size
        bf16_gbs = payload_elems * 2 / bf16 / 1e3 if bf16 > 0 else float("nan")
        fp8_gbs = payload_elems * 1 / fp8 / 1e3 if fp8 > 0 else float("nan")
        speedup = bf16 / fp8 if fp8 > 0 else float("nan")
        print(
            f"{tokens:>7} {bf16:>10.1f}us {fp8:>10.1f}us "
            f"{speedup:>7.2f}x {bf16_gbs:>10.1f} {fp8_gbs:>10.1f}"
        )
    print("\ncombine time = (dispatch+combine) - (dispatch only)")
    print("GB/s counts payload bytes only, not fabric hops")


if __name__ == "__main__":
    main()
