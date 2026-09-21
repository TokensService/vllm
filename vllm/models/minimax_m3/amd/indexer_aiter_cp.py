# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 context-parallel AITER indexer for ROCm.

Each TP rank scores its own round-robin shard of KV blocks using the fp8 MFMA
``pa_sparse_block_score_decode`` kernel, then a MAX allreduce across the TP
group reconstructs the full score matrix. ``pa_sparse_block_topk`` runs on the
allreduced result.

Enabled by ``VLLM_ROCM_MINIMAX_INDEXER_CP=1`` when the AITER indexer is
selected (fp8 index cache, gfx950). Prefill is unchanged; delegates to base.

Block ownership: ``owned_block = local_index * world_size + rank`` — each rank
scores a disjoint subset of the global block table, filling the rest of the
score tensor with ``-inf`` before the allreduce.
"""

import torch

import vllm.envs as envs
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.minimax_m3.amd.indexer_aiter import (
    MiniMaxM3IndexerAiterImpl,
    select_aiter_indexer_impl_cls,
)
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerMetadata,
)

logger = init_logger(__name__)


class MiniMaxM3IndexerAiterCPImpl(MiniMaxM3IndexerAiterImpl):
    """AITER indexer with context-parallel decode scoring for ROCm.

    Decode: each rank scores its 1/world_size shard of blocks → MAX allreduce
    → top-k on full scores. Prefill: unchanged, delegates to base AITER impl.
    """

    def forward(
        self,
        index_query: torch.Tensor,
        *,
        decode_page16_block_table: torch.Tensor | None = None,
        prefill_page16_block_table: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        from vllm._aiter_ops import rocm_aiter_ops

        pa_sparse_block_score_decode = (
            rocm_aiter_ops.pa_sparse_block_score_decode
        )
        pa_sparse_block_topk = rocm_aiter_ops.pa_sparse_block_topk

        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None, None

        md = attn_metadata[self.index_cache.prefix]
        assert isinstance(md, MiniMaxM3IndexerMetadata)
        num_tokens = md.num_actual_tokens
        nd = md.num_decode_tokens
        iq = index_query[:num_tokens].view(
            -1, self.num_index_heads, self.index_head_dim
        )
        kv = self.index_cache.kv_cache

        buf = self.topk_indices_buffer
        if buf is None:
            buf = torch.empty(
                (self.num_index_heads, num_tokens, self.topk_blocks),
                dtype=torch.int32,
                device=iq.device,
            )

        decode_topk: torch.Tensor | None = None
        prefill_topk: torch.Tensor | None = None

        if md.num_decodes > 0:
            d = md.decode
            assert d is not None
            assert decode_page16_block_table is not None

            world_size = get_tensor_model_parallel_world_size()
            rank = get_tp_group().rank_in_group
            tp_group = get_tp_group().device_group

            max_blocks = d.block_table.shape[1]
            # Build shard block table: owned columns only (round-robin).
            # owned_cols[i] = local_i * world_size + rank
            local_blocks = (max_blocks + world_size - 1) // world_size
            owned_cols = torch.arange(
                rank, max_blocks, world_size,
                dtype=d.block_table.dtype,
                device=d.block_table.device,
            )  # shape: [local_blocks]

            # Shard block table: gather owned columns for each request.
            shard_bt = d.block_table[:, owned_cols]  # [batch, local_blocks]

            # Allocate full score tensor, fill with -inf so unowned
            # positions become neutral elements for MAX allreduce.
            score = self._new_score(nd, d.max_seq_len)
            score.fill_(float("-inf"))

            # Score only owned blocks — kernel writes into owned positions.
            pa_sparse_block_score_decode(
                iq[:nd],
                kv,
                score,
                shard_bt,
                d.seq_lens,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                query_len=d.decode_query_len,
                max_seq_len=d.max_seq_len,
            )

            # MAX allreduce: each rank's -inf positions become the
            # owning rank's scores. Result is globally correct.
            import torch.distributed as dist
            dist.all_reduce(score, op=dist.ReduceOp.MAX, group=tp_group)

            decode_topk = buf[:, :nd, :]
            sparse_bt, sparse_ctx = self._table_rows(0, nd)
            pa_sparse_block_topk(
                score,
                decode_topk,
                decode_page16_block_table,
                d.seq_lens,
                sparse_bt,
                sparse_ctx,
                max_seq_len=d.max_seq_len,
                block_size=self.block_size,
                query_len=d.decode_query_len,
                num_kv_heads=self.num_kv_heads,
                pages_per_block=self.pages_per_block,
            )

        if md.num_prefills > 0:
            # Prefill delegates to the base AITER impl unchanged.
            _, prefill_topk = super().forward(
                index_query,
                decode_page16_block_table=None,
                prefill_page16_block_table=prefill_page16_block_table,
            )

        return decode_topk, prefill_topk


def select_aiter_cp_indexer_impl_cls(
    **kwargs,
) -> type[MiniMaxM3IndexerAiterCPImpl] | None:
    """Return CP AITER impl if enabled and base AITER impl is available."""
    if not (
        get_tensor_model_parallel_world_size() > 1
        and envs.VLLM_ROCM_MINIMAX_INDEXER_CP
    ):
        return None
    base = select_aiter_indexer_impl_cls(**kwargs)
    if base is None:
        return None
    logger.info_once(
        "MiniMax M3 indexer: selected AITER CP (context-parallel, ROCm) "
        "[topk_blocks=%d, tp=%d]",
        kwargs.get("topk_blocks", "?"),
        get_tensor_model_parallel_world_size(),
    )
    return MiniMaxM3IndexerAiterCPImpl
