# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pluggable KV cache config builder resolution."""

from typing import TYPE_CHECKING

from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheLayout,
        KVCacheSpec,
    )


class KVCacheConfigBuilder:
    """Active KV cache config builder, resolved per call.

    Resolution priority is owned by the platform hook
    (:meth:`vllm.platforms.interface.Platform.get_kv_cache_config_builder_cls`);
    this class resolves it on every delegation and forwards the call to a
    fresh builder instance. Builders are stateless (every method takes
    ``vllm_config``), so there is nothing to cache and engines with different
    configs in the same process always get the correct builder.
    """

    @classmethod
    def _resolve(cls, vllm_config: "VllmConfig") -> "DefaultKVCacheConfigBuilder":
        from vllm.platforms import current_platform

        builder_cls = resolve_obj_by_qualname(
            current_platform.get_kv_cache_config_builder_cls(vllm_config)
        )
        return builder_cls()

    @classmethod
    def get_kv_cache_configs(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_specs: list[dict[str, "KVCacheSpec"]],
        available_memory: list[int],
    ) -> list["KVCacheConfig"]:
        """Generate the full KV cache configurations for every worker.

        The main entry point: takes the per-worker KV cache specs and the
        memory available on each worker, runs the whole planning pipeline
        (merge specs, group layers, project to workers, auto-fit
        max_model_len, admission checks, per-worker layouts, min-blocks
        convergence), and returns one ready-to-allocate
        :class:`~vllm.v1.kv_cache_interface.KVCacheConfig` per worker.

        Consumed by ``vllm/v1/engine/core.py``.
        """
        return cls._resolve(vllm_config).get_kv_cache_configs(
            vllm_config, kv_cache_specs, available_memory
        )

    @classmethod
    def get_profiling_kv_cache_config(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: dict[str, "KVCacheSpec"],
        min_blocks: int,
    ) -> "KVCacheConfig":
        """Build the smallest KV cache config for CUDA graph profiling.

        The config is sized to ``min_blocks`` blocks (at least one block per
        captured sequence) so the profiling run can capture every graph
        without consuming real KV cache memory. Custom builders that change
        grouping or layouts only need this method to keep working; they do
        not need to re-implement the profiling choreography.

        Consumed by the cudagraph profiling workers
        (``cudagraph_utils.py`` and ``gpu_model_runner.py``).
        """
        return cls._resolve(vllm_config).get_profiling_kv_cache_config(
            vllm_config, kv_cache_spec, min_blocks
        )

    @classmethod
    def check_enough_kv_cache_memory(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: dict[str, "KVCacheSpec"],
        available_memory: int,
    ) -> None:
        """Raise if the KV cache cannot hold a single request.

        Early-startup admission check: verifies that ``available_memory``
        fits one request of ``max_model_len`` under the given specs,
        accounting for the null block reserved by the block pool. Runs
        before :meth:`get_kv_cache_configs`, when no full config exists
        yet.

        Raises:
            ValueError: If the memory cannot hold one request.

        """
        return cls._resolve(vllm_config).check_enough_kv_cache_memory(
            vllm_config, kv_cache_spec, available_memory
        )

    @classmethod
    def get_kv_cache_groups(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: dict[str, "KVCacheSpec"],
    ) -> list["KVCacheGroupSpec"]:
        """Split a worker's layers into logical KV cache groups."""
        return cls._resolve(vllm_config).get_kv_cache_groups(vllm_config, kv_cache_spec)

    @classmethod
    def get_kv_cache_config_from_groups(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_groups: list["KVCacheGroupSpec"],
        available_memory: int,
        *,
        num_gpu_blocks_override: int | None = None,
    ) -> "KVCacheConfig":
        """Turn groups and available memory into a physical layout.

        ``num_gpu_blocks_override`` forces the block count without touching
        ``vllm_config.cache_config``; it is what the profiling path uses to
        size a minimal cache.
        """
        return cls._resolve(vllm_config).get_kv_cache_config_from_groups(
            vllm_config,
            kv_cache_groups,
            available_memory,
            num_gpu_blocks_override=num_gpu_blocks_override,
        )

    @classmethod
    def validate_kv_cache_layout(
        cls,
        vllm_config: "VllmConfig",
        layout: "KVCacheLayout",
        kv_cache_groups: list["KVCacheGroupSpec"],
    ) -> None:
        """Validate that ``layout`` can express the groups' packing.

        Raises:
            ValueError: If the layout cannot express the groups.

        """
        return cls._resolve(vllm_config).validate_kv_cache_layout(
            vllm_config, layout, kv_cache_groups
        )

    @classmethod
    def get_pool_bytes_per_block(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_groups: list["KVCacheGroupSpec"],
    ) -> int:
        """Return the divisor converting available memory into num_blocks."""
        return cls._resolve(vllm_config).get_pool_bytes_per_block(
            vllm_config, kv_cache_groups
        )

    @classmethod
    def may_override_num_blocks(
        cls,
        vllm_config: "VllmConfig",
        num_blocks: int,
        num_gpu_blocks_override: int | None = None,
    ) -> int:
        """Apply ``num_gpu_blocks_override`` if set."""
        return cls._resolve(vllm_config).may_override_num_blocks(
            vllm_config, num_blocks, num_gpu_blocks_override
        )

    @classmethod
    def get_max_memory_usage_bytes_from_groups(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_groups: list["KVCacheGroupSpec"],
    ) -> int:
        """Calculate maximum memory usage in bytes from KV cache groups."""
        return cls._resolve(vllm_config).get_max_memory_usage_bytes_from_groups(
            vllm_config, kv_cache_groups
        )

    @classmethod
    def estimate_max_model_len_from_groups(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_groups: list["KVCacheGroupSpec"],
        available_memory: int,
    ) -> int:
        """Estimate the largest model length that fits ``available_memory``.

        Binary search over model lengths; returns 0 if even one token
        cannot fit.
        """
        return cls._resolve(vllm_config).estimate_max_model_len_from_groups(
            vllm_config, kv_cache_groups, available_memory
        )

    @classmethod
    def auto_fit_max_model_len(
        cls,
        vllm_config: "VllmConfig",
        projected_groups_per_worker: list[list["KVCacheGroupSpec"]],
        available_memory: list[int],
    ) -> None:
        """When max_model_len is -1, fit it to the available memory."""
        return cls._resolve(vllm_config).auto_fit_max_model_len(
            vllm_config, projected_groups_per_worker, available_memory
        )
