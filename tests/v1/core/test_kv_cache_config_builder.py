# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for KVCacheConfigBuilder resolution."""

from unittest.mock import MagicMock, patch

from vllm.platforms import Platform
from vllm.v1.core.kv_cache_config_builder import KVCacheConfigBuilder
from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder


def _make_vllm_config(builder_cls_path: str | None = None) -> MagicMock:
    """Create a minimal mock VllmConfig for builder resolution tests."""
    cfg = MagicMock()
    cfg.model_config.kv_cache_config_builder_cls = builder_cls_path
    return cfg


CUSTOM_PATH = "tests.v1.core.test_kv_cache_config_builder.CustomBuilder"
DEFAULT_PATH = "vllm.v1.core.kv_cache_planning.DefaultKVCacheConfigBuilder"


class CustomBuilder(DefaultKVCacheConfigBuilder):
    """A test builder subclass."""

    pass


class TestPlatformHookResolution:
    """The platform hook owns the resolution priority."""

    def test_default_hook_prefers_model_declaration(self):
        cfg = _make_vllm_config(builder_cls_path=CUSTOM_PATH)
        assert Platform.get_kv_cache_config_builder_cls(cfg) == CUSTOM_PATH

    def test_default_hook_falls_back_to_default_builder(self):
        cfg = _make_vllm_config(builder_cls_path=None)
        assert Platform.get_kv_cache_config_builder_cls(cfg) == DEFAULT_PATH


class TestBuilderResolution:
    @patch("vllm.platforms.current_platform")
    def test_resolves_default_builder(self, mock_platform):
        mock_platform.get_kv_cache_config_builder_cls.return_value = DEFAULT_PATH
        cfg = _make_vllm_config()
        assert type(KVCacheConfigBuilder._resolve(cfg)) is DefaultKVCacheConfigBuilder

    @patch("vllm.platforms.current_platform")
    def test_model_declared_builder(self, mock_platform):
        mock_platform.get_kv_cache_config_builder_cls.return_value = CUSTOM_PATH
        cfg = _make_vllm_config()
        assert isinstance(KVCacheConfigBuilder._resolve(cfg), CustomBuilder)

    @patch("vllm.platforms.current_platform")
    def test_resolves_fresh_instance_per_call(self, mock_platform):
        mock_platform.get_kv_cache_config_builder_cls.return_value = CUSTOM_PATH
        cfg = _make_vllm_config()
        assert KVCacheConfigBuilder._resolve(cfg) is not KVCacheConfigBuilder._resolve(
            cfg
        )

    @patch("vllm.platforms.current_platform")
    def test_resolution_per_config(self, mock_platform):
        # Each config resolves the builder the platform hook selects for it.
        mock_platform.get_kv_cache_config_builder_cls.side_effect = (
            lambda vllm_config: vllm_config.model_config.kv_cache_config_builder_cls
            or DEFAULT_PATH
        )
        custom = KVCacheConfigBuilder._resolve(_make_vllm_config(CUSTOM_PATH))
        default = KVCacheConfigBuilder._resolve(_make_vllm_config(None))
        assert isinstance(custom, CustomBuilder)
        assert type(default) is DefaultKVCacheConfigBuilder

    @patch("vllm.platforms.current_platform")
    def test_entry_points_delegate_to_resolved_builder(self, mock_platform):
        mock_platform.get_kv_cache_config_builder_cls.return_value = CUSTOM_PATH
        cfg = _make_vllm_config()
        with (
            patch.object(CustomBuilder, "get_kv_cache_configs", return_value=[]) as g,
            patch.object(CustomBuilder, "get_kv_cache_groups", return_value=[]) as h,
        ):
            assert KVCacheConfigBuilder.get_kv_cache_configs(cfg, [], [0]) == []
            g.assert_called_once()
            assert KVCacheConfigBuilder.get_kv_cache_groups(cfg, {}) == []
            h.assert_called_once()


class TestPlatformCustomPriority:
    """A vendor platform can override the hook to customize priority."""

    def test_platform_builder_wins_over_model_declaration(self):
        class PlatformFirstPlatform(Platform):
            @classmethod
            def get_kv_cache_config_builder_cls(cls, vllm_config):
                return CUSTOM_PATH

        cfg = _make_vllm_config(builder_cls_path=CUSTOM_PATH)
        # Model declares CustomBuilder too; the platform forces it anyway.
        assert PlatformFirstPlatform.get_kv_cache_config_builder_cls(cfg) == CUSTOM_PATH
        with patch("vllm.platforms.current_platform", PlatformFirstPlatform):
            assert isinstance(KVCacheConfigBuilder._resolve(cfg), CustomBuilder)

    def test_platform_delegates_to_model_declaration(self):
        class ModelFirstPlatform(Platform):
            @classmethod
            def get_kv_cache_config_builder_cls(cls, vllm_config):
                model_path = vllm_config.model_config.kv_cache_config_builder_cls
                return model_path or DEFAULT_PATH

        cfg = _make_vllm_config(builder_cls_path=CUSTOM_PATH)
        assert ModelFirstPlatform.get_kv_cache_config_builder_cls(cfg) == CUSTOM_PATH


class TestDefaultBuilderDelegation:
    """Without a custom builder, the methods hit the default builder, which
    implements the planning steps in :mod:`kv_cache_planning`."""

    @patch("vllm.platforms.current_platform")
    @patch.object(DefaultKVCacheConfigBuilder, "get_kv_cache_groups")
    def test_get_kv_cache_groups_delegates_to_default(self, mock_impl, mock_platform):
        mock_platform.get_kv_cache_config_builder_cls.return_value = DEFAULT_PATH
        cfg = _make_vllm_config()
        spec = {"layer": MagicMock()}
        assert (
            KVCacheConfigBuilder.get_kv_cache_groups(cfg, spec)
            is mock_impl.return_value
        )
        mock_impl.assert_called_once_with(cfg, spec)

    @patch("vllm.platforms.current_platform")
    @patch.object(DefaultKVCacheConfigBuilder, "get_kv_cache_config_from_groups")
    def test_get_kv_cache_config_from_groups_delegates_to_default(
        self, mock_impl, mock_platform
    ):
        mock_platform.get_kv_cache_config_builder_cls.return_value = DEFAULT_PATH
        cfg = _make_vllm_config()
        groups = [MagicMock()]
        result = KVCacheConfigBuilder.get_kv_cache_config_from_groups(cfg, groups, 0)
        assert result is mock_impl.return_value
        mock_impl.assert_called_once_with(cfg, groups, 0, num_gpu_blocks_override=None)

    @patch("vllm.platforms.current_platform")
    @patch.object(DefaultKVCacheConfigBuilder, "get_kv_cache_configs")
    def test_get_kv_cache_configs_delegates_to_default(self, mock_impl, mock_platform):
        mock_platform.get_kv_cache_config_builder_cls.return_value = DEFAULT_PATH
        cfg = _make_vllm_config()
        specs, memory = [MagicMock()], [0]
        result = KVCacheConfigBuilder.get_kv_cache_configs(cfg, specs, memory)
        assert result is mock_impl.return_value
        mock_impl.assert_called_once_with(cfg, specs, memory)

    @patch("vllm.platforms.current_platform")
    @patch.object(DefaultKVCacheConfigBuilder, "get_profiling_kv_cache_config")
    def test_get_profiling_kv_cache_config_delegates_to_default(
        self, mock_impl, mock_platform
    ):
        mock_platform.get_kv_cache_config_builder_cls.return_value = DEFAULT_PATH
        cfg = _make_vllm_config()
        spec = {"layer": MagicMock()}
        result = KVCacheConfigBuilder.get_profiling_kv_cache_config(cfg, spec, 4)
        assert result is mock_impl.return_value
        mock_impl.assert_called_once_with(cfg, spec, 4)

    @patch("vllm.platforms.current_platform")
    @patch.object(DefaultKVCacheConfigBuilder, "check_enough_kv_cache_memory")
    def test_check_enough_kv_cache_memory_delegates_to_default(
        self, mock_impl, mock_platform
    ):
        mock_platform.get_kv_cache_config_builder_cls.return_value = DEFAULT_PATH
        cfg = _make_vllm_config()
        spec = {"layer": MagicMock()}
        result = KVCacheConfigBuilder.check_enough_kv_cache_memory(cfg, spec, 0)
        assert result is mock_impl.return_value
        mock_impl.assert_called_once_with(cfg, spec, 0)

    @patch("vllm.platforms.current_platform")
    @patch.object(DefaultKVCacheConfigBuilder, "get_pool_bytes_per_block")
    def test_get_pool_bytes_per_block_delegates_to_default(
        self, mock_impl, mock_platform
    ):
        mock_platform.get_kv_cache_config_builder_cls.return_value = DEFAULT_PATH
        cfg = _make_vllm_config()
        groups = [MagicMock()]
        assert (
            KVCacheConfigBuilder.get_pool_bytes_per_block(cfg, groups)
            is mock_impl.return_value
        )
        mock_impl.assert_called_once_with(cfg, groups)

    @patch("vllm.platforms.current_platform")
    @patch.object(DefaultKVCacheConfigBuilder, "validate_kv_cache_layout")
    def test_validate_kv_cache_layout_delegates_to_default(
        self, mock_impl, mock_platform
    ):
        mock_platform.get_kv_cache_config_builder_cls.return_value = DEFAULT_PATH
        cfg = _make_vllm_config()
        groups = [MagicMock()]
        layout = MagicMock()
        result = KVCacheConfigBuilder.validate_kv_cache_layout(cfg, layout, groups)
        assert result is mock_impl.return_value
        mock_impl.assert_called_once_with(cfg, layout, groups)

    @patch("vllm.platforms.current_platform")
    @patch.object(DefaultKVCacheConfigBuilder, "get_max_memory_usage_bytes_from_groups")
    def test_get_max_memory_usage_bytes_from_groups_delegates_to_default(
        self, mock_impl, mock_platform
    ):
        mock_platform.get_kv_cache_config_builder_cls.return_value = DEFAULT_PATH
        cfg = _make_vllm_config()
        groups = [MagicMock()]
        result = KVCacheConfigBuilder.get_max_memory_usage_bytes_from_groups(
            cfg, groups
        )
        assert result is mock_impl.return_value
        mock_impl.assert_called_once_with(cfg, groups)

    @patch("vllm.platforms.current_platform")
    @patch.object(DefaultKVCacheConfigBuilder, "estimate_max_model_len_from_groups")
    def test_estimate_max_model_len_from_groups_delegates_to_default(
        self, mock_impl, mock_platform
    ):
        mock_platform.get_kv_cache_config_builder_cls.return_value = DEFAULT_PATH
        cfg = _make_vllm_config()
        groups = [MagicMock()]
        result = KVCacheConfigBuilder.estimate_max_model_len_from_groups(cfg, groups, 0)
        assert result is mock_impl.return_value
        mock_impl.assert_called_once_with(cfg, groups, 0)

    @patch("vllm.platforms.current_platform")
    @patch.object(DefaultKVCacheConfigBuilder, "auto_fit_max_model_len")
    def test_auto_fit_max_model_len_delegates_to_default(
        self, mock_impl, mock_platform
    ):
        mock_platform.get_kv_cache_config_builder_cls.return_value = DEFAULT_PATH
        cfg = _make_vllm_config()
        groups, memory = [MagicMock()], [0]
        result = KVCacheConfigBuilder.auto_fit_max_model_len(cfg, groups, memory)
        assert result is mock_impl.return_value
        mock_impl.assert_called_once_with(cfg, groups, memory)

    @patch("vllm.platforms.current_platform")
    @patch.object(DefaultKVCacheConfigBuilder, "may_override_num_blocks")
    def test_may_override_num_blocks_delegates_to_default(
        self, mock_impl, mock_platform
    ):
        mock_platform.get_kv_cache_config_builder_cls.return_value = DEFAULT_PATH
        cfg = _make_vllm_config()
        result = KVCacheConfigBuilder.may_override_num_blocks(cfg, 4)
        assert result is mock_impl.return_value
        mock_impl.assert_called_once_with(cfg, 4, None)
