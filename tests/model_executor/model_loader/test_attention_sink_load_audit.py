# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard against checkpoint-backed attention sinks bypassing weight_loader.

A model ``load_weights`` that writes a sink parameter with a bare ``copy_`` is
invisible to ``online_process_loader``, so a live weight update silently keeps
the old sink value. This is a source-level audit rather than a runtime check
because the models involved need large checkpoints and several hardware
backends to instantiate.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# Checkpoint-backed attention sink parameters, and the runtime/derived tensors
# that intentionally mirror them.
SINK_NAMES = (
    "attn_sink",
    "attention_sink",
    "attention_sink_bias",
    "learnable_sink",
    "learnable_sink_param",
    "sinks",
)

# Files that create or load checkpoint-backed sinks. Kept explicit so the audit
# fails when a new sink path is added without being reviewed.
SINK_LOAD_FILES = (
    "vllm/models/deepseek_v4/cpu/model.py",
    "vllm/models/deepseek_v4/xpu/model.py",
    "vllm/models/deepseek_v4/amd/model.py",
    "vllm/models/deepseek_v4/nvidia/model.py",
    "vllm/models/deepseek_v4/xpu/mtp.py",
    "vllm/models/deepseek_v4/amd/mtp.py",
    "vllm/models/deepseek_v4/nvidia/mtp.py",
    "vllm/models/deepseek_v4/xpu/dspark.py",
    "vllm/models/deepseek_v4/amd/dspark.py",
    "vllm/models/deepseek_v4/nvidia/dspark.py",
    "vllm/models/deepseek_v41/amd/model.py",
    "vllm/models/deepseek_v41/nvidia/model.py",
    "vllm/models/deepseek_v41/amd/dspark.py",
    "vllm/models/deepseek_v41/nvidia/dspark.py",
    "vllm/models/hy_v4/nvidia/model.py",
    "vllm/models/hy_v4/nvidia/mtp.py",
    "vllm/model_executor/models/mimo_v2.py",
    "vllm/model_executor/models/mimo_v2_mtp.py",
    "vllm/model_executor/models/gpt_oss.py",
    # Modules that only create the parameter. Reviewed here too, so a new sink
    # definition cannot appear without a decision about how it is loaded.
    "vllm/models/deepseek_v4/attention.py",
    "vllm/models/deepseek_v41/attention.py",
    "vllm/models/hy_v4/nvidia/attention.py",
    "vllm/model_executor/models/granite.py",
    "vllm/model_executor/models/granitemoe.py",
    "vllm/model_executor/models/mimo_v2_omni.py",
    "vllm/model_executor/models/qwen3_dflash.py",
)


def _mentions_sink(node: ast.AST) -> bool:
    """Whether a subtree references an attention sink by name."""
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            if any(sink in child.value for sink in SINK_NAMES):
                return True
        elif isinstance(child, ast.Name) and any(
            sink in child.id for sink in SINK_NAMES
        ):
            return True
    return False


def _copy_target_names(call: ast.Call) -> set[str]:
    """Names a ``copy_`` call writes into.

    Handles ``X.copy_(...)``, ``X[:n].copy_(...)`` and
    ``params_dict[name][:n].copy_(...)``; for the latter the destination name
    comes from the subscript index, which the caller correlates with the sink
    test.
    """
    target = call.func.value  # type: ignore[attr-defined]
    names: set[str] = set()
    while isinstance(target, (ast.Subscript, ast.Attribute)):
        if isinstance(target, ast.Subscript):
            index = target.slice
            for child in ast.walk(index):
                if isinstance(child, ast.Name):
                    names.add(child.id)
            if isinstance(index, ast.Constant) and isinstance(index.value, str):
                names.add(index.value)
        target = target.value
    if isinstance(target, ast.Name):
        names.add(target.id)
    return names


def _find_direct_sink_copies(path: Path) -> list[int]:
    """Return line numbers of ``copy_`` calls that write attention sink data.

    A call is a violation when it writes a bare local alias of a sink parameter
    -- ``param = params_dict[name]`` under a sink name test, followed by
    ``param.copy_(...)`` or ``param[:n].copy_(...)``. All three historical forms
    are covered:

        params_dict[name][:n].copy_(narrow_weight)
        param.data.copy_(narrow_weight)        # after param = params_dict[name]
        param.copy_(narrow_weight)

    ``param.data.copy_`` / ``param[:n].copy_`` used for derived runtime state
    inside ``process_weights_after_loading`` is intentionally not matched: only
    loads driven by a checkpoint name are in scope.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents: dict[ast.AST, ast.AST] = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    violations: list[int] = []

    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # Locals bound to the sink parameter inside a sink test:
        # ``param = params_dict[name]`` under ``if "attn_sink" in name``.
        sink_locals: set[str] = set()
        for node in ast.walk(func):
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            if not (
                isinstance(value, ast.Subscript)
                and isinstance(value.value, ast.Name)
                and value.value.id == "params_dict"
                and _is_under_sink_test(node, parents)
            ):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    sink_locals.add(target.id)

        for node in ast.walk(func):
            if not isinstance(node, ast.Call):
                continue
            callee = node.func
            if not isinstance(callee, ast.Attribute) or callee.attr != "copy_":
                continue
            if not _is_under_sink_test(node, parents):
                continue
            target = callee.value
            root = target
            while isinstance(root, (ast.Subscript, ast.Attribute)):
                root = root.value
            is_sink_alias = isinstance(root, ast.Name) and root.id in sink_locals
            # ``params_dict[name][:n].copy_(...)``, or a copy into the sink
            # parameter itself: ``param.copy_`` / ``param.data.copy_`` /
            # ``param[:n].copy_``. The bare ``param`` forms are matched when the
            # loop aliases the parameter outside the sink branch (the MiMo-V2
            # shape), so an attribute or subscript destination reached under a
            # sink test is treated as a violation.
            if is_sink_alias or isinstance(target, (ast.Attribute, ast.Subscript)):
                violations.append(node.lineno)

    return sorted(set(violations))


def _is_under_sink_test(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """Whether ``node`` is nested inside a statement that tests a sink name."""
    current = node
    while current is not None:
        parent = parents.get(current)
        if parent is None:
            break
        if isinstance(parent, (ast.If, ast.While, ast.IfExp)) and _mentions_sink(
            parent.test
        ):
            return True
        current = parent
    return False


@pytest.mark.parametrize("relative_path", SINK_LOAD_FILES)
def test_sink_load_does_not_bypass_weight_loader(relative_path: str):
    """Sink parameters must be loaded through the parameter's current loader."""
    path = REPO_ROOT / relative_path
    assert path.is_file(), f"missing sink load file: {relative_path}"

    violations = _find_direct_sink_copies(path)
    assert not violations, (
        f"{relative_path} writes an attention sink with a bare copy_ at "
        f"line(s) {violations}; route it through param.weight_loader instead "
        "(see docs/design/attention_sink_weight_update.md)"
    )


def test_audit_covers_every_sink_parameter_assignment():
    """Every module that assigns a sink parameter must be in the audit list."""
    audited = {REPO_ROOT / p for p in SINK_LOAD_FILES}
    discovered: set[Path] = set()

    for path in (REPO_ROOT / "vllm").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "Parameter(" not in text:
            continue
        for sink in SINK_NAMES:
            if f"self.{sink} = " in text:
                discovered.add(path)
                break

    unaudited = sorted(str(p.relative_to(REPO_ROOT)) for p in discovered - audited)
    assert not unaudited, (
        "these modules assign an attention sink parameter but are not covered "
        f"by SINK_LOAD_FILES: {unaudited}"
    )
