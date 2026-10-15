#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Render one role of a model_arch.yaml strategy into serve-ready output.

model_arch.yaml is a registry of Tokens deployment strategies keyed by
arch_name (``<model>-<quant>-<platform>-<topology>``). Each role's ``engine``
section is ``vllm serve --config`` compatible; this tool extracts one role so
the registry stays the single source of truth:

    vllm serve --config <(render_serve_config.py <arch_name> prefill)
    eval "$(render_serve_config.py <arch_name> prefill --env)"
"""

import argparse
import shlex
import sys
from pathlib import Path

import yaml

ROLES = ("prefill", "decode", "standalone")


def load_role(path: Path, arch_name: str, role: str) -> dict:
    """Return the per-role section of one strategy from the registry.

    Args:
        path: Path to model_arch.yaml.
        arch_name: Strategy key, e.g. GLM-5.2-W4A4C8-MG40-BNT3-1P1D.
        role: One of prefill / decode / standalone.

    Returns:
        The role's dict with ``engine`` and optional ``env`` sections.

    Raises:
        SystemExit: If the file, arch_name, or role is missing.
    """
    try:
        registry = yaml.safe_load(path.read_text())
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    if not isinstance(registry, dict) or arch_name not in registry:
        known = ", ".join(sorted(registry)) if isinstance(registry, dict) else "none"
        raise SystemExit(f"unknown arch_name {arch_name!r}; known: {known}")
    strategy = registry[arch_name]
    section = strategy.get(role) if isinstance(strategy, dict) else None
    if section is None:
        known = ", ".join(k for k in strategy if k in ROLES)
        raise SystemExit(f"{arch_name!r} has no {role!r} role; available: {known}")
    return section


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arch_name", help="strategy key in model_arch.yaml")
    parser.add_argument("role", choices=ROLES)
    parser.add_argument(
        "--file",
        type=Path,
        default=Path(__file__).with_name("model_arch.yaml"),
        help="registry file (default: model_arch.yaml next to this script)",
    )
    parser.add_argument(
        "--env",
        action="store_true",
        help="print the role's env section as export lines instead of the engine YAML",
    )
    args = parser.parse_args()

    role = load_role(args.file, args.arch_name, args.role)

    if args.env:
        for key, value in (role.get("env") or {}).items():
            print(f"export {key}={shlex.quote(str(value))}")
        return

    engine = role.get("engine")
    if not isinstance(engine, dict) or not engine:
        raise SystemExit(f"{args.arch_name!r} role {args.role!r} has no engine section")
    yaml.safe_dump(engine, sys.stdout, sort_keys=False)


if __name__ == "__main__":
    main()
