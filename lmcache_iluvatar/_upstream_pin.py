# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Load the upstream LMCache pin shipped with lmcache-iluvatar."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_PIN_BASENAME = "upstream_pin.json"


def _candidate_pin_paths() -> tuple[Path, ...]:
    module_dir = Path(__file__).resolve().parent
    repo_root = module_dir.parent
    package_pin = module_dir / _PIN_BASENAME
    root_pin = repo_root / _PIN_BASENAME
    # In a source checkout prefer the repo-root pin so a stale/unwritable
    # package-local copy cannot override the bumped pin during development.
    if (repo_root / "setup.py").is_file() and root_pin.is_file():
        return (root_pin, package_pin)
    return (package_pin, root_pin)


@lru_cache(maxsize=1)
def load_upstream_pin() -> dict[str, Any]:
    for path in _candidate_pin_paths():
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    searched = ", ".join(str(path) for path in _candidate_pin_paths())
    raise FileNotFoundError(f"{_PIN_BASENAME} not found; searched: {searched}")


def _lmcache_section() -> dict[str, Any]:
    section = load_upstream_pin()["lmcache"]
    if not isinstance(section, dict):
        raise ValueError("upstream_pin.json: lmcache must be an object")
    return section


def _require_non_empty_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"upstream_pin.json: {field} must be a non-empty string")
    return value


def pinned_lmcache_pypi_version() -> str:
    return _require_non_empty_str(
        _lmcache_section()["pypi_version"],
        "lmcache.pypi_version",
    )
