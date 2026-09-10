# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Runtime entry point for the LMCache Iluvatar plugin."""

from __future__ import annotations

from importlib import import_module
import logging
import sys

__version__ = "0.1.0"

# --- lmcache_iluvatar logging namespace ------------------------------------
# Use the same format and colouring as upstream LMCache so operator tooling
# (log scrapers, colour rules) treats both namespaces uniformly.
_GREEN = "\x1b[32;20m"
_RESET = "\x1b[0m"
_UNDERLINE = "\x1b[3m"
_LOG_FMT = (
    f"{_GREEN}[%(asctime)s] LMCache_Iluvatar %(levelname)s:{_RESET} %(message)s "
    f"{_UNDERLINE}(%(filename)s:%(lineno)d:%(name)s){_RESET}"
)
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter(_LOG_FMT))
_handler.setLevel(logging.INFO)
_lm_iluvatar_root = logging.getLogger("lmcache_iluvatar")
_lm_iluvatar_root.addHandler(_handler)
_lm_iluvatar_root.setLevel(logging.INFO)
_lm_iluvatar_root.propagate = True


def ensure_lmcache_install() -> None:
    """Fail fast when upstream LMCache is missing."""

    try:
        import_module("lmcache")
    except ImportError as exc:
        message = (
            "lmcache-iluvatar requires upstream LMCache installed in the same "
            f"Python environment. Unable to import lmcache: {exc}"
        )
        raise RuntimeError(message) from exc


from lmcache_iluvatar.integration.patch import (  # noqa: E402
    PatchResult,
    PatchSpec,
    VersionRange,
    activate_c_ops_redirect,
    activate_patches,
    activate_post_import_patches,
    get_patch_state,
)

# Importing the plugin should be enough for vLLM's module-path loading to patch
# LMCache before connector objects are constructed. Preinstall the c_ops redirect
# first so upstream LMCache does not try to load its CUDA backend during checks.
activate_c_ops_redirect(strict=False)
ensure_lmcache_install()
PATCH_RESULT = activate_patches(strict=True)

__all__ = [
    "PATCH_RESULT",
    "PatchResult",
    "PatchSpec",
    "VersionRange",
    "__version__",
    "activate_c_ops_redirect",
    "activate_patches",
    "activate_post_import_patches",
    "ensure_lmcache_install",
    "get_patch_state",
]
