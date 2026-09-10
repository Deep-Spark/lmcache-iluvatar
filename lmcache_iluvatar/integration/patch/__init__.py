# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Idempotent LMCache runtime patching for Iluvatar."""

from lmcache_iluvatar.integration.patch.runtime import (
    PatchSpec,
    PatchResult,
    PatchState,
    VersionRange,
    activate_c_ops_redirect,
    activate_patches,
    activate_post_import_patches,
    get_patch_state,
)

__all__ = [
    "PatchResult",
    "PatchSpec",
    "PatchState",
    "VersionRange",
    "activate_c_ops_redirect",
    "activate_patches",
    "activate_post_import_patches",
    "get_patch_state",
]
