# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Iluvatar lookup client compatibility hooks."""

from lmcache_iluvatar.v1.lookup_client.async_lookup_client import (
    build_iluvatar_async_lookup_client,
)

__all__ = ["build_iluvatar_async_lookup_client"]
