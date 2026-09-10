# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Iluvatar storage connector compatibility hooks."""

from lmcache_iluvatar.v1.storage_backend.connector.eic_connector import (
    build_iluvatar_eic_connector,
)

__all__ = ["build_iluvatar_eic_connector"]
