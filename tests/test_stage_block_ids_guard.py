# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for blocking stage_block_ids Iluvatar guard (no GPU torch)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _load_guard(monkeypatch):
    torch = ModuleType("torch")
    torch.long = "long"
    captured: dict = {}

    def frombuffer(flat, dtype):
        captured["frombuffer"] = (flat, dtype)
        return SimpleNamespace(dtype=dtype, flat=flat)

    torch.frombuffer = frombuffer
    monkeypatch.setitem(sys.modules, "torch", torch)

    path = (
        Path(__file__).resolve().parents[1]
        / "lmcache_iluvatar"
        / "v1"
        / "stage_block_ids_guard.py"
    )
    spec = importlib.util.spec_from_file_location("_test_stage_block_ids_guard", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, captured


class _Buffer:
    def __init__(self, length: int = 16) -> None:
        self._length = length
        self.copy_calls: list[dict] = []

    @property
    def shape(self):
        return (self._length,)

    def __getitem__(self, item):
        return self

    def copy_(self, src, non_blocking=False):
        self.copy_calls.append({"non_blocking": non_blocking})
        return self


class _UpstreamContext:
    def __init__(self) -> None:
        self.block_ids_buffer_ = _Buffer()

    def stage_block_ids(self, block_ids_per_group):
        raise AssertionError("upstream non_blocking path must not run")


def test_build_blocking_stage_block_ids_is_idempotent_and_blocking(monkeypatch):
    guard, captured = _load_guard(monkeypatch)
    cls = guard.build_blocking_stage_block_ids(_UpstreamContext)
    assert getattr(cls, guard._MARKER, False)
    assert guard.build_blocking_stage_block_ids(cls) is cls

    ctx = cls()
    views = ctx.stage_block_ids([[10, 11], [12]])
    assert len(views) == 2
    assert "frombuffer" in captured
    assert ctx.block_ids_buffer_.copy_calls == [{"non_blocking": False}]


def test_build_blocking_stage_block_ids_rejects_oversized(monkeypatch):
    guard, _ = _load_guard(monkeypatch)
    cls = guard.build_blocking_stage_block_ids(_UpstreamContext)
    ctx = cls()
    ctx.block_ids_buffer_ = _Buffer(length=1)
    try:
        ctx.stage_block_ids([[1, 2]])
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "exceeds the pre-allocated buffer" in str(exc)
