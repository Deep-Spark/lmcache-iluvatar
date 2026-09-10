# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Prevent premature GC of CUDA IPC events in LMCache MP processes.

When either side of an LMCache MP transfer exports a CUDA IPC event handle,
the local event object must stay alive until the peer imports the handle via
``from_ipc_handle``. Premature garbage collection causes the import to fail
because the underlying CUDA event has already been destroyed. This applies
to both server completion events and vLLM worker readiness events. Chunked
prefill makes the client-side case particularly likely because upstream
tracks only one event per ``request_id`` while submitting multiple STOREs.

The upstream fix (commit ``b8596b9c``) adds an ACK protocol.  This module
uses a simpler approach: a module-level FIFO ``deque`` that holds
references to every interprocess event created inside each patched process.

Important: constructors must return the **real** ``torch_dev.Event`` instance
(not a Python proxy). A proxy that forwards via ``__getattr__`` breaks
PyTorch/Iluvatar type expectations and has been observed to correlate with
PP + shm_broadcast hangs under sustained STORE traffic. The FIFO only keeps
an extra strong reference; call sites keep using a genuine Event object.
"""

from __future__ import annotations

from collections import deque
from typing import Any
import logging

logger = logging.getLogger(__name__)

_FIFO_MAXLEN = 10000
_ipc_event_fifo: deque = deque(maxlen=_FIFO_MAXLEN)


def retain_ipc_event(event: Any) -> Any:
    """Keep ``event`` alive in the process-local FIFO; return it unchanged."""
    _ipc_event_fifo.append(event)
    return event


def build_guarded_torch_dev(current_torch_dev: Any) -> Any:
    """Return a ``torch_dev`` proxy whose ``Event(...)`` retains real events.

    ``torch_dev.Event(...)`` still constructs and returns the upstream
    ``Event`` type. Only ``interprocess=True`` instances are appended to the
    FIFO. ``from_ipc_handle`` is unchanged and does not enter the FIFO.
    """

    RealEvent = current_torch_dev.Event

    class _FifoRetainingEvent:
        """Constructor stand-in: builds a real Event and optionally retains it."""

        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            event = RealEvent(*args, **kwargs)
            if kwargs.get("interprocess", False):
                retain_ipc_event(event)
            return event

        @staticmethod
        def from_ipc_handle(*args: Any, **kwargs: Any) -> Any:
            return RealEvent.from_ipc_handle(*args, **kwargs)

    class _GuardedTorchDev:
        Event = _FifoRetainingEvent

        def __getattr__(self, name: str) -> Any:
            return getattr(current_torch_dev, name)

    return _GuardedTorchDev()


_EVENT_IPC_MARKER = "__lmcache_iluvatar_event_ipc_fifo__"


def build_guarded_event_ipc_backend(backend_cls: Any) -> Any:
    """Wrap ``DefaultEventIPCBackend.create_event`` to FIFO-retain events.

    LMCache v0.5.3 ``lmcache_driven`` creates completion events via
    ``event_backend.create_event()``, which uses the backend's captured
    ``_event_module`` (package ``lmcache.torch_dev``), not the transfer
    module's ``torch_dev`` attribute. Package-level ``torch_dev`` wrapping
    is intentionally not used; this patches the backend class method.

    MUSA's ``MusaEventIPCBackend.create_event`` delegates to
    ``DefaultEventIPCBackend.create_event``, so one wrap covers both paths.
    """

    if backend_cls is None:
        return None
    if getattr(backend_cls, _EVENT_IPC_MARKER, False):
        return backend_cls

    original = backend_cls.create_event

    def create_event(self: Any, device: Any) -> Any:
        event = original(self, device)
        return retain_ipc_event(event)

    backend_cls.create_event = create_event
    setattr(backend_cls, _EVENT_IPC_MARKER, True)
    logger.info(
        "Patched %s.create_event to retain IPC events in FIFO",
        getattr(backend_cls, "__name__", type(backend_cls).__name__),
    )
    return backend_cls
