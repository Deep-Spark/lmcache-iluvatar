# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from types import ModuleType, SimpleNamespace
import sys
import threading


def install_fake_lmcache(
    monkeypatch,
    *,
    include_dynamic_connector: bool = True,
) -> dict[str, ModuleType]:
    purge_modules("lmcache", "lmcache_iluvatar")
    modules: dict[str, ModuleType] = {}

    # Unit tests install a fake lmcache tree; hide any real PyPI distribution
    # so VersionRange(min=max=pin) does not skip patches against host 0.5.0.
    try:
        import importlib.metadata as importlib_metadata
    except ImportError:  # pragma: no cover
        import importlib_metadata  # type: ignore

    real_version = importlib_metadata.version

    def _fake_version(dist_name: str) -> str:
        if dist_name == "lmcache":
            raise importlib_metadata.PackageNotFoundError(dist_name)
        return real_version(dist_name)

    monkeypatch.setattr(importlib_metadata, "version", _fake_version)

    for name in (
        "lmcache_iluvatar.c_ops",
        "lmcache_iluvatar.native_storage_ops",
        "lmcache_iluvatar.lmcache_redis",
        "lmcache_iluvatar.lmcache_fs",
    ):
        stub = ModuleType(name)
        if name == "lmcache_iluvatar.c_ops":
            stub.EngineKVFormat = SimpleNamespace(NL_X_NB_BSV_BSS=14)
            for symbol in (
                "execute_object_group_transfer",
                "execute_cb_retrieve_plan_flat",
                "is_cross_layer",
                "is_kv_list",
                "is_layer_list",
                "is_mla",
            ):
                setattr(stub, symbol, lambda *args, **kwargs: None)
        monkeypatch.setitem(sys.modules, name, stub)

    try:
        import torch  # noqa: F401
    except ImportError:
        torch_stub = ModuleType("torch")
        torch_stub.int8 = object()
        monkeypatch.setitem(sys.modules, "torch", torch_stub)

    for name in (
        "lmcache",
        "lmcache.c_ops",
        "lmcache.v1",
        "lmcache.v1.multiprocess",
        "lmcache.v1.multiprocess.mq",
        "lmcache.v1.multiprocess.protocol",
        "lmcache.v1.multiprocess.protocols",
        "lmcache.v1.multiprocess.protocols.base",
        "lmcache.v1.multiprocess.protocols.engine",
        "lmcache.v1.multiprocess.transfer_context",
        "lmcache.v1.multiprocess.transfer_context.worker_transfer",
        "lmcache.v1.multiprocess.engine_context",
        "lmcache.v1.multiprocess.server",
        "lmcache.v1.multiprocess.modules",
        "lmcache.v1.multiprocess.modules.lookup",
        "lmcache.v1.multiprocess.modules.lmcache_driven_transfer",
        "lmcache.v1.multiprocess.modules.blend",
        "lmcache.v1.multiprocess.modules.blend_v3",
        "lmcache.v1.platform",
        "lmcache.v1.platform.base",
        "lmcache.v1.platform.base.event_ipc",
        "lmcache.v1.platform.base.cache_context",
        "lmcache.v1.cache_engine",
        "lmcache.v1.gpu_connector",
        "lmcache.v1.gpu_connector.gpu_connectors",
        "lmcache.v1.manager",
        "lmcache.v1.distributed",
        "lmcache.v1.distributed.api",
        "lmcache.v1.distributed.storage_manager",
        "lmcache.v1.distributed.storage_controllers",
        "lmcache.v1.distributed.storage_controllers.prefetch_controller",
        "lmcache.v1.storage_backend",
        "lmcache.v1.storage_backend.storage_manager",
        "lmcache.v1.transfer_channel",
        "lmcache.v1.transfer_channel.transfer_utils",
        "lmcache.v1.storage_backend.pd_backend",
        "lmcache.v1.storage_backend.pd_backend_async",
        "lmcache.v1.storage_backend.p2p_backend",
        "lmcache.v1.event_manager",
        "lmcache.v1.lookup_client",
        "lmcache.v1.lookup_client.lmcache_async_lookup_client",
        "lmcache.v1.lookup_client.factory",
        "lmcache.v1.protocol",
        "lmcache.v1.compute",
        "lmcache.v1.compute.attention",
        "lmcache.v1.compute.attention.utils",
        "lmcache.v1.compute.blend",
        "lmcache.v1.compute.blend.blender",
        "lmcache.v1.compute.models",
        "lmcache.v1.compute.models.base",
        "lmcache.v1.compute.models.utils",
        "lmcache.integration",
        "lmcache.integration.vllm",
        "lmcache.integration.vllm.utils",
        "lmcache.integration.vllm.kv_cache_group_edits",
        "lmcache.integration.vllm.vllm_multi_process_adapter",
        "lmcache.integration.vllm.vllm_v1_adapter",
        "lmcache.integration.vllm.lmcache_connector_v1",
        "lmcache.integration.vllm.lmcache_connector_v1_085",
        "lmcache.integration.vllm.lmcache_mp_connector",
        "vllm",
        "vllm.v1",
        "vllm.v1.worker",
        "vllm.v1.worker.gpu_worker",
        "vllm.distributed",
        "vllm.distributed.kv_transfer",
        "vllm.distributed.kv_transfer.kv_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1",
        "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector",
    ):
        module = ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)

    def original_factory(*args, **kwargs):
        return args, kwargs

    def original_device():
        return None, "cuda"

    def original_storage_factory(*args, **kwargs):
        return {"storage": (args, kwargs)}

    def original_transfer_factory(*args, **kwargs):
        return {"transfer": (args, kwargs)}

    def original_correct_device(device, worker_id):
        return f"{device}:{worker_id}"

    def original_infer_attn_backend_from_vllm(vllm_attn, enable_sparse=False):
        return {
            "vllm_attn": vllm_attn,
            "enable_sparse": enable_sparse,
            "source": "upstream",
        }

    class GPUConnectorInterface:
        pass

    class VLLMPagedMemGPUConnectorV2(GPUConnectorInterface):
        pass

    class OriginalImpl:
        pass

    class OriginalIPCEvent:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        @staticmethod
        def from_ipc_handle(*args, **kwargs):
            return ("ipc-handle", args, kwargs)

    class TorchDev:
        Event = OriginalIPCEvent

    class LMCBlender:
        def __init__(self, cache_engine, gpu_connector, vllm_model, config):
            self.cache_engine = cache_engine
            self.gpu_connector = gpu_connector
            self.vllm_model = vllm_model
            self.config = config

    class VLLMModelTracker:
        registered: dict[str, object] = {}

        @classmethod
        def register_model(cls, engine_name, model):
            cls.registered[engine_name] = model

    class FakeModel:
        def embed_input_ids(self, input_ids):
            return ("embedded", input_ids)

    class ModelRunner:
        def __init__(self):
            self.model = FakeModel()

    class Worker:
        def __init__(self):
            self.model_runner = ModelRunner()

        def load_model(self):
            return "loaded"

    class _EmptyBlocks:
        def get_block_ids(self):
            return ()

    class _RealBlocks:
        def __init__(self, block_ids):
            self._block_ids = block_ids

        def get_block_ids(self):
            return self._block_ids

        def new_empty(self):
            return _EmptyBlocks()

    class _ChildConnector:
        def __init__(self, name):
            self.name = name
            self.calls = []

        def update_state_after_alloc(self, request, blocks, num_external_tokens):
            self.calls.append(
                {
                    "request_id": request.request_id,
                    "block_ids": blocks.get_block_ids(),
                    "num_external_tokens": num_external_tokens,
                    "empty": isinstance(blocks, _EmptyBlocks),
                }
            )

    class MultiConnector:
        """Upstream-shaped stub: empty blocks for non-chosen connectors."""

        def __init__(self):
            self._connectors = [_ChildConnector("a"), _ChildConnector("b")]
            self._requests_to_connector = {}

        def update_state_after_alloc(self, request, blocks, num_external_tokens):
            chosen_connector = self._requests_to_connector.get(request.request_id, -1)
            empty_blocks = blocks.new_empty()
            for i, connector in enumerate(self._connectors):
                if i == chosen_connector:
                    connector.update_state_after_alloc(
                        request, blocks, num_external_tokens
                    )
                else:
                    connector.update_state_after_alloc(request, empty_blocks, 0)

    class EventType(Enum):
        LOADING = auto()

    class EventStatus(Enum):
        ONGOING = auto()
        DONE = auto()
        NOT_FOUND = auto()

    class EventManager:
        def __init__(self):
            self.events = {EventType.LOADING: {status: {} for status in EventStatus}}
            self.lock = threading.Lock()

        def add_event(self, event_type, event_id, future):
            with self.lock:
                self.events[event_type][EventStatus.ONGOING][event_id] = future

        def update_event_status(self, event_type, event_id, status):
            with self.lock:
                event = None
                for current_status in EventStatus:
                    if event_id in self.events[event_type][current_status]:
                        event = self.events[event_type][current_status].pop(event_id)
                        break
                if event is None:
                    raise KeyError(event_id)
                self.events[event_type][status][event_id] = event

        def get_event_status(self, event_type, event_id):
            with self.lock:
                for status in EventStatus:
                    if event_id in self.events[event_type][status]:
                        return status
                return EventStatus.NOT_FOUND

        def pop_event(self, event_type, event_id):
            with self.lock:
                return self.events[event_type][EventStatus.DONE].pop(event_id)

    class LMCacheEngine:
        def __init__(self):
            self.storage_manager = None
            self.remove_after_retrieve = False
            self.lookup_pins = {}
            self.retrieved = []
            self.unpinned = []
            self.cleaned_up = []

        def _is_passive(self):
            return False

        def retrieve(self, tokens, mask=None, **kwargs):
            self.retrieved.append((tokens, mask, kwargs))
            return kwargs

        def lookup_unpin(self, lookup_id):
            self.unpinned.append(lookup_id)

        def cleanup_memory_objs(self, lookup_id):
            self.cleaned_up.append(lookup_id)

    class StorageManager:
        def __init__(self, storage_backends=None):
            self.storage_backends = storage_backends or {}

        def prefetch_single_done_callback(self, future, keys, backend_name):
            return None

        def get_active_storage_backends(self, search_range=None):
            return []

        def prefetch_all_done_callback(
            self,
            task,
            lookup_id,
            cum_chunk_lengths_total,
            tier_expected_chunks,
            loading_tasks=None,
        ):
            return None

        async def async_lookup_and_prefetch(
            self,
            lookup_id,
            keys,
            cum_chunk_lengths,
            search_range=None,
            pin=False,
        ):
            return None

    class LookupCleanupMsg:
        lookup_id: str

    class LMCacheAsyncLookupClient:
        def __init__(self, config=None, metadata=None):
            self.config = config
            self.metadata = metadata
            self.world_size = 1
            self.push_sockets = []
            self.reqs_status = {}
            self.first_lookup_time = {}
            self.res_for_each_worker = {}
            self.aborted_lookups = set()
            self.lock = threading.Lock()

        def lookup_cache(self, lookup_id):
            return -1

        def clear_lookup_status(self, lookup_id):
            self.reqs_status.pop(lookup_id, None)

        def cancel_lookup(self, lookup_id):
            self.aborted_lookups.add(lookup_id)

        def _cleanup_finished_aborted_lookups(self):
            return None

        def _send_cleanup_message(self, lookup_id):
            return None

    class PDBackend:
        pass

    class PDBackendAsync:
        pass

    class UpstreamConnector:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class LMCacheMPWorkerAdapter:
        worker_id = 0

        def _send_register_kv_caches_request(self, kv_caches):
            return None

        def _block_ids_per_group(self, op):
            return [list(op.block_ids)]

    class ProtocolDefinition:
        def __init__(self, payload_classes):
            self.payload_classes = payload_classes

    class RequestType(Enum):
        REGISTER_KV_CACHE = auto()

    def get_engine_protocol_definitions():
        return {
            "REGISTER_KV_CACHE": ProtocolDefinition(
                [int, object, str, int, object, object, list]
            )
        }

    class LMCacheDrivenTransferContext:
        def register(self, *args, **kwargs):
            return None

    class LayoutDescRegistry:
        def register(self, *args, **kwargs):
            return None

        def unregister(self, *args, **kwargs):
            return None

    class LMCacheDrivenTransferModule:
        # Explicit required params so rank-layout can append worker_id without
        # breaking inspect.Signature parameter ordering rules.
        def register_kv_cache(
            self,
            key,
            instance_id,
            kv_caches,
            event_ipc_handle,
            arg4,
            arg5,
            arg6,
        ):
            return None

        def _release_entries(self, entries):
            return None

    @dataclass(frozen=True)
    class PrefetchRequestSpec:
        keys: list
        group_layout_descs: dict
        extra_count: int = 0

        def __post_init__(self):
            return None

    class ObjectKey:
        @staticmethod
        def ComputeKVRank(world_size, global_rank, local_world_size, local_rank):
            return (
                world_size << 28
                | global_rank << 16
                | local_world_size << 12
                | local_rank
            )

    class LookupModule:
        def lookup(self, key, tp_size):
            return None

    class PrefetchController:
        def _start_lookup_phase(self, request_id, spec):
            return None

        def _reserve_load_buffers(self, request, keys_to_reserve):
            return set()

    class _SubpagedAttentionViewEdit:
        """Upstream-like NHD-only subpaged edit (PR #3613 shape)."""

        name = "subpaged-attention-view"

        def matches(self, spec, kv_cache):
            import torch

            # Loose upstream match: five-D attention with kernel!=logical BS
            # at NHD dim 2. HND false-matches here before Iluvatar tightening.
            return (
                isinstance(kv_cache, torch.Tensor)
                and kv_cache.ndim == 5
                and int(kv_cache.shape[2]) != int(spec.block_size)
            )

        def apply(self, spec, kv_cache, layout_hints=None):
            if kv_cache.shape[1] != 2:
                raise ValueError(
                    "expected a (num_blocks, 2, block_size, num_heads, "
                    f"head_size) attention KV tensor, got {tuple(kv_cache.shape)}"
                )
            return kv_cache

    class _MambaPageViewEdit:
        """Minimal stand-in so hybrid fallback cannot skip non-attention edits."""

        name = "mamba-page-view"
        calls = 0

        def matches(self, spec, kv_cache):
            return isinstance(kv_cache, list)

        def apply(self, spec, kv_cache, layout_hints=None):
            type(self).calls += 1
            return kv_cache

    _EDITS = (_MambaPageViewEdit(), _SubpagedAttentionViewEdit())

    def apply_kv_cache_group_edits(kv_cache_config, kv_caches, layout_hints=None):
        """Upstream-like registry dispatch over ``_EDITS`` (v0.5.3 layout_hints)."""

        if kv_cache_config is None or not getattr(
            kv_cache_config, "has_mamba_layers", False
        ):
            return dict(kv_caches)

        edited = dict(kv_caches)
        for group in kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            for name in group.layer_names:
                cache = edited[name]
                for edit in modules[
                    "lmcache.integration.vllm.kv_cache_group_edits"
                ]._EDITS:
                    if edit.matches(spec, cache):
                        edited[name] = edit.apply(spec, cache, layout_hints)
                        break
        return edited

    class DefaultEventIPCBackend:
        """Minimal stand-in for v0.5.3 event_ipc create_event path."""

        def __init__(self, event_module=None, device_type="cuda"):
            self._event_module = event_module
            self.device_type = device_type

        def create_event(self, device):
            if self._event_module is not None and hasattr(self._event_module, "Event"):
                return self._event_module.Event(interprocess=True)
            return object()

    class _FakeBuffer:
        def __init__(self, length: int):
            self._length = length
            self.copy_kwargs: list[dict] = []

        @property
        def shape(self):
            return (self._length,)

        def __getitem__(self, item):
            return self

        def copy_(self, src, non_blocking=False):
            self.copy_kwargs.append({"non_blocking": non_blocking, "src": src})
            return self

    class BaseCacheContext:
        """Upstream-shaped stub: non_blocking stage_block_ids (to be patched)."""

        def __init__(self, buffer_len: int = 16):
            self.block_ids_buffer_ = _FakeBuffer(buffer_len)
            self.calls: list[dict] = []

        def stage_block_ids(self, block_ids_per_group):
            # Mirrors the upstream lifetime hole for patch targeting.
            total = sum(len(ids) for ids in block_ids_per_group)
            self.calls.append({"total": total, "non_blocking": True})
            return [object() for _ in block_ids_per_group]

    modules["lmcache.v1.event_manager"].EventManager = EventManager
    modules["lmcache.v1.event_manager"].EventType = EventType
    modules["lmcache.v1.event_manager"].EventStatus = EventStatus
    modules[
        "lmcache.v1.lookup_client.lmcache_async_lookup_client"
    ].LMCacheAsyncLookupClient = (  # noqa: E501
        LMCacheAsyncLookupClient
    )
    modules[
        "lmcache.v1.lookup_client.lmcache_async_lookup_client"
    ].LookupCleanupMsg = LookupCleanupMsg
    modules[
        "lmcache.v1.lookup_client.factory"
    ].LMCacheAsyncLookupClient = LMCacheAsyncLookupClient
    for name in (
        "lmcache.v1.multiprocess.server",
        "lmcache.v1.multiprocess.modules.lmcache_driven_transfer",
        "lmcache.v1.multiprocess.modules.blend",
        "lmcache.v1.multiprocess.modules.blend_v3",
        "lmcache.integration.vllm.lmcache_mp_connector",
    ):
        modules[name].torch_dev = TorchDev()
    modules[
        "lmcache.v1.platform.base.event_ipc"
    ].DefaultEventIPCBackend = DefaultEventIPCBackend
    modules[
        "lmcache.v1.platform.base.cache_context"
    ].BaseCacheContext = BaseCacheContext
    modules["lmcache.v1.platform.base"].cache_context = modules[
        "lmcache.v1.platform.base.cache_context"
    ]
    modules["lmcache.v1.protocol"].DTYPE_TO_INT = {}
    modules["lmcache.v1.protocol"].INT_TO_DTYPE = {}
    modules[
        "lmcache.v1.compute.attention.utils"
    ].infer_attn_backend_from_vllm = original_infer_attn_backend_from_vllm
    modules["lmcache.v1.compute.blend.blender"].LMCBlender = LMCBlender
    modules[
        "lmcache.v1.compute.models.base"
    ].infer_attn_backend_from_vllm = original_infer_attn_backend_from_vllm
    modules["lmcache.v1.compute.models.utils"].VLLMModelTracker = VLLMModelTracker
    modules["lmcache.v1.cache_engine"].LMCacheEngine = LMCacheEngine
    modules["lmcache.v1.cache_engine"].StorageManager = StorageManager
    modules["lmcache.v1.manager"].LMCacheEngine = LMCacheEngine
    modules["lmcache.v1.gpu_connector"].CreateGPUConnector = original_factory
    modules[
        "lmcache.v1.gpu_connector.gpu_connectors"
    ].GPUConnectorInterface = GPUConnectorInterface
    modules[
        "lmcache.v1.gpu_connector.gpu_connectors"
    ].VLLMPagedMemGPUConnectorV2 = VLLMPagedMemGPUConnectorV2
    modules["lmcache.v1.manager"].CreateGPUConnector = original_factory
    modules[
        "lmcache.v1.storage_backend"
    ].CreateStorageBackends = original_storage_factory
    modules["lmcache.v1.storage_backend"].PDBackend = PDBackend
    modules["lmcache.v1.storage_backend"].PDBackendAsync = PDBackendAsync
    modules[
        "lmcache.v1.storage_backend.storage_manager"
    ].CreateStorageBackends = original_storage_factory
    modules[
        "lmcache.v1.storage_backend.storage_manager"
    ].StorageManager = StorageManager
    modules["lmcache.v1.storage_backend.pd_backend"].PDBackend = PDBackend
    modules[
        "lmcache.v1.storage_backend.pd_backend_async"
    ].PDBackendAsync = PDBackendAsync
    modules[
        "lmcache.v1.transfer_channel"
    ].CreateTransferChannel = original_transfer_factory
    modules[
        "lmcache.v1.storage_backend.pd_backend"
    ].CreateTransferChannel = original_transfer_factory
    modules[
        "lmcache.v1.storage_backend.p2p_backend"
    ].CreateTransferChannel = original_transfer_factory
    modules[
        "lmcache.v1.transfer_channel.transfer_utils"
    ].get_correct_device = original_correct_device
    modules["lmcache.integration.vllm.utils"].get_vllm_torch_dev = original_device
    modules["lmcache.integration.vllm.utils"].ENGINE_NAME = "vllm-instance"
    modules["lmcache.integration.vllm.kv_cache_group_edits"]._EDITS = _EDITS
    modules[
        "lmcache.integration.vllm.kv_cache_group_edits"
    ].apply_kv_cache_group_edits = apply_kv_cache_group_edits
    modules[
        "lmcache.integration.vllm.lmcache_mp_connector"
    ].apply_kv_cache_group_edits = apply_kv_cache_group_edits
    modules["vllm.v1.worker.gpu_worker"].Worker = Worker
    modules[
        "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector"
    ].MultiConnector = MultiConnector
    modules[
        "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector"
    ].LMCacheMPConnector = UpstreamConnector
    # Expose helpers for unit tests that exercise the MultiConnector patch.
    modules[
        "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector"
    ]._RealBlocks = _RealBlocks
    modules[
        "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector"
    ]._EmptyBlocks = _EmptyBlocks
    modules["lmcache"].c_ops = modules["lmcache.c_ops"]
    modules[
        "lmcache.integration.vllm.vllm_v1_adapter"
    ].LMCacheConnectorV1Impl = OriginalImpl
    modules["lmcache.integration.vllm.vllm_v1_adapter"].LMCacheEngine = LMCacheEngine
    modules[
        "lmcache.integration.vllm.lmcache_connector_v1"
    ].LMCacheConnectorV1Impl = OriginalImpl
    if include_dynamic_connector:
        modules[
            "lmcache.integration.vllm.lmcache_connector_v1"
        ].LMCacheConnectorV1Dynamic = UpstreamConnector
    modules[
        "lmcache.integration.vllm.lmcache_mp_connector"
    ].LMCacheMPConnector = UpstreamConnector
    modules[
        "lmcache.integration.vllm.vllm_multi_process_adapter"
    ].LMCacheMPWorkerAdapter = LMCacheMPWorkerAdapter
    modules[
        "lmcache.v1.multiprocess.protocols.engine"
    ].get_protocol_definitions = get_engine_protocol_definitions
    modules["lmcache.v1.multiprocess.protocols.base"].RequestType = RequestType
    protocol_definitions = {
        RequestType.REGISTER_KV_CACHE: ProtocolDefinition(
            [int, object, str, int, object, object, list]
        )
    }

    def get_payload_classes(request_type):
        return protocol_definitions[request_type].payload_classes

    modules["lmcache.v1.multiprocess.protocol"].get_payload_classes = (
        get_payload_classes
    )
    modules["lmcache.v1.multiprocess.mq"].get_payload_classes = get_payload_classes
    modules["lmcache.v1.multiprocess.server"].get_payload_classes = (
        get_payload_classes
    )
    modules[
        "lmcache.v1.multiprocess.transfer_context.worker_transfer"
    ].LMCacheDrivenTransferContext = LMCacheDrivenTransferContext
    modules[
        "lmcache.v1.multiprocess.transfer_context"
    ].LMCacheDrivenTransferContext = LMCacheDrivenTransferContext
    modules[
        "lmcache.v1.multiprocess.engine_context"
    ].LayoutDescRegistry = LayoutDescRegistry
    modules[
        "lmcache.v1.multiprocess.modules.lmcache_driven_transfer"
    ].LMCacheDrivenTransferModule = LMCacheDrivenTransferModule
    modules["lmcache.v1.distributed.api"].PrefetchRequestSpec = PrefetchRequestSpec
    modules["lmcache.v1.distributed.api"].ObjectKey = ObjectKey
    modules[
        "lmcache.v1.distributed.storage_manager"
    ].PrefetchRequestSpec = PrefetchRequestSpec
    modules[
        "lmcache.v1.distributed.storage_controllers.prefetch_controller"
    ].PrefetchRequestSpec = PrefetchRequestSpec
    modules[
        "lmcache.v1.distributed.storage_controllers.prefetch_controller"
    ].PrefetchController = PrefetchController
    modules[
        "lmcache.v1.multiprocess.modules.lookup"
    ].PrefetchRequestSpec = PrefetchRequestSpec
    modules["lmcache.v1.multiprocess.modules.lookup"].LookupModule = LookupModule
    modules[
        "lmcache.integration.vllm.lmcache_connector_v1_085"
    ].LMCacheConnectorV1Impl = OriginalImpl

    modules["lmcache"].v1 = modules["lmcache.v1"]
    modules["lmcache"].integration = modules["lmcache.integration"]
    modules["lmcache.v1"].multiprocess = modules["lmcache.v1.multiprocess"]
    modules["lmcache.v1.multiprocess"].server = modules[
        "lmcache.v1.multiprocess.server"
    ]
    modules["lmcache.v1.multiprocess"].modules = modules[
        "lmcache.v1.multiprocess.modules"
    ]
    modules["lmcache.v1.multiprocess.modules"].lmcache_driven_transfer = modules[
        "lmcache.v1.multiprocess.modules.lmcache_driven_transfer"
    ]
    modules["lmcache.v1.multiprocess.modules"].blend = modules[
        "lmcache.v1.multiprocess.modules.blend"
    ]
    modules["lmcache.v1.multiprocess.modules"].blend_v3 = modules[
        "lmcache.v1.multiprocess.modules.blend_v3"
    ]
    modules["lmcache.v1"].platform = modules["lmcache.v1.platform"]
    modules["lmcache.v1.platform"].base = modules["lmcache.v1.platform.base"]
    modules["lmcache.v1.platform.base"].event_ipc = modules[
        "lmcache.v1.platform.base.event_ipc"
    ]
    modules["lmcache.v1"].cache_engine = modules["lmcache.v1.cache_engine"]
    modules["lmcache.v1"].protocol = modules["lmcache.v1.protocol"]
    modules["lmcache.v1"].compute = modules["lmcache.v1.compute"]
    modules["lmcache.v1.compute"].attention = modules["lmcache.v1.compute.attention"]
    modules["lmcache.v1.compute.attention"].utils = modules[
        "lmcache.v1.compute.attention.utils"
    ]
    modules["lmcache.v1.compute"].blend = modules["lmcache.v1.compute.blend"]
    modules["lmcache.v1.compute"].models = modules["lmcache.v1.compute.models"]
    modules["lmcache.v1.compute.blend"].blender = modules[
        "lmcache.v1.compute.blend.blender"
    ]
    modules["lmcache.v1.compute.models"].utils = modules[
        "lmcache.v1.compute.models.utils"
    ]
    modules["lmcache.v1.compute.models"].base = modules[
        "lmcache.v1.compute.models.base"
    ]
    modules["lmcache.v1"].gpu_connector = modules["lmcache.v1.gpu_connector"]
    modules["lmcache.v1.gpu_connector"].gpu_connectors = modules[
        "lmcache.v1.gpu_connector.gpu_connectors"
    ]
    modules["lmcache.v1"].storage_backend = modules["lmcache.v1.storage_backend"]
    modules["lmcache.v1"].transfer_channel = modules["lmcache.v1.transfer_channel"]
    modules["lmcache.integration"].vllm = modules["lmcache.integration.vllm"]
    modules["lmcache.integration.vllm"].kv_cache_group_edits = modules[
        "lmcache.integration.vllm.kv_cache_group_edits"
    ]
    modules["lmcache.integration.vllm"].vllm_multi_process_adapter = modules[
        "lmcache.integration.vllm.vllm_multi_process_adapter"
    ]
    modules["lmcache.integration.vllm"].lmcache_mp_connector = modules[
        "lmcache.integration.vllm.lmcache_mp_connector"
    ]
    modules["vllm"].v1 = modules["vllm.v1"]
    modules["vllm.v1"].worker = modules["vllm.v1.worker"]
    modules["vllm.v1.worker"].gpu_worker = modules["vllm.v1.worker.gpu_worker"]
    modules["vllm"].distributed = modules["vllm.distributed"]
    modules["vllm.distributed"].kv_transfer = modules["vllm.distributed.kv_transfer"]
    modules["vllm.distributed.kv_transfer"].kv_connector = modules[
        "vllm.distributed.kv_transfer.kv_connector"
    ]
    modules["vllm.distributed.kv_transfer.kv_connector"].v1 = modules[
        "vllm.distributed.kv_transfer.kv_connector.v1"
    ]
    modules["vllm.distributed.kv_transfer.kv_connector.v1"].multi_connector = modules[
        "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector"
    ]
    return modules


def purge_modules(*prefixes: str) -> None:
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            del sys.modules[name]
