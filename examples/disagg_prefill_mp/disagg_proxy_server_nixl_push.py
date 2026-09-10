# Copyright The LMCache Authors.
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""MP disagg proxy with NixlPushConnector kv_transfer_params injection.

This file is the Nixl-aware fork of ``disagg_proxy_server.py`` (kept close to
upstream LMCache ``examples/disagg_prefill_mp/disagg_proxy_server.py`` for
smoke / bench-shared-system).

Consumers:
  - ``nixl-push-1p1d/`` (``--nixl-push`` + ``--skip-kv-notify-wait``)
  - ``p-cache-nixl-transfer-{1p1d,2p2d}/`` (``--nixl-push`` + LMCache telemetry wait)

Alignment notes:
  - D-side ``kv_transfer_params`` (``do_remote_prefill``, P ``remote_engine_id`` /
    host / side-channel port, ``remote_request_id``, ``tp_size``) follow the
    shape used by vLLM
    ``examples/disaggregated/disaggregated_serving/disagg_proxy_pushconnector_demo.py``.
  - Local extensions (not in that vLLM demo): optional LMCache
    ``request_store_finished`` wait; CSV per-instance NIXL metadata; nested RR
    covering the full P×D product; P also receives selected D ``engine_id``.
"""
# Standard
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional
import argparse
import asyncio
import itertools
import json
import os
import threading
import time
import uuid

# Third Party
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Global dictionary to store asyncio.Events indexed by request ID.
# This is shared between the main proxy app and the telemetry app.
pending_requests: dict[str, asyncio.Event] = {}
pending_requests_lock = threading.Lock()

# Reference to the main event loop (set at startup)
main_event_loop: Optional[asyncio.AbstractEventLoop] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()

    # Startup: Initialize clients
    pref_hosts = global_args.prefiller_host
    pref_ports = global_args.prefiller_port

    def pair_hosts_and_ports(hosts, ports, count=None):
        """
        Flexible host-port pairing with expansion strategies.
        """
        if not isinstance(hosts, list):
            hosts = [hosts]
        if not isinstance(ports, list):
            ports = [ports]
        if len(hosts) == 1 and len(ports) == 1:
            if count is None or count <= 1:
                return [(hosts[0], ports[0])]
            else:
                return [(hosts[0], ports[0] + i) for i in range(count)]
        if len(hosts) == 1:
            return [(hosts[0], p) for p in ports]
        if len(ports) == 1:
            return [(h, ports[0]) for h in hosts]
        if len(hosts) != len(ports):
            raise ValueError(
                "Length mismatch between hosts and ports lists for pairing"
            )
        return list(zip(hosts, ports, strict=False))

    prefill_pairs = pair_hosts_and_ports(
        pref_hosts, pref_ports, global_args.num_prefillers
    )
    pref_nixl = _expand_nixl_prefiller_metadata(
        global_args, len(prefill_pairs)
    )
    for i, (host, port) in enumerate(prefill_pairs):
        prefiller_base_url = f"http://{host}:{int(port)}"
        prefill_client = httpx.AsyncClient(timeout=None, base_url=prefiller_base_url)
        nixl_meta = pref_nixl[i] if pref_nixl else {}
        app.state.prefill_clients.append(
            ClientInfo(
                prefill_client,
                host=host,
                nixl_engine_id=nixl_meta.get("engine_id"),
                nixl_side_channel_host=nixl_meta.get("side_channel_host"),
                nixl_side_channel_port=nixl_meta.get("side_channel_port"),
            )
        )

    # Build decoder clients
    dec_hosts = global_args.decoder_host
    dec_ports = global_args.decoder_port

    decoder_pairs = pair_hosts_and_ports(dec_hosts, dec_ports, global_args.num_decoders)
    dec_nixl = _expand_nixl_decoder_metadata(global_args, len(decoder_pairs))

    for i, (host, port) in enumerate(decoder_pairs):
        decoder_base_url = f"http://{host}:{int(port)}"
        decode_client = httpx.AsyncClient(timeout=None, base_url=decoder_base_url)
        nixl_meta = dec_nixl[i] if dec_nixl else {}
        app.state.decode_clients.append(
            ClientInfo(
                decode_client,
                host=host,
                nixl_engine_id=nixl_meta.get("engine_id"),
            )
        )

    app.state.total_clients = app.state.prefill_clients + app.state.decode_clients

    yield

    # Shutdown: Close clients
    for client in app.state.prefill_clients:
        await client.aclose()
    for client in app.state.decode_clients:
        await client.aclose()


# Main proxy FastAPI app
app = FastAPI(lifespan=lifespan)


def csv_ints(s):
    return [int(x) for x in s.split(",")]


def csv_strs(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def _align_csv(values, count: int, name: str, *, allow_broadcast: bool = False):
    """Return a length-``count`` list.

    When ``allow_broadcast`` is True, a single value expands to all slots
    (useful for side-channel host). Engine ids / ports must match ``count``
    exactly so multi-instance NIXL peers stay unique.
    """
    if values is None:
        return None
    if not isinstance(values, list):
        values = [values]
    if len(values) == count:
        return list(values)
    if allow_broadcast and len(values) == 1:
        return values * count
    raise ValueError(
        f"{name}: expected {count} CSV value(s)"
        + (" or 1 to broadcast" if allow_broadcast else "")
        + f", got {len(values)}"
    )


def _require_unique(values, name: str) -> None:
    """Fail fast when multi-instance NIXL peers share an id or port."""
    if values is None:
        return
    if len(values) != len(set(values)):
        raise ValueError(f"{name}: values must be unique, got {values}")


def _expand_nixl_prefiller_metadata(args, count: int) -> list[dict]:
    """Per-prefiller NIXL metadata; empty when --nixl-push is off."""
    if not getattr(args, "nixl_push", False):
        return []
    engine_ids = _align_csv(
        args.nixl_prefill_engine_id, count, "--nixl-prefill-engine-id"
    )
    hosts = _align_csv(
        args.nixl_prefill_side_channel_host,
        count,
        "--nixl-prefill-side-channel-host",
        allow_broadcast=True,
    )
    ports = _align_csv(
        args.nixl_prefill_side_channel_port,
        count,
        "--nixl-prefill-side-channel-port",
    )
    if engine_ids is None or hosts is None or ports is None:
        raise ValueError("--nixl-push requires prefiller NIXL metadata")
    _require_unique(engine_ids, "--nixl-prefill-engine-id")
    _require_unique([int(p) for p in ports], "--nixl-prefill-side-channel-port")
    return [
        {
            "engine_id": engine_ids[i],
            "side_channel_host": hosts[i],
            "side_channel_port": int(ports[i]),
        }
        for i in range(count)
    ]


def _expand_nixl_decoder_metadata(args, count: int) -> list[dict]:
    """Per-decoder NIXL metadata; empty when --nixl-push is off."""
    if not getattr(args, "nixl_push", False):
        return []
    engine_ids = _align_csv(
        args.nixl_decode_engine_id, count, "--nixl-decode-engine-id"
    )
    if engine_ids is None:
        raise ValueError("--nixl-push requires --nixl-decode-engine-id")
    _require_unique(engine_ids, "--nixl-decode-engine-id")
    return [{"engine_id": engine_ids[i]} for i in range(count)]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--prefiller-host", type=csv_strs, default=["localhost"])
    parser.add_argument("--prefiller-port", type=csv_ints, default=[8100])
    parser.add_argument("--num-prefillers", type=int, default=1)
    parser.add_argument("--decoder-host", type=csv_strs, default=["localhost"])
    parser.add_argument("--decoder-port", type=csv_ints, default=[8200])
    parser.add_argument("--num-decoders", type=int, default=1)
    parser.add_argument("--telemetry-port", type=int, default=5768)
    parser.add_argument(
        "--skip-kv-notify-wait",
        action="store_true",
        help=(
            "After prefiller returns, forward to decoder immediately without "
            "waiting for LMCache request_store_finished telemetry. Use for "
            "pure vLLM NixlPush stacks that do not emit that event."
        ),
    )
    parser.add_argument(
        "--nixl-push",
        action="store_true",
        help=(
            "Inject vLLM NixlPushConnector kv_transfer_params and share "
            "X-Request-Id across P/D (required for PUSH_REG matching). "
            "Does not skip LMCache telemetry wait; combine with "
            "--skip-kv-notify-wait for pure NixlPush stacks."
        ),
    )
    parser.add_argument(
        "--nixl-prefill-engine-id",
        type=csv_strs,
        default=None,
        help=(
            "P-side engine_id(s), CSV aligned with prefillers "
            "(must match prefiller --kv-transfer-config). Single value OK for 1P1D."
        ),
    )
    parser.add_argument(
        "--nixl-decode-engine-id",
        type=csv_strs,
        default=None,
        help=(
            "D-side engine_id(s), CSV aligned with decoders "
            "(must match decoder --kv-transfer-config). Single value OK for 1P1D."
        ),
    )
    parser.add_argument(
        "--nixl-prefill-side-channel-host",
        type=csv_strs,
        default=["localhost"],
        help=(
            "Host(s) of P VLLM_NIXL_SIDE_CHANNEL (D uses this to reach P). "
            "CSV or single value broadcast."
        ),
    )
    parser.add_argument(
        "--nixl-prefill-side-channel-port",
        type=csv_ints,
        default=[5600],
        help=(
            "Port(s) of P VLLM_NIXL_SIDE_CHANNEL, CSV aligned 1:1 with "
            "prefillers (must be unique; single value only for 1P1D)."
        ),
    )
    parser.add_argument(
        "--nixl-tp-size",
        type=int,
        default=2,
        help="Tensor parallel size of the prefiller (passed as tp_size).",
    )
    args = parser.parse_args()
    if args.nixl_push and (
        not args.nixl_prefill_engine_id or not args.nixl_decode_engine_id
    ):
        parser.error(
            "--nixl-push requires --nixl-prefill-engine-id and "
            "--nixl-decode-engine-id"
        )
    if args.nixl_push:
        # Eager length + uniqueness check (lifespan expands again for stepped ports).
        try:
            _expand_nixl_prefiller_metadata(args, args.num_prefillers)
            _expand_nixl_decoder_metadata(args, args.num_decoders)
        except ValueError as exc:
            parser.error(str(exc))
    return args


@dataclass
class ClientInfo:
    client: httpx.AsyncClient
    host: Optional[str] = None
    nixl_engine_id: Optional[str] = None
    nixl_side_channel_host: Optional[str] = None
    nixl_side_channel_port: Optional[int] = None

    async def aclose(self):
        await self.client.aclose()


# Initialize state
app.state.prefill_clients = []
app.state.decode_clients = []
app.state.total_clients = []


def round_robin_pick_client(clients, idx):
    return clients[idx % len(clients)]


# Nested RR covers the full P×D product (not only diagonal P_i↔D_i).
# For 2P2D over 4 requests: (P0,D0), (P1,D0), (P0,D1), (P1,D1).
_round_robin_counter = itertools.count()


def round_robin_pick_clients() -> tuple[ClientInfo, ClientInfo]:
    idx = next(_round_robin_counter)
    n_p = len(app.state.prefill_clients)
    prefill_client = round_robin_pick_client(app.state.prefill_clients, idx)
    # Advance D every n_p prefiller steps so equal-sized pools still cross-pair.
    decode_client = round_robin_pick_client(
        app.state.decode_clients, idx // max(n_p, 1)
    )
    return prefill_client, decode_client


def _build_headers(**extra: str) -> dict[str, str]:
    """Build common HTTP headers, including auth if OPENAI_API_KEY is set."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    headers.update(extra)
    return headers


async def send_request_to_prefiller(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
    request_id: str,
):
    """
    Send a request to prefiller with request-id header.
    """
    headers = _build_headers(**{"X-Request-Id": request_id})
    response = await client.post(endpoint, json=req_data, headers=headers)
    response.raise_for_status()
    return response


async def send_request_to_decoder(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
    request_id: Optional[str] = None,
):
    """
    Send a request to decoder service.

    When ``request_id`` is set (NixlPush), it must match the prefiller
    X-Request-Id so P can match D's PUSH_REG against finished blocks.
    """
    extra = {"X-Request-Id": request_id} if request_id else {}
    headers = _build_headers(**extra)
    response = await client.post(endpoint, json=req_data, headers=headers)
    response.raise_for_status()
    return response


def _sse_chunk(data: dict) -> bytes:
    return ("data: " + json.dumps(data, separators=(",", ":")) + "\n\n").encode()


def _extract_prefill_text(prefill_output: dict, endpoint: str) -> str:
    choice = (prefill_output.get("choices") or [{}])[0]
    if endpoint == "/v1/chat/completions":
        message = choice.get("message") or {}
        return message.get("content") or " "
    return choice.get("text") or " "


def _build_prefill_stream_chunks(prefill_output: dict, endpoint: str) -> list[bytes]:
    prefill_text = _extract_prefill_text(prefill_output, endpoint)
    if endpoint == "/v1/chat/completions":
        initial_chunk = {
            "id": prefill_output["id"],
            "object": "chat.completion.chunk",
            "created": prefill_output["created"],
            "model": prefill_output["model"],
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
        }
        head_chunk = {
            "id": prefill_output["id"],
            "object": "chat.completion.chunk",
            "created": prefill_output["created"],
            "model": prefill_output["model"],
            "disagg_source": "prefill",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": prefill_text},
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
        }
        return [_sse_chunk(initial_chunk), _sse_chunk(head_chunk)]

    head_chunk = {
        "id": prefill_output["id"],
        "object": "text_completion",
        "created": prefill_output["created"],
        "model": prefill_output["model"],
        "disagg_source": "prefill",
        "choices": [
            {
                "index": 0,
                "text": prefill_text,
                "logprobs": None,
                "finish_reason": None,
                "stop_reason": None,
            }
        ],
        "usage": None,
    }
    return [_sse_chunk(head_chunk)]


async def stream_service_response(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
    request_id: Optional[str] = None,
):
    """
    Asynchronously stream the response from a service.
    """
    extra = {"X-Request-Id": request_id} if request_id else {}
    headers = _build_headers(**extra)
    async with client.stream(
        "POST", endpoint, json=req_data, headers=headers
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


def _nixl_prefill_kv_params(decode_client: ClientInfo) -> dict:
    """kv_transfer_params for P (producer / do_remote_decode).

    Uses the *selected* decoder's engine_id so independent RR P_i↔D_j works.
    """
    if not decode_client.nixl_engine_id:
        raise RuntimeError(
            "NixlPush enabled but selected decoder has no nixl_engine_id"
        )
    return {
        "do_remote_decode": True,
        "remote_engine_id": decode_client.nixl_engine_id,
        "tp_size": global_args.nixl_tp_size,
    }


def _nixl_decode_kv_params(
    request_id: str, prefill_client: ClientInfo
) -> dict:
    """kv_transfer_params for D (consumer / do_remote_prefill).

    Uses the *selected* prefiller's engine_id + side-channel so D reaches P_i.
    """
    if not prefill_client.nixl_engine_id:
        raise RuntimeError(
            "NixlPush enabled but selected prefiller has no nixl_engine_id"
        )
    if (
        not prefill_client.nixl_side_channel_host
        or prefill_client.nixl_side_channel_port is None
    ):
        raise RuntimeError(
            "NixlPush enabled but selected prefiller lacks side-channel host/port"
        )
    return {
        "do_remote_prefill": True,
        "remote_engine_id": prefill_client.nixl_engine_id,
        "remote_host": prefill_client.nixl_side_channel_host,
        "remote_port": prefill_client.nixl_side_channel_port,
        "remote_request_id": request_id,
        "tp_size": global_args.nixl_tp_size,
    }


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())[:16]


def create_pending_request(request_id: str) -> asyncio.Event:
    """Create an asyncio.Event for a request and store it."""
    event = asyncio.Event()
    with pending_requests_lock:
        pending_requests[request_id] = event
    return event


def remove_pending_request(request_id: str):
    """Remove a pending request from the dictionary."""
    with pending_requests_lock:
        pending_requests.pop(request_id, None)


def notify_request(chatcmpl_request_id: str) -> bool:
    """
    Notify a pending request that the KV store is complete.
    Returns True if the request was found and notified.
    """
    with pending_requests_lock:
        # vLLM wraps the request ID as "chatcmpl-{uuid}-{suffix}",
        # so strip the first and last segments to recover the original UUID.
        request_id = "-".join(chatcmpl_request_id.split("-")[1:])
        request_id = request_id[:16]
        event = pending_requests.get(request_id, None)
        if event:
            # Schedule the event.set() on the main event loop
            if main_event_loop is not None:
                main_event_loop.call_soon_threadsafe(event.set)
            return True
    return False


async def _handle_disagg_request(request: Request, endpoint: str):
    """
    Common handler for disaggregated prefill/decode requests.

    Works for both /v1/completions and /v1/chat/completions — the only
    difference is the *endpoint* path forwarded to the prefiller and decoder.
    """
    try:
        req_data = await request.json()

        # Generate a random request ID
        request_id = generate_request_id()
        logger.info(f"Received {endpoint} request with generated ID: {request_id}")

        # Pick prefill and decode clients
        prefill_client, decode_client = round_robin_pick_clients()
        if nixl_push := bool(getattr(global_args, "nixl_push", False)):
            logger.info(
                f"Request {request_id}: selected P="
                f"{prefill_client.nixl_engine_id or prefill_client.client.base_url} "
                f"D={decode_client.nixl_engine_id or decode_client.client.base_url}"
            )

        # Create event for this request (signaled when KV store finishes).
        # Pure NixlPush stacks skip the wait — they never emit LMCache telemetry.
        skip_kv_notify_wait = bool(
            getattr(global_args, "skip_kv_notify_wait", False)
        )
        event = None if skip_kv_notify_wait else create_pending_request(request_id)

        notify_wait_ms = 0.0
        try:
            # Modify request for prefiller: set max_tokens=1
            prefill_req_data = req_data.copy()
            prefill_req_data["max_tokens"] = 1
            if "max_completion_tokens" in prefill_req_data:
                prefill_req_data["max_completion_tokens"] = 1
            prefill_req_data["stream"] = False
            prefill_req_data.pop("stream_options", None)
            if nixl_push:
                prefill_req_data["kv_transfer_params"] = _nixl_prefill_kv_params(
                    decode_client
                )
                # Prefer min_tokens=0 so P can finish after prompt without
                # forcing an extra decode token when max_tokens=1.
                prefill_req_data.setdefault("min_tokens", 0)

            # Send to prefiller with X-Request-Id header (ignore output)
            prefill_send_time = time.monotonic()
            prefill_response = await send_request_to_prefiller(
                prefill_client.client,
                endpoint,
                prefill_req_data,
                request_id,
            )
            prefill_output = prefill_response.json()
            prefill_first_response_time = time.monotonic()
            prefill_duration = prefill_first_response_time - prefill_send_time
            logger.info(
                f"Request {request_id}: prefill request"
                f" duration = {prefill_duration:.4f}s"
            )

            if skip_kv_notify_wait:
                logger.info(
                    f"Request {request_id}: skip KV notify wait "
                    "(--skip-kv-notify-wait); forwarding to decoder"
                )
            else:
                # Wait for LMCache telemetry (request_store_finished)
                await event.wait()
                notify_time = time.monotonic()
                notify_wait_duration = notify_time - prefill_first_response_time
                notify_wait_ms = notify_wait_duration * 1000.0
                logger.info(
                    f"Request {request_id}: finished saving KV caches after prefill"
                    f" response = {notify_wait_ms:.2f}ms"
                )
                logger.debug(
                    f"Event signaled for {request_id}, forwarding to decoder"
                )

        finally:
            if event is not None:
                remove_pending_request(request_id)

        # Forward original request to decoder (optionally with NixlPush params).
        decode_req_data = req_data.copy()
        if nixl_push:
            decode_req_data["kv_transfer_params"] = _nixl_decode_kv_params(
                request_id, prefill_client
            )
        is_stream = decode_req_data.get("stream", False)

        if is_stream:

            async def generate_stream():
                for chunk in _build_prefill_stream_chunks(prefill_output, endpoint):
                    yield chunk

                first_chunk = True
                async for chunk in stream_service_response(
                    decode_client.client,
                    endpoint,
                    decode_req_data,
                    request_id=request_id if nixl_push else None,
                ):
                    if first_chunk:
                        decode_first_response_time = time.monotonic()
                        gap_ms = (
                            decode_first_response_time - prefill_first_response_time
                        ) * 1000.0
                        logger.info(
                            f"Request {request_id}: latency between prefill first "
                            f"response and decode first response = "
                            f"{gap_ms:.2f}ms"
                        )
                        first_chunk = False
                    yield chunk

            return StreamingResponse(generate_stream(), media_type="application/json")
        else:
            response = await send_request_to_decoder(
                decode_client.client,
                endpoint,
                decode_req_data,
                request_id=request_id if nixl_push else None,
            )
            return JSONResponse(content=response.json())

    except Exception as e:
        # Standard
        import sys
        import traceback

        exc_info = sys.exc_info()
        logger.error(f"Error in {endpoint} endpoint")
        logger.error(str(e))
        logger.error("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    """Handle /v1/completions requests."""
    return await _handle_disagg_request(request, "/v1/completions")


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    """Handle /v1/chat/completions requests."""
    return await _handle_disagg_request(request, "/v1/chat/completions")


@app.get("/v1/models")
async def handle_models():
    """Handle /v1/models requests by forwarding to the first prefiller."""
    try:
        prefill_client = app.state.prefill_clients[0]
        headers = _build_headers()
        response = await prefill_client.client.get("/v1/models", headers=headers)
        response.raise_for_status()
        return JSONResponse(content=response.json())
    except Exception as e:
        logger.error(f"Error in /v1/models endpoint: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500,
        )


# ============================================================================
# Telemetry FastAPI app (runs on separate port)
# ============================================================================

telemetry_app = FastAPI()


@telemetry_app.post("/api/v1/telemetry")
async def handle_telemetry(request: Request):
    """
    Handle telemetry POST requests.

    Expected payload format (from FastAPIRequestTelemetry):
    {
        "event": "request_store_finished",
        "request_ids_set": ["id1", "id2", ...],
        "model_name": "...",
        "world_size": N,
        "kv_rank": K,
    }

    For each request ID in request_ids_set, signal the corresponding
    event if it exists.
    """
    try:
        payload = await request.json()

        event_type = payload.get("event")
        request_ids = payload.get("request_ids_set", [])

        for request_id in request_ids:
            logger.info(
                f"Received telemetry event: {event_type} for request: {request_id}"
            )

        notified_count = 0
        for request_id in request_ids:
            if notify_request(request_id):
                notified_count += 1
                logger.info(f"Notified request: {request_id}")

        return JSONResponse(
            content={
                "status": "ok",
                "notified": notified_count,
                "total": len(request_ids),
            }
        )

    except Exception as e:
        logger.error(f"Error processing telemetry: {e}")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500,
        )


def run_telemetry_server(host: str, port: int):
    """Run the telemetry server in a separate thread."""
    # Third Party
    import uvicorn

    uvicorn.run(telemetry_app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    # Third Party
    import uvicorn

    # Start telemetry server in a background thread
    telemetry_thread = threading.Thread(
        target=run_telemetry_server,
        args=(global_args.host, global_args.telemetry_port),
        daemon=True,
    )
    telemetry_thread.start()
    logger.info(
        f"Telemetry server started on {global_args.host}:{global_args.telemetry_port}"
    )

    # Run main proxy server
    uvicorn.run(app, host=global_args.host, port=global_args.port)
