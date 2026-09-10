#!/usr/bin/env python3
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Repeated long-prefix smoke for MP cache reuse through a disagg proxy.

Sends a long repeated-prefix chat completion request through the proxy, optionally
repeats it, and prints LMCache MP metrics from /metrics and /status.

Any example with an MP server + disagg proxy can use this script once the stack
is running.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class SmokeConfig:
    title: str
    proxy_url: str
    lmcache_url: str
    model: str
    max_tokens: int
    num_runs: int
    timeout: float
    prompt_sentence: str
    prompt_repeat: int
    chunk_size: int


@dataclass
class MpMetrics:
    requested_tokens: str = "?"
    hit_tokens: str = "?"
    l1_bytes: str = "?"
    l1_objects: str = "?"
    l1_mem_bytes: str = "?"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw is not None else default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def fetch_text(base_url: str, path: str, timeout: float = 10.0) -> str:
    try:
        with urllib.request.urlopen(
            f"{base_url.rstrip('/')}/{path.lstrip('/')}",
            timeout=timeout,
        ) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError):
        return ""


def _metric_value(metrics_text: str, name: str) -> str:
    for line in metrics_text.splitlines():
        if line.startswith(name):
            return line.strip().split()[-1]
    return "?"


def fetch_mp_metrics(lmcache_url: str) -> MpMetrics:
    metrics = MpMetrics()
    metrics_text = fetch_text(lmcache_url, "metrics")
    metrics.requested_tokens = _metric_value(
        metrics_text, "lmcache_mp_lookup_requested_tokens_total"
    )
    metrics.hit_tokens = _metric_value(metrics_text, "lmcache_mp_lookup_hit_tokens_total")
    metrics.l1_bytes = _metric_value(metrics_text, "lmcache_mp_l1_memory_usage_bytes ")

    status_text = fetch_text(lmcache_url, "status")
    if status_text:
        try:
            manager = json.loads(status_text)["storage_manager"]["l1_manager"]
            metrics.l1_objects = str(manager["total_object_count"])
            metrics.l1_mem_bytes = str(manager["memory_used_bytes"])
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return metrics


def format_mp_metrics(label: str, metrics: MpMetrics) -> str:
    return (
        f"  [{label}] req={metrics.requested_tokens} "
        f"hit={metrics.hit_tokens} l1_bytes={metrics.l1_bytes} "
        f"l1_objects={metrics.l1_objects} l1_mem_bytes={metrics.l1_mem_bytes}"
    )


def print_mp_metrics(label: str, lmcache_url: str) -> None:
    print(format_mp_metrics(label, fetch_mp_metrics(lmcache_url)))


def build_long_prompt_payload(
    model: str,
    prompt_sentence: str,
    prompt_repeat: int,
    max_tokens: int,
) -> dict[str, object]:
    prompt = prompt_sentence * prompt_repeat
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }


def send_chat_completion(
    proxy_url: str,
    payload: dict[str, object],
    timeout: float,
) -> tuple[dict[str, object], float]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{proxy_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            response = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError:
        raise
    return response, time.perf_counter() - start


def run_smoke(config: SmokeConfig) -> None:
    print(config.title)
    print(f"proxy={config.proxy_url}  model={config.model}")
    print(
        f"max_tokens={config.max_tokens}  num_runs={config.num_runs}  "
        f"repeat={config.prompt_repeat}"
    )
    print()

    if not fetch_text(config.proxy_url, "v1/models"):
        raise SystemExit(f"Proxy is not reachable at {config.proxy_url}")

    prompt = config.prompt_sentence * config.prompt_repeat
    payload = build_long_prompt_payload(
        config.model,
        config.prompt_sentence,
        config.prompt_repeat,
        config.max_tokens,
    )
    prompt_tokens_est = len(prompt) // 4
    print(
        f"prompt_chars={len(prompt)} prompt_tokens_est={prompt_tokens_est} "
        f"chunks_est={prompt_tokens_est // config.chunk_size}"
    )
    print()
    print("--- before ---")
    print_mp_metrics("BEFORE", config.lmcache_url)

    for run in range(1, config.num_runs + 1):
        print()
        start = time.perf_counter()
        try:
            response, elapsed = send_chat_completion(
                config.proxy_url, payload, config.timeout
            )
        except urllib.error.HTTPError as exc:
            elapsed = time.perf_counter() - start
            err = exc.read().decode("utf-8", errors="replace").strip()
            print(f"  Run {run} FAILED: HTTP {exc.code} elapsed={elapsed:.3f}s")
            print(err[:500] + ("..." if len(err) > 500 else ""))
            raise SystemExit(1) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise SystemExit(f"Run {run} failed: {exc}") from exc

        usage = response.get("usage", {})
        text = response["choices"][0]["message"]["content"].strip()
        print(
            f"  Run {run}: elapsed={elapsed:.3f}s "
            f"prompt_tokens={usage.get('prompt_tokens', '?')} "
            f"completion_tokens={usage.get('completion_tokens', '?')} "
            f"total_tokens={usage.get('total_tokens', '?')}"
        )
        print(
            f"         reply: {text[:120]}..."
            if len(text) > 120
            else f"         reply: {text}"
        )
        print_mp_metrics(f"AFTER R{run}", config.lmcache_url)

    print()
    print(f"[DONE] {config.num_runs} run(s) complete.")


def build_config(args: argparse.Namespace) -> SmokeConfig:
    proxy_port = args.proxy_port
    proxy_url = args.proxy_url or f"http://localhost:{proxy_port}"
    lmcache_url = args.lmcache_url or (
        f"http://localhost:{args.lmcache_http_port}"
    )
    return SmokeConfig(
        title=args.title,
        proxy_url=proxy_url,
        lmcache_url=lmcache_url,
        model=args.model,
        max_tokens=args.max_tokens,
        num_runs=args.num_runs,
        timeout=args.timeout,
        prompt_sentence=args.prompt_sentence,
        prompt_repeat=args.prompt_repeat,
        chunk_size=args.chunk_size,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    proxy_port = _env_int("PROXY_PORT", 9110)
    parser = argparse.ArgumentParser(
        description="Long-prefix MP cache smoke through a disagg proxy.",
    )
    parser.add_argument(
        "--title",
        default=_env_str("SMOKE_TITLE", "=== LMCache long-prompt cache smoke ==="),
    )
    parser.add_argument("--proxy-url", default=os.environ.get("PROXY_URL"))
    parser.add_argument("--proxy-port", type=int, default=proxy_port)
    parser.add_argument("--lmcache-url", default=os.environ.get("LMCACHE_URL"))
    parser.add_argument(
        "--lmcache-http-port",
        type=int,
        default=_env_int("LMCACHE_HTTP_PORT", 8080),
    )
    parser.add_argument(
        "--model",
        default=_env_str(
            "MODEL_NAME",
            _env_str("SERVED_MODEL_NAME", "Qwen3-8B"),
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=_env_int("MAX_TOKENS", 32))
    parser.add_argument("--num-runs", type=int, default=_env_int("NUM_RUNS", 2))
    parser.add_argument("--timeout", type=float, default=_env_float("TIMEOUT", 600.0))
    parser.add_argument(
        "--prompt-sentence",
        default=_env_str(
            "PROMPT_SENTENCE",
            "Explain KV cache tiering in LLM serving systems. ",
        ),
    )
    parser.add_argument(
        "--prompt-repeat",
        type=int,
        default=_env_int("PROMPT_REPEAT", 100),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=_env_int("LMCACHE_CHUNK_SIZE", 256),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run_smoke(build_config(parse_args(argv)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
