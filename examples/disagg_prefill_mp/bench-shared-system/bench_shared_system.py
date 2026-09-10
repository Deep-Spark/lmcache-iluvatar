#!/usr/bin/env python3
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""MP 2P2D shared-system, multi-user concurrent benchmark client."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI


@dataclass
class RequestResult:
    user_id: int
    phase: str
    round_idx: int
    ttft_ms: float
    e2e_ms: float
    prompt_tokens: int
    completion_tokens: int
    http_ok: bool
    error: str | None = None


def _load_tokenizer(model_path: str):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
    except Exception:
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception as exc:
            print(f"warning: tokenizer unavailable ({exc}); using char/4 estimate", file=sys.stderr)
            return None


def _repeat_to_token_length(tokenizer, seed_text: str, target_tokens: int) -> str:
    if target_tokens <= 0:
        return ""
    if tokenizer is None:
        unit = seed_text or "hi "
        text = unit
        while len(text) // 4 < target_tokens:
            text += unit
        return text[: target_tokens * 4]

    unit = seed_text or "context "
    text = unit
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
        text += unit
    tokens = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    return tokenizer.decode(tokens, skip_special_tokens=True)


def _build_shared_system(tokenizer, target_tokens: int) -> str:
    body = _repeat_to_token_length(
        tokenizer,
        "Shared system policy for KV cache tiering evaluation. ",
        target_tokens,
    )
    return f"[SHARED_SYSTEM]\n{body}"


def _build_user_unique(tokenizer, user_id: int, target_tokens: int) -> str:
    seed = f"user-{user_id} private context. "
    return _repeat_to_token_length(tokenizer, seed, target_tokens)


def _parse_prometheus_counter(metrics_text: str, metric_name: str) -> float | None:
    total = 0.0
    found = False
    for line in metrics_text.splitlines():
        if line.startswith("#"):
            continue
        if not line.startswith(metric_name):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        total += float(parts[-1])
        found = True
    return total if found else None


def fetch_metrics(url: str) -> dict[str, float]:
    req = urllib.request.Request(f"{url.rstrip('/')}/metrics", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"warning: failed to fetch metrics from {url}: {exc}", file=sys.stderr)
        return {}

    out: dict[str, float] = {}
    for name in (
        "lmcache_mp_lookup_requested_tokens_total",
        "lmcache_mp_lookup_hit_tokens_total",
        "lmcache_mp_l1_memory_usage_bytes",
    ):
        value = _parse_prometheus_counter(text, name)
        if value is not None:
            out[name] = value
    return out


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def summarize_phase(results: list[RequestResult], phase: str) -> dict[str, Any]:
    rows = [r for r in results if r.phase == phase and r.http_ok]
    ttfts = [r.ttft_ms for r in rows if r.ttft_ms >= 0]
    if not ttfts:
        return {
            "phase": phase,
            "successful_requests": len(rows),
            "failed_requests": sum(1 for r in results if r.phase == phase and not r.http_ok),
            "mean_ttft_ms": None,
            "p50_ttft_ms": None,
            "p99_ttft_ms": None,
        }
    return {
        "phase": phase,
        "successful_requests": len(rows),
        "failed_requests": sum(1 for r in results if r.phase == phase and not r.http_ok),
        "mean_ttft_ms": statistics.mean(ttfts),
        "p50_ttft_ms": statistics.median(ttfts),
        "p99_ttft_ms": _percentile(ttfts, 99),
    }


async def _run_one_request(
    client: AsyncOpenAI,
    *,
    model: str,
    shared_system: str,
    user_content: str,
    max_tokens: int,
    user_id: int,
    phase: str,
    round_idx: int,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    async with semaphore:
        messages = [
            {"role": "system", "content": shared_system},
            {"role": "user", "content": user_content},
        ]
        start = time.perf_counter()
        first_token_at: float | None = None
        prompt_tokens = 0
        completion_tokens = 0
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
                stream=True,
                stream_options={"include_usage": True},
            )
            async for chunk in stream:
                if chunk.usage is not None:
                    prompt_tokens = chunk.usage.prompt_tokens or prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens or completion_tokens
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content and first_token_at is None:
                    first_token_at = time.perf_counter()
            end = time.perf_counter()
            ttft_ms = (first_token_at - start) * 1000 if first_token_at is not None else -1.0
            return RequestResult(
                user_id=user_id,
                phase=phase,
                round_idx=round_idx,
                ttft_ms=ttft_ms,
                e2e_ms=(end - start) * 1000,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                http_ok=first_token_at is not None,
                error=None if first_token_at is not None else "no first token",
            )
        except Exception as exc:
            end = time.perf_counter()
            return RequestResult(
                user_id=user_id,
                phase=phase,
                round_idx=round_idx,
                ttft_ms=-1.0,
                e2e_ms=(end - start) * 1000,
                prompt_tokens=0,
                completion_tokens=0,
                http_ok=False,
                error=str(exc),
            )


async def _run_barrier_batch(
    client: AsyncOpenAI,
    *,
    model: str,
    shared_system: str,
    tokenizer,
    user_ids: list[int],
    phase: str,
    round_idx: int,
    user_unique_tokens: int,
    user_query: str,
    max_tokens: int,
    max_inflight: int,
) -> list[RequestResult]:
    semaphore = asyncio.Semaphore(max_inflight)

    async def _one(uid: int) -> RequestResult:
        unique = _build_user_unique(tokenizer, uid, user_unique_tokens)
        user_content = f"{unique}\n{user_query}"
        return await _run_one_request(
            client,
            model=model,
            shared_system=shared_system,
            user_content=user_content,
            max_tokens=max_tokens,
            user_id=uid,
            phase=phase,
            round_idx=round_idx,
            semaphore=semaphore,
        )

    return await asyncio.gather(*[_one(uid) for uid in user_ids])


async def run_benchmark(args: argparse.Namespace) -> int:
    tokenizer = _load_tokenizer(args.model_path)
    shared_system = _build_shared_system(tokenizer, args.shared_system_tokens)

    api_key = os.getenv("OPENAI_API_KEY", "sk-dummy")
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=api_key,
        timeout=args.request_timeout,
    )

    metrics_before = fetch_metrics(args.lmcache_url)
    results: list[RequestResult] = []

    print(f"shared_system_chars={len(shared_system)} target_tokens={args.shared_system_tokens}")
    print(f"proxy={args.base_url} model={args.model} users={args.num_users}")

    warmup_ids = list(range(1, args.warmup_users + 1))
    if warmup_ids:
        print(f"phase=warmup users={warmup_ids}")
        results.extend(
            await _run_barrier_batch(
                client,
                model=args.model,
                shared_system=shared_system,
                tokenizer=tokenizer,
                user_ids=warmup_ids,
                phase="warmup",
                round_idx=0,
                user_unique_tokens=args.user_unique_tokens,
                user_query=args.user_query,
                max_tokens=args.max_tokens,
                max_inflight=args.max_inflight,
            )
        )

    bench_ids = list(range(1, args.num_users + 1))
    for round_idx in range(args.bench_rounds):
        phase = "bench" if round_idx == 0 else f"bench{round_idx + 1}"
        print(f"phase={phase} users={len(bench_ids)} round={round_idx}")
        results.extend(
            await _run_barrier_batch(
                client,
                model=args.model,
                shared_system=shared_system,
                tokenizer=tokenizer,
                user_ids=bench_ids,
                phase=phase,
                round_idx=round_idx,
                user_unique_tokens=args.user_unique_tokens,
                user_query=args.user_query,
                max_tokens=args.max_tokens,
                max_inflight=args.max_inflight,
            )
        )

    metrics_after = fetch_metrics(args.lmcache_url)
    hit_before = metrics_before.get("lmcache_mp_lookup_hit_tokens_total", 0.0)
    hit_after = metrics_after.get("lmcache_mp_lookup_hit_tokens_total", 0.0)
    req_before = metrics_before.get("lmcache_mp_lookup_requested_tokens_total", 0.0)
    req_after = metrics_after.get("lmcache_mp_lookup_requested_tokens_total", 0.0)
    hit_delta = hit_after - hit_before
    req_delta = req_after - req_before

    phases = sorted({r.phase for r in results})
    summary = {
        "model": args.model,
        "base_url": args.base_url,
        "shared_system_tokens": args.shared_system_tokens,
        "num_users": args.num_users,
        "warmup_users": args.warmup_users,
        "bench_rounds": args.bench_rounds,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "lookup_hit_tokens_delta": hit_delta,
        "lookup_requested_tokens_delta": req_delta,
        "phases": [summarize_phase(results, p) for p in phases],
        "failed_requests": sum(1 for r in results if not r.http_ok),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "requests.csv")
    json_path = os.path.join(args.output_dir, "summary.json")

    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write(
            "user_id,phase,round_idx,ttft_ms,e2e_ms,prompt_tokens,"
            "completion_tokens,http_ok,error\n"
        )
        for row in results:
            fh.write(
                f"{row.user_id},{row.phase},{row.round_idx},{row.ttft_ms:.3f},"
                f"{row.e2e_ms:.3f},{row.prompt_tokens},{row.completion_tokens},"
                f"{int(row.http_ok)},{row.error or ''}\n"
            )

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")

    if any(not r.http_ok for r in results):
        print("error: one or more requests failed", file=sys.stderr)
        return 1
    if args.assert_hit and hit_delta <= 0:
        print(
            f"error: expected lookup hit token delta > 0, got {hit_delta}",
            file=sys.stderr,
        )
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MP shared-system multi-user concurrent benchmark."
    )
    parser.add_argument("--base-url", default=os.getenv("PROXY_BASE_URL", "http://127.0.0.1:9110/v1"))
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "Qwen3-32B-W8A8"))
    parser.add_argument("--model-path", default=os.getenv("MODEL_PATH", "/data/nlp/Qwen3-32B-W8A8"))
    parser.add_argument("--lmcache-url", default=os.getenv("LMCACHE_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--shared-system-tokens", type=int, default=int(os.getenv("SHARED_SYSTEM_TOKENS", "1500")))
    parser.add_argument("--user-unique-tokens", type=int, default=int(os.getenv("USER_UNIQUE_TOKENS", "32")))
    parser.add_argument("--user-query", default=os.getenv("USER_QUERY", "Summarize the system context in one sentence."))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS", "32")))
    parser.add_argument("--num-users", type=int, default=int(os.getenv("NUM_USERS", "16")))
    parser.add_argument("--warmup-users", type=int, default=int(os.getenv("WARMUP_USERS", "1")))
    parser.add_argument("--bench-rounds", type=int, default=int(os.getenv("BENCH_ROUNDS", "1")))
    parser.add_argument("--max-inflight", type=int, default=int(os.getenv("MAX_INFLIGHT", "8")))
    parser.add_argument("--request-timeout", type=float, default=float(os.getenv("REQUEST_TIMEOUT", "900")))
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--assert-hit",
        action="store_true",
        default=False,
        help="Exit non-zero if lmcache_mp_lookup_hit_tokens_total did not increase.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    raise SystemExit(main())
