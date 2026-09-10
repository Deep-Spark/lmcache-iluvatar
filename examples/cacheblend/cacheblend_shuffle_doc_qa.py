#!/usr/bin/env python3
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CacheBlend benchmark with document derangements and blend separators.

This is intentionally close to LMCache's multi_doc_qa benchmark: warm up by
storing each document as its own blended segment, then query with derangements
of the same documents. The query round changes document order while preserving
the `blend_special_str` boundaries required by CacheBlend's segment database.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable


OUTPUT_FILE: str | None = None


def write_resp(text: str) -> None:
    if OUTPUT_FILE:
        with Path(OUTPUT_FILE).open("a", encoding="utf-8") as resp_file:
            resp_file.write(text)
    else:
        sys.stdout.write(text)
        sys.stdout.flush()


def has_content(chunk) -> bool:
    return bool(chunk.choices and chunk.choices[0].text)


def extract_content(chunk) -> str:
    return chunk.choices[0].text or ""


async def process_single_prompt(
    client: Any,
    model: str,
    prompt_ids: list[int],
    label: str,
    output_len: int,
    semaphore: asyncio.Semaphore,
) -> float:
    async with semaphore:
        write_resp(f"\n--- {label} ---\n")
        start_time = time.time()
        first_token_time: float | None = None
        response = await client.completions.create(
            model=model,
            prompt=prompt_ids,
            max_tokens=output_len,
            temperature=0.0,
            stream=True,
            extra_body={"ignore_eos": True},
        )

        chunks: list[str] = []
        async for chunk in response:
            if has_content(chunk):
                content = extract_content(chunk)
                if first_token_time is None and content:
                    first_token_time = time.time()
                chunks.append(content)

        write_resp(f"\nResponse: {''.join(chunks)}\n")
        ttft = first_token_time - start_time if first_token_time is not None else 0.0
        write_resp(f"TTFT: {ttft:.3f} seconds\n")
        return ttft


async def run_prompts(
    client: Any,
    model: str,
    prompts: list[list[int]],
    labels: list[str],
    output_len: int,
    max_inflight_requests: int,
) -> list[float]:
    semaphore = asyncio.Semaphore(max_inflight_requests)
    tasks = [
        process_single_prompt(client, model, prompt, label, output_len, semaphore)
        for prompt, label in zip(prompts, labels, strict=True)
    ]
    return await asyncio.gather(*tasks)


def all_derangements(n: int) -> list[tuple[int, ...]]:
    return [
        perm
        for perm in itertools.permutations(range(n))
        if all(perm[i] != i for i in range(n))
    ]


def random_derangement(n: int, rng: random.Random) -> tuple[int, ...]:
    for _ in range(500_000):
        perm = list(range(n))
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            return tuple(perm)
    raise RuntimeError(f"failed to sample a derangement for n={n}")


def pick_derangements(n: int, count: int, rng: random.Random) -> list[tuple[int, ...]]:
    if count < 1:
        return []
    if n <= 8:
        derangements = all_derangements(n)
        if len(derangements) < count:
            raise ValueError(
                f"need {count} distinct derangements for n={n}, "
                f"but only {len(derangements)} exist"
            )
        rng.shuffle(derangements)
        return derangements[:count]

    seen: set[tuple[int, ...]] = set()
    for _ in range(count * 200_000):
        if len(seen) >= count:
            break
        seen.add(random_derangement(n, rng))
    if len(seen) < count:
        raise RuntimeError(
            f"could not collect {count} distinct derangements for n={n}; "
            "try a different --random-seed"
        )
    return list(seen)[:count]


def generate_documents(num_documents: int, document_length: int) -> list[str]:
    return [
        f"{i} " + " ".join(["hi"] * document_length) for i in range(num_documents)
    ]


def generate_warmup_prompt_ids(
    doc_prompts: list[str],
    sys_prompt: str,
    query_prompt: str,
    blend_special_str: str,
    tokenizer,
    offset: int = 1,
) -> list[list[int]]:
    blend_special_ids = tokenizer.encode(blend_special_str)[offset:]
    sys_prompt_ids = tokenizer.encode(sys_prompt)
    query_prompt_ids = tokenizer.encode(query_prompt)[offset:]
    return [
        sys_prompt_ids
        + blend_special_ids
        + tokenizer.encode(doc_prompt)[offset:]
        + blend_special_ids
        + query_prompt_ids
        for doc_prompt in doc_prompts
    ]


def generate_deranged_prompt_ids(
    doc_prompts: list[str],
    sys_prompts: list[str],
    query_prompts: list[str],
    derangements: Iterable[tuple[int, ...]],
    blend_special_str: str,
    tokenizer,
    offset: int = 1,
) -> list[list[int]]:
    blend_special_ids = tokenizer.encode(blend_special_str)[offset:]
    prompt_ids: list[list[int]] = []
    for sys_prompt, query_prompt, perm in zip(
        sys_prompts, query_prompts, derangements, strict=True
    ):
        ids = tokenizer.encode(sys_prompt)
        for doc_index in perm:
            ids += blend_special_ids + tokenizer.encode(doc_prompts[doc_index])[offset:]
        ids += blend_special_ids + tokenizer.encode(query_prompt)[offset:]
        prompt_ids.append(ids)
    return prompt_ids


def validate_args(args: argparse.Namespace) -> None:
    if args.num_documents < 2:
        raise SystemExit("--num-documents must be >= 2 for derangement queries")
    num_requests = args.num_requests or args.num_documents - 1
    if num_requests < 1:
        raise SystemExit("--num-requests must be >= 1")
    if args.document_length < 1:
        raise SystemExit("--document-length must be >= 1")
    if args.max_inflight_requests < 1:
        raise SystemExit("--max-inflight-requests must be >= 1")


async def main(args: argparse.Namespace) -> None:
    from openai import AsyncOpenAI
    from transformers import AutoTokenizer

    validate_args(args)
    rng = random.Random(args.random_seed)
    client = AsyncOpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="sk-dummy")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    doc_prompts = generate_documents(args.num_documents, args.document_length)
    num_requests = args.num_requests or args.num_documents - 1
    derangements = pick_derangements(args.num_documents, num_requests, rng)

    sys_prompt = "You are a helpful assistant."
    query_prompt = "What's up? how are you recently?"
    warmup_prompts = generate_warmup_prompt_ids(
        doc_prompts,
        sys_prompt,
        query_prompt,
        args.blend_special_str,
        tokenizer,
        offset=1,
    )
    query_prompts = generate_deranged_prompt_ids(
        doc_prompts,
        [sys_prompt] * len(derangements),
        [query_prompt] * len(derangements),
        derangements,
        args.blend_special_str,
        tokenizer,
        offset=1,
    )

    write_resp("------warm up round------\n")
    warmup_start = time.time()
    warmup_ttfts = await run_prompts(
        client,
        args.model,
        warmup_prompts,
        [f"Warmup doc {i + 1}/{len(warmup_prompts)}" for i in range(len(warmup_prompts))],
        args.output_len,
        args.max_inflight_requests,
    )
    warmup_end = time.time()

    write_resp("------query round------\n")
    if args.sleep_time_after_warmup > 0:
        write_resp(f"Sleeping for {args.sleep_time_after_warmup} seconds after warmup...\n")
        time.sleep(args.sleep_time_after_warmup)

    query_start = time.time()
    query_ttfts = await run_prompts(
        client,
        args.model,
        query_prompts,
        [
            f"Query {i + 1}/{len(derangements)} derangement {perm}"
            for i, perm in enumerate(derangements)
        ],
        args.output_len,
        args.max_inflight_requests,
    )
    query_end = time.time()

    warmup_mean = sum(warmup_ttfts) / len(warmup_ttfts)
    query_mean = sum(query_ttfts) / len(query_ttfts)
    write_resp("\n=== CACHEBLEND SHUFFLE BENCHMARK RESULTS ===\n")
    write_resp(f"Warmup round mean TTFT: {warmup_mean:.3f}s\n")
    write_resp(f"Warmup round time: {warmup_end - warmup_start:.3f}s\n")
    write_resp(f"Warmup round prompt count: {len(warmup_ttfts)}\n")
    write_resp(f"Query round mean TTFT: {query_mean:.3f}s\n")
    write_resp(f"Query round time: {query_end - query_start:.3f}s\n")
    write_resp(f"Query round prompt count: {len(query_ttfts)}\n")
    write_resp(f"Actual TTFT gain: {(warmup_mean / query_mean):.2f}x\n")


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CacheBlend benchmark with document derangements."
    )
    parser.add_argument("--num-documents", type=int, required=True)
    parser.add_argument("--document-length", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--blend-special-str", type=str, default=" # # ")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--max-inflight-requests", type=int, default=1)
    parser.add_argument("--sleep-time-after-warmup", type=float, default=0.0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    return parser


if __name__ == "__main__":
    parsed_args = create_argument_parser().parse_args()
    OUTPUT_FILE = parsed_args.output
    asyncio.run(main(parsed_args))
