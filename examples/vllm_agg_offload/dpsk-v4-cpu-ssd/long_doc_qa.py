# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_long_document_qa_throughput.py

"""
Local LongQA client for dpsk-v4-cpu-ssd offload hit-rate experiments.

Prefix hit-rate mode: cold stores a shorter prefix of each document; query
sends the full document (same doc id + shared body prefix). Expected offload
hit rate ≈ warmup_document_length / document_length (chunk-aligned).

Commandline arguments:
    --num-documents: Number of documents (same set for cold and query).

    --document-length: Full document body length for the query round
                       (token-proxy words). Optional, default: 20000.

    --warmup-document-length: Cold-round body length. Must be <= document-length.
                              If omitted, derived from --target-hit-rate
                              (default 1.0 → same as document-length).

    --target-hit-rate: Fraction in (0, 1] used when --warmup-document-length
                       is omitted. Example: 0.8 with document-length=1e6 →
                       warmup≈800k (aligned down to --chunk-size).

    --chunk-size: Align warmup length down to this size (LMCache chunk).
                  Default: 256.

    --output-len: The number of tokens to generate for each prompt.
                  (Optional, default: 100)

    --repeat-count: The number of times to repeat each prompt.
                    (Optional, default: 2)

    --repeat-mode: The mode to repeat prompts. The supported modes are:
        - 'random': shuffle the prompts randomly. (Default)
        - 'tile': the entire prompt list is repeated in sequence.
        - 'interleave': each prompt is repeated consecutively before
                        moving to the next element.

    --shuffle-seed: Random seed when the repeat mode is "random".
                    (Optional, default: 0)

    --host: Host to query the vLLM server
    --port: Port to query the vLLM server
    --base-url: Base URL to query the LLM server (exclusive with --host/--port)

    --model: Model name

    --max-inflight-requests: Maximum number of in-flight requests. Default is 2

    --sleep-time-after-warmup: Fixed wait after the cold/warmup round before
                              issuing the warm query.
                              (Optional, default: 600 seconds; set 0 to disable)

    --output: Filename to write all responses to. If omitted, writes to stdout.

    --expected-ttft-gain: Expected minimum speed-up in time-to-first-token
                         (warmup/query) as a factor, e.g. 4.3 for 4.3×. If
                         actual gain is below this, exits.

    --expected-latency-gain: Expected minimum speed-up in total round time
                            (warmup/query) as a factor, e.g. 4.5 for 4.5×.
                            If actual gain is below this, exits.

    --expected-latency: Expected end to end latency for the first query round.
    --completions: Use completions API instead of chat completions API

    --pd-disagg-ttft: For PD disagg proxies, measure TTFT from the decoder's
                      first token. Skips stream chunks tagged with
                      disagg_source=prefill (read via the OpenAI SDK).

    --output-dir: Directory for warmup_round.csv and query_round.csv
                  (default: current directory).

    --visualize: Visualize the results

    --eos-token-id: EOS token id. we bias against this token id so we always
                   get the number of output tokens we specify

    --hit-miss-ratio: In query round, control how many of the prompts
    will miss the cache. For example, 3:1 means every fourth repeated prompt
    will miss the cache. 2:2 means 2 hits and 2 misses.

    --trim-fraction: Exclude the smallest X fraction and largest X fraction
    and calculate mean of the rest. For example, 0.1 drops bottom 10% and top 10%.
"""

# Standard
from dataclasses import dataclass
import argparse
import asyncio
import math
import os
import random
import sys
import time

# Third Party
from openai import AsyncOpenAI
import pandas as pd

# Global output filename (set in __main__)
OUTPUT_FILE = None
completions_mode = False
pd_disagg_ttft = False
visualize = False
eos_token_id = None


@dataclass
class RequestStats:
    prompt_id: int
    request_id: str
    request_start: float
    ttft: float
    request_end: float
    successful: bool


def get_url_from_args(args):
    """
    Get the base URL from command line arguments.
    Args:
        args: Command line arguments.
    Returns:
        str: The base URL.
    """
    if args.base_url is not None:
        return args.base_url
    else:
        host = args.host if args.host is not None else "localhost"
        port = args.port if args.port is not None else 8000
        return f"http://{host}:{port}/v1"


def extract_reasoning_content(chunk):
    """
    Extract reasoning content from the response chunk.
    Args:
        chunk: The response chunk from OpenAI Chat Completions API.
    Returns:
        str | None: The reasoning content extracted from the chunk.
            None means no reasoning content in this chunk.
    """
    delta = chunk.choices[0].delta
    potential_reasoning_keys = [
        "reasoning_content",
        "reasoning",
        "tool_calls",
        "tool_call",
        "tool_responses",
    ]
    for key in potential_reasoning_keys:
        if hasattr(delta, key) and getattr(delta, key):
            return getattr(delta, key)
    return None


def extract_normal_content(chunk):
    """
    Extract normal content from the response chunk.
    Args:
        chunk: The response chunk from OpenAI Chat Completions API.
    Returns:
        str | None: The normal content extracted from the chunk.
            None means no normal content in this chunk.
    """
    delta = chunk.choices[0].delta
    if hasattr(delta, "content") and delta.content:
        return chunk.choices[0].delta.content
    return None


def has_content_completions(chunk):
    """
    Completions streaming emits text at choices[0].text.
    """
    return bool(chunk.choices) and (chunk.choices[0].text is not None)


def _chunk_disagg_source(chunk) -> str | None:
    """Read proxy-injected disagg_source from an OpenAI SDK stream chunk."""
    value = getattr(chunk, "disagg_source", None)
    if value:
        return value
    extra = getattr(chunk, "model_extra", None)
    if isinstance(extra, dict):
        return extra.get("disagg_source")
    return None


def has_content(chunk, completions_mode=False):
    """
    Check if the chunk has content in the choices.
    Args:
        chunk: The response chunk from OpenAI Chat Completions API.

    Returns:
        bool: True if content exists, False otherwise.
    """
    if completions_mode:
        return has_content_completions(chunk)

    return (
        chunk.choices
        and chunk.choices[0].delta
        and (
            extract_normal_content(chunk) is not None
            or extract_reasoning_content(chunk) is not None
        )
    )


def extract_content_completions(chunk):
    """
    Extract content from a Completions stream chunk.
    """
    return chunk.choices[0].text or ""


def extract_content(chunk, completions_mode=False):
    """
    Extract content from the response chunk.
    Args:
        chunk: The response chunk from OpenAI Chat Completions API.
    Returns:
        str: The content extracted from the chunk.
    """
    if completions_mode:
        return extract_content_completions(chunk)

    if normal_content := extract_normal_content(chunk):
        return normal_content
    elif reasoning_content := extract_reasoning_content(chunk):
        return reasoning_content
    else:
        return ""


def write_resp(text: str):
    """
    Write text to the specified output file (if any), otherwise to stdout.
    """
    if OUTPUT_FILE:
        with open(OUTPUT_FILE, "a") as resp_file:
            resp_file.write(text)
    else:
        sys.stdout.write(text)


async def process_single_prompt(
    client, model, prompt, prompt_index, total_prompts, output_len, semaphore
) -> RequestStats:
    """
    Process a single prompt with the given client and model.

    Args:
        client: The OpenAI client for making API calls.
        model: The model name to use for generation.
        prompt: The prompt string to be processed.
        prompt_index: Index of the current prompt (0-based).
        total_prompts: Total number of prompts being processed.
        output_len: The maximum number of tokens to generate.
        semaphore: Asyncio semaphore to limit concurrent requests.

    Returns:
        RequestStats: RequestStats object containing the request stats
    """
    async with semaphore:  # Acquire semaphore to limit concurrent requests
        write_resp(f"\n--- Sending prompt {prompt_index + 1}/{total_prompts} ---\n")
        # a request starts once it acquires the semaphore
        start_time = time.time()
        first_token_time = None

        # add stop None so we always get the number of output tokens we specify
        if completions_mode:
            response = await client.completions.create(
                model=model,
                prompt=prompt,
                stream=True,
                max_tokens=output_len,
                temperature=0.0,
                stream_options={"include_usage": True},
                logit_bias={str(eos_token_id): -100}
                if eos_token_id is not None
                else None,
            )
        else:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                max_tokens=output_len,
                temperature=0.0,
                stream_options={"include_usage": True},
                logit_bias={str(eos_token_id): -100}
                if eos_token_id is not None
                else None,
            )

        responses = []
        request_id = None
        pd_prefill_seen = False
        # Collect the response chunks
        async for chunk in response:
            chunk_request_id = getattr(chunk, "id", None)
            if chunk_request_id:
                if request_id is not None and request_id != chunk_request_id:
                    raise RuntimeError(
                        f"stream request ID changed from {request_id} "
                        f"to {chunk_request_id}"
                    )
                request_id = chunk_request_id
            if not chunk.choices:
                continue

            if pd_disagg_ttft:
                is_prefill = _chunk_disagg_source(chunk) == "prefill"
                if is_prefill or not pd_prefill_seen:
                    pd_prefill_seen = True
                    if has_content(chunk, completions_mode):
                        content = extract_content(chunk, completions_mode)
                        if content:
                            responses.append(content)
                    continue

            if has_content(chunk, completions_mode):
                content = extract_content(chunk, completions_mode)
                if first_token_time is None and content != "":
                    first_token_time = time.time()
                responses.append(content)

        end_time = time.time()
        if not request_id:
            raise RuntimeError(
                f"stream for prompt {prompt_index} did not include a request ID"
            )
        final_response = "".join(responses)
        write_resp(
            f"\nResponse of request {prompt_index} "
            f"(request_id={request_id}): {final_response}\n"
        )

        # TTFT < 0 means not successful
        ttft = (first_token_time - start_time) if first_token_time is not None else -1
        return RequestStats(
            prompt_id=prompt_index,
            request_id=request_id,
            request_start=start_time,
            ttft=ttft,
            request_end=end_time,
            successful=ttft > 0,
        )


async def test_long_document_qa(
    client, model, prompts=None, output_len=100, max_inflight_requests=10
) -> list[RequestStats]:
    """
    Test long document QA with the given prompts and sampling parameters.
    Process prompts concurrently with a limit on inflight requests.

    Args:
        client: The OpenAI client for making API calls.
        model: The model name to use for generation.
        prompts: A list of prompt strings to be processed by the LLM.
        output_len: The maximum number of tokens to generate.
        max_inflight_requests: Maximum number of concurrent requests.

    Returns:
        list: request_stats - a list of RequestStats objects
    """
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_inflight_requests)

    # Create tasks for all prompts
    tasks = [
        process_single_prompt(
            client=client,
            model=model,
            prompt=prompt,
            prompt_index=i,
            total_prompts=len(prompts),
            output_len=output_len,
            semaphore=semaphore,
        )
        for i, prompt in enumerate(prompts)
    ]
    # Execute all tasks concurrently and collect results
    # The semaphore will control max concurrent requests
    request_stats = await asyncio.gather(*tasks)

    return request_stats


def repeat_prompts(prompts, repeat_count, mode: str):
    """
    Repeat each prompt in the list for a specified number of times.
    The order of prompts in the output list depends on the mode.

    Args:
        prompts: A list of prompts to be repeated.
        repeat_count: The number of times each prompt is repeated.
        mode: The mode of repetition. Supported modes are:
            - 'random': Shuffle the prompts randomly after repetition.
            - 'tile': Repeat the entire prompt list in sequence.
              Example: [1, 2, 3] -> [1, 2, 3, 1, 2, 3].
            - 'interleave': Repeat each prompt consecutively before moving to
              the next. Example: [1, 2, 3] -> [1, 1, 2, 2, 3, 3].

    Returns:
        A list of repeated prompts in the specified order.

    Raises:
        ValueError: If an invalid mode is provided.
    """
    write_resp(f"Repeat mode:  {mode}\n")
    if mode == "random":
        repeated_prompts = prompts * repeat_count
        random.shuffle(repeated_prompts)
        return repeated_prompts
    elif mode == "tile":
        return prompts * repeat_count
    elif mode == "interleave":
        repeated_prompts = []
        for prompt in prompts:
            repeated_prompts.extend([prompt] * repeat_count)
        return repeated_prompts
    else:
        raise ValueError(
            f"Invalid mode: {mode}, only support 'random', 'tile', 'interleave'"
        )


def add_cache_misses(prompts, hit_miss_ratio):
    """
    Add cache misses to the prompts and return a boolean mask aligned with prompts.
    """
    if hit_miss_ratio is None:
        return prompts, [False] * len(prompts)

    hit, miss = map(int, hit_miss_ratio.split(":", 1))
    period = hit + miss
    miss_mask = [False] * len(prompts)

    for i in range(len(prompts)):
        # every (hit+miss) window: first `hit` are hits, rest are misses
        if period and (i % period) >= hit:
            miss_mask[i] = True
            prompts[i] = f"{random.randint(-10_000_000, 10_000_000)} {prompts[i]}"

    return prompts, miss_mask


def relative_time(df, start_time):
    """
    Relative time to the start of the benchmark.
    """
    df["request_start"] = df["request_start"] - start_time
    df["request_end"] = df["request_end"] - start_time
    df["ttft_time"] = df["request_start"] + df["ttft"]


def visualize_results(warmup_df, benchmark_df):
    def plot_bars(df, title, filename):
        plt.figure(figsize=(12, 6))

        if "is_miss" in df.columns:
            is_miss = df["is_miss"]
        else:
            is_miss = pd.Series(False, index=df.index)

        hits = df[~is_miss]
        misses = df[is_miss]

        # Prefill: dark blue (hit), dark orange (miss)
        if not hits.empty:
            plt.barh(
                hits["prompt_id"],
                hits["ttft_time"] - hits["request_start"],
                left=hits["request_start"],
                color="darkblue",
                label="Loading",  # prefill hits
            )
        if not misses.empty:
            plt.barh(
                misses["prompt_id"],
                misses["ttft_time"] - misses["request_start"],
                left=misses["request_start"],
                color="darkorange",
                label="Compute",  # prefill misses
            )

        # Decode: light blue (hit), light orange (miss)
        if not hits.empty:
            plt.barh(
                hits["prompt_id"],
                hits["request_end"] - hits["ttft_time"],
                left=hits["ttft_time"],
                color="skyblue",
                label="Decoding after loading",
            )
        if not misses.empty:
            plt.barh(
                misses["prompt_id"],
                misses["request_end"] - misses["ttft_time"],
                left=misses["ttft_time"],
                color="pink",
                label="Decoding after compute",
            )

        plt.xlabel("Time (s)")
        plt.ylabel("Prompt ID")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    plot_bars(warmup_df, "Warmup Round", "warmup_round.png")
    plot_bars(benchmark_df, "Query Round", "query_round.png")


def trimmed_mean(series: pd.Series, trim_fraction: float) -> float:
    """
    Exclude the smallest trim_fraction and largest trim_fraction and take mean
    of the rest. If trim_fraction <= 0, returns normal mean.
    """
    s = series.dropna()
    if len(s) == 0:
        return float("nan")
    if trim_fraction <= 0:
        return float(s.mean())

    if not (0.0 <= trim_fraction < 0.5):
        raise ValueError("--trim-fraction must be in [0, 0.5).")

    s = s.sort_values()
    n = len(s)
    k = int(n * trim_fraction)
    if n - 2 * k <= 0:
        return float(s.mean())
    return float(s.iloc[k : n - k].mean())


async def main(args):
    random.seed(args.shuffle_seed)

    # Create the OpenAI client
    # No timeout: some benchmarks can take 4-5 minutes per request
    base_url = get_url_from_args(args)
    print("Using base URL:", base_url)

    api_key = os.getenv("OPENAI_API_KEY", "sk-dummy")

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=None,
    )
    model = args.model
    if model == "auto":
        print("Auto-selecting model...", end=" ")
        models = (await client.models.list(),)
        model = models[0].data[0].id
        print(f"selected model: {model}")

    full_len = args.document_length
    chunk_size = args.chunk_size
    if chunk_size < 1:
        raise ValueError(f"--chunk-size must be >= 1, got {chunk_size}")

    if args.warmup_document_length is not None:
        warmup_len = args.warmup_document_length
    else:
        rate = args.target_hit_rate
        if not (0.0 < rate <= 1.0):
            raise ValueError(
                f"--target-hit-rate must be in (0, 1], got {rate}"
            )
        warmup_len = int(full_len * rate)

    # Align down so LMCache PREFIX hit length is chunk-stable.
    warmup_len = (warmup_len // chunk_size) * chunk_size
    if warmup_len < chunk_size:
        warmup_len = chunk_size if full_len >= chunk_size else full_len
    if warmup_len > full_len:
        raise ValueError(
            f"warmup length ({warmup_len}) > --document-length ({full_len})"
        )

    target_hit_rate = warmup_len / full_len if full_len else 0.0
    write_resp(
        f"prefix hit-rate mode: warmup_len={warmup_len} "
        f"query_len={full_len} target_hit_rate={target_hit_rate:.4f} "
        f"chunk_size={chunk_size}\n"
    )

    # Same doc id + shared body prefix so query extends the cold-cached prefix.
    warmup_prompts = [
        str(i) + " " + " ".join(["hi"] * warmup_len)
        for i in range(args.num_documents)
    ]
    query_source_prompts = [
        str(i) + " " + " ".join(["hi"] * full_len)
        for i in range(args.num_documents)
    ]

    prompts = repeat_prompts(
        query_source_prompts, args.repeat_count, mode=args.repeat_mode
    )
    prompts, miss_mask = add_cache_misses(prompts, args.hit_miss_ratio)

    output_dir = args.output_dir or "."
    os.makedirs(output_dir, exist_ok=True)
    CSI = "\x1b["
    RESET = CSI + "0m"

    write_resp(
        "Effective fixed wait after warmup before query: "
        f"{args.sleep_time_after_warmup} seconds.\n"
    )

    write_resp("------warm up round (cold)------\n")
    warmup_start_time = time.time()
    warmup_request_stats = await test_long_document_qa(
        client=client,
        model=model,
        prompts=warmup_prompts,
        output_len=args.output_len,
        max_inflight_requests=args.max_inflight_requests,
    )
    warmup_end_time = time.time()

    # Print cold/warmup TTFT as soon as the round finishes (don't wait for query).
    warmup_df = pd.DataFrame([stats.__dict__ for stats in warmup_request_stats])
    relative_time(warmup_df, warmup_start_time)
    warmup_df["is_miss"] = True
    warmup_csv = os.path.join(output_dir, "warmup_round.csv")
    warmup_df.to_csv(warmup_csv, index=False)
    warmup_mean_ttft = trimmed_mean(
        warmup_df.query("successful == True")["ttft"], args.trim_fraction
    )
    warmup_success_count = warmup_df.query("successful == True").shape[0]
    print(f"{CSI}33;1m\n=== WARMUP (COLD) RESULTS ==={RESET}")
    print(f"Warmup round mean TTFT: {warmup_mean_ttft:.3f}s")
    print(f"Warmup round time: {warmup_end_time - warmup_start_time:.3f}s")
    print(f"Warmup round prompt count: {len(warmup_df)}")
    print(f"Warmup round successful prompt count: {warmup_success_count}")
    print(
        f"Warmup body length: {warmup_len}  "
        f"query body length: {full_len}  "
        f"target hit rate: {target_hit_rate:.4f}"
    )
    sys.stdout.flush()

    sleep_time_after_warmup = args.sleep_time_after_warmup
    if sleep_time_after_warmup > 0:
        write_resp(
            f"Waiting {sleep_time_after_warmup} seconds for warmup STOREs "
            "to drain before query...\n"
        )
        await asyncio.sleep(sleep_time_after_warmup)
    else:
        write_resp(
            "Warmup-to-query wait is 0 seconds; issuing the query immediately "
            "for HOL reproduction.\n"
        )

    write_resp("------query round------\n")

    benchmark_start_time = time.time()
    benchmark_request_stats = await test_long_document_qa(
        client=client,
        model=model,
        prompts=prompts,
        output_len=args.output_len,
        max_inflight_requests=args.max_inflight_requests,
    )
    benchmark_end_time = time.time()

    benchmark_df = pd.DataFrame([stats.__dict__ for stats in benchmark_request_stats])
    benchmark_df["is_miss"] = miss_mask
    relative_time(benchmark_df, benchmark_start_time)
    query_csv = os.path.join(output_dir, "query_round.csv")
    benchmark_df.to_csv(query_csv, index=False)

    query_mean_ttft = trimmed_mean(
        benchmark_df.query("successful == True")["ttft"], args.trim_fraction
    )
    query_success_count = benchmark_df.query("successful == True").shape[0]
    print(f"{CSI}36;1m\n=== QUERY (WARM) RESULTS ==={RESET}")
    print(f"{CSI}32mQuery round mean TTFT: {query_mean_ttft:.3f}s{RESET}")
    print(
        f"{CSI}33mQuery round time: "
        f"{benchmark_end_time - benchmark_start_time:.3f}s{RESET}"
    )
    print(f"{CSI}35mQuery round prompt count: {len(benchmark_df)}{RESET}")
    print(f"{CSI}34mQuery round successful prompt count: {query_success_count}{RESET}")
    sys.stdout.flush()

    if visualize:
        visualize_results(warmup_df, benchmark_df)

    if args.json_output:
        query_duration = benchmark_end_time - benchmark_start_time
        query_round_time_per_prompt = query_duration / len(benchmark_df)
        warmup_duration = warmup_end_time - warmup_start_time
        warmup_round_time_per_prompt = warmup_duration / len(warmup_df)
        # Standard
        import json

        summary = {
            "query_ttft_per_prompt": query_mean_ttft,
            "query_round_time_per_prompt": query_round_time_per_prompt,
            "warmup_round_time_per_prompt": warmup_round_time_per_prompt,
            "document_length": full_len,
            "warmup_document_length": warmup_len,
            "target_hit_rate": target_hit_rate,
            "chunk_size": chunk_size,
            "num_documents": args.num_documents,
            "warmup_request_ids": [
                stats.request_id for stats in warmup_request_stats
            ],
            "sleep_time_after_warmup": args.sleep_time_after_warmup,
        }
        print(json.dumps(summary))


def create_argument_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark the performance with or "
        "without automatic prefix caching."
    )

    parser.add_argument(
        "--document-length",
        type=int,
        # Roughly the number of tokens for a system paper,
        # excluding images
        default=20000,
        help="Full document body length for the query round (token-proxy words).",
    )

    parser.add_argument(
        "--warmup-document-length",
        type=int,
        default=None,
        help=(
            "Cold-round body length (must be <= --document-length). "
            "If omitted, derived from --target-hit-rate."
        ),
    )

    parser.add_argument(
        "--target-hit-rate",
        type=float,
        default=1.0,
        help=(
            "Used when --warmup-document-length is omitted: "
            "warmup_len = floor(document_length * rate), aligned down to "
            "--chunk-size. Example: 0.8 → expect ~80%% offload prefix hit."
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Align warmup length down to this size (LMCache chunk size).",
    )

    parser.add_argument(
        "--num-documents",
        type=int,
        default=8,
        help="Number of documents (same set for cold and query rounds).",
    )

    parser.add_argument(
        "--output-len",
        type=int,
        default=100,
        help="Maximum number of tokens to generate for each prompt.",
    )

    parser.add_argument(
        "--repeat-count",
        type=int,
        default=2,
        help="Number of times to repeat each prompt",
    )

    parser.add_argument(
        "--repeat-mode",
        type=str,
        default="random",
        help="The mode to repeat prompts. The supported "
        'modes are "random", "tile", and "interleave". '
        "See repeat_prompts() in the source code for details.",
    )

    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help='Random seed when the repeat mode is "random"',
    )

    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to query the vLLM server",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to query the vLLM server",
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL to query the LLM server",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="auto",
        help="Model name, can be set to 'auto' if the "
        "endpoint support openai api /models",
    )

    parser.add_argument(
        "--max-inflight-requests",
        type=int,
        default=2,
        help="Maximum number of concurrent inflight requests",
    )

    parser.add_argument(
        "--sleep-time-after-warmup",
        type=float,
        default=600.0,
        help=(
            "Fixed seconds to wait after warmup before the query (default: 600). "
            "Set to 0 for immediate-query HOL reproduction."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Filename to write all responses to; if omitted, writes to stdout.",
    )

    parser.add_argument(
        "--completions",
        action="store_true",
        help="Use completions API instead of chat completions API",
    )

    parser.add_argument(
        "--pd-disagg-ttft",
        action="store_true",
        help=(
            "PD disagg: measure TTFT from the decoder's first token. "
            "Skips proxy-injected prefill chunks tagged disagg_source=prefill."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for warmup_round.csv and query_round.csv.",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the results",
    )

    parser.add_argument(
        "--hit-miss-ratio",
        type=str,
        default=None,
        help=(
            "In query round, control how many of the prompts will miss the cache."
            "For example, 3:1 means every fourth repeated prompt will be randomized "
            "to force a cache miss. 2:2 means 2 hits and 2 misses"
        ),
    )

    parser.add_argument(
        "--eos-token-id",
        type=int,
        default=None,
        help=(
            "EOS token id. we bias against this token id so we always "
            "get the number of output tokens we specify"
        ),
    )

    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Print benchmark summary as a single JSON line to stdout.",
    )

    parser.add_argument(
        "--trim-fraction",
        type=float,
        default=0.0,
        help=(
            "Exclude the smallest and largest fraction of successful samples "
            "before averaging. Example: 0.1 drops bottom 10%% and top 10%%."
        ),
    )

    return parser


def validate_args(args):
    # Verify port and base_url are exclusive
    has_host_port = args.host is not None and args.port is not None
    has_base_url = args.base_url is not None
    if has_host_port and has_base_url:
        raise ValueError("Cannot use --host/--port and --base-url together.")
    if (
        not math.isfinite(args.sleep_time_after_warmup)
        or args.sleep_time_after_warmup < 0
    ):
        raise ValueError("--sleep-time-after-warmup must be a finite number >= 0.")


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    validate_args(args)
    completions_mode = args.completions
    pd_disagg_ttft = args.pd_disagg_ttft
    visualize = args.visualize
    if visualize:
        # Third Party
        import matplotlib.pyplot as plt
    if args.eos_token_id is not None:
        eos_token_id = args.eos_token_id
    OUTPUT_FILE = args.output
    asyncio.run(main(args))
