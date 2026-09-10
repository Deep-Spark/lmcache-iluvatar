#!/usr/bin/env python3
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Send one deterministic long request and validate response semantics/UTF-8."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


FILLER = (
    "This archival paragraph describes stable cache validation procedures, "
    "repeatable experiments, storage tiers, deterministic decoding, and careful "
    "measurement. It is background material and does not change the arithmetic "
    "instruction at the end of the request. "
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", default="7319")
    parser.add_argument("--prompt-repeat", type=int, default=160)
    parser.add_argument("--min-prompt-tokens", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompt = (
        FILLER * args.prompt_repeat
        + "\n\nUse only this final task: ALPHA is 7000 and BETA is 319. "
        "Compute ALPHA + BETA. Return only the decimal integer, with no "
        "explanation, punctuation, markdown, or surrounding words."
    )
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": "Follow the final arithmetic task and answer with only its integer.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"{args.label}: HTTP {exc.code}: {detail[:1000]}") from exc
    elapsed = time.monotonic() - started

    # Strict decoding catches malformed output bytes. U+FFFD catches a server
    # that already replaced invalid bytes before serializing the JSON response.
    decoded = raw.decode("utf-8", errors="strict")
    if "\ufffd" in decoded:
        raise SystemExit(f"{args.label}: response contains Unicode replacement characters")
    data = json.loads(decoded)
    choice = data["choices"][0]
    message = choice["message"]
    text = message.get("content") or ""
    reasoning_content = message.get("reasoning_content")
    normalized = text.strip()
    if normalized != args.expected:
        raise SystemExit(
            f"{args.label}: semantic mismatch: expected {args.expected!r}, got {text!r}"
        )
    if any(ord(char) < 32 and char not in "\t\r\n" for char in text):
        raise SystemExit(f"{args.label}: response contains unexpected control characters")
    usage = data.get("usage", {})
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    if prompt_tokens < args.min_prompt_tokens:
        raise SystemExit(
            f"{args.label}: prompt only has {prompt_tokens} tokens; "
            f"need at least {args.min_prompt_tokens} to exercise multiple L2 chunks"
        )

    result = {
        "label": args.label,
        "text": text,
        "reasoning_content": reasoning_content,
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": usage.get("completion_tokens"),
        "elapsed_seconds": round(elapsed, 3),
        "utf8_valid": True,
        "contains_replacement_character": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
