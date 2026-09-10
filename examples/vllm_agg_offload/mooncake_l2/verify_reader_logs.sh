#!/usr/bin/env bash
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Validate that the reader's first request came from Mooncake L2, not local L1.

set -euo pipefail

: "${READER_MP_LOG:?READER_MP_LOG must point to the reader LMCache MP log}"
: "${READER_VLLM_LOG:?READER_VLLM_LOG must point to the reader vLLM log}"

for path in "${READER_MP_LOG}" "${READER_VLLM_LOG}"; do
  if [[ ! -f "${path}" ]]; then
    echo "log file does not exist: ${path}" >&2
    exit 2
  fi
done

l2_regex='Prefetch request completed \(L1\+L2\): [1-9][0-9]*/[1-9][0-9]* retained keys \(0 L1, [1-9][0-9]* L2\)'
if ! grep -Eq "${l2_regex}" "${READER_MP_LOG}"; then
  echo "FAIL: reader MP log has no full positive L2 prefetch with empty L1" >&2
  echo "Expected: Prefetch ... N/N retained keys (0 L1, N L2)" >&2
  exit 1
fi

if ! grep -Eq 'Retrieved [1-9][0-9]* tokens' "${READER_MP_LOG}"; then
  echo "FAIL: reader MP log has no positive Retrieved token evidence" >&2
  exit 1
fi

external_hit_regex='Prefix cache hit rate: 0\.0%, External prefix cache hit rate: ([1-9][0-9]*(\.[0-9]+)?|0\.[0-9]*[1-9][0-9]*)%'
if ! grep -Eq "${external_hit_regex}" "${READER_VLLM_LOG}"; then
  echo "FAIL: reader vLLM log does not show zero GPU-prefix hit and positive external hit" >&2
  exit 1
fi

echo "PASS: reader used Mooncake L2"
grep -E "${l2_regex}" "${READER_MP_LOG}" | tail -n 1
grep -E 'Retrieved [1-9][0-9]* tokens' "${READER_MP_LOG}" | tail -n "${TENSOR_PARALLEL:-2}"
grep -E "${external_hit_regex}" "${READER_VLLM_LOG}" | tail -n 1
