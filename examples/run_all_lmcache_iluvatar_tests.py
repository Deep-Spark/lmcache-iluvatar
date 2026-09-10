#!/usr/bin/env python3
# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Run GPU integration cases listed in examples/ci/manifest.json.

Each case is a shell script under its example directory (convention: <example>/ci/run.sh).
Add or remove CI coverage by editing the manifest — no Python changes required.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


MANIFEST_PATH = Path(__file__).resolve().parent / "ci" / "manifest.json"
VALID_LEVELS = ("L0", "L1", "L2")


@dataclass
class CaseSpec:
    name: str
    script: str
    gpus: int
    requires_model: bool
    level: str
    enabled: bool = True
    description: str = ""


@dataclass
class CaseResult:
    name: str
    level: str
    status: str
    duration_s: float
    message: str = ""
    log_dir: str = ""

    @property
    def pass_value(self) -> bool | str:
        if self.status == "passed":
            return True
        if self.status == "skipped":
            return "skipped"
        return False


class RunnerError(RuntimeError):
    pass


def load_manifest(path: Path) -> list[CaseSpec]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases: list[CaseSpec] = []
    for item in data.get("cases", []):
        name = item["name"]
        level = str(item.get("level", "")).strip().upper()
        if level not in VALID_LEVELS:
            raise RunnerError(
                f"Case {name} has invalid level {item.get('level')!r}; "
                f"expected one of {', '.join(VALID_LEVELS)}"
            )
        cases.append(
            CaseSpec(
                name=name,
                script=item["script"],
                gpus=int(item.get("gpus", 1)),
                requires_model=bool(item.get("requires_model", False)),
                level=level,
                enabled=bool(item.get("enabled", True)),
                description=item.get("description", ""),
            )
        )
    return cases


def resolve_gpu_ids(gpu_arg: str | None) -> list[str]:
    if gpu_arg:
        return [item.strip() for item in gpu_arg.split(",") if item.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [item.strip() for item in visible.split(",") if item.strip()]
    for counter_cmd in (("ixsmi", "-L"), ("nvidia-smi", "-L")):
        binary = shutil.which(counter_cmd[0])
        if not binary:
            continue
        proc = subprocess.run(
            [binary, counter_cmd[1]],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode == 0:
            return [str(i) for i, line in enumerate(proc.stdout.splitlines()) if line.strip()]
    try:
        import torch

        return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001
        return []


def served_model_name(base_model_path: str, override: str | None) -> str:
    if override:
        return override
    return Path(base_model_path.rstrip("/")).name


def parse_cases(cases_arg: str, manifest: list[CaseSpec]) -> list[CaseSpec]:
    by_name = {case.name: case for case in manifest}
    if cases_arg == "all":
        enabled = [case for case in manifest if case.enabled]
        if not enabled:
            raise RunnerError("No enabled cases in manifest.")
        return enabled
    requested = [item.strip() for item in cases_arg.split(",") if item.strip()]
    invalid = [name for name in requested if name not in by_name]
    if invalid:
        valid = ", ".join(by_name)
        raise RunnerError(f"Unknown cases: {', '.join(invalid)}. Valid: {valid}")
    return [by_name[name] for name in requested]


def precheck(case: CaseSpec, gpu_ids: list[str], base_model_path: str) -> None:
    if case.requires_model and not base_model_path:
        raise RunnerError(f"--base-model-path is required for case {case.name}")
    if len(gpu_ids) < case.gpus:
        raise RunnerError(
            f"Case {case.name} requires {case.gpus} GPU(s), "
            f"but only {len(gpu_ids)} configured."
        )
    script_path = Path(__file__).resolve().parent / case.script
    if not script_path.is_file():
        raise RunnerError(f"Case script missing: {script_path}")


def run_case(
    case: CaseSpec,
    *,
    examples_dir: Path,
    repo_root: Path,
    log_root: Path,
    gpu_ids: list[str],
    base_model_path: str,
    served_name: str | None,
    startup_timeout: int,
    request_timeout: int,
    cleanup_timeout: int,
) -> CaseResult:
    start = time.time()
    case_log_dir = (log_root / case.name).resolve()
    case_log_dir.mkdir(parents=True, exist_ok=True)
    script_path = examples_dir / case.script

    env = os.environ.copy()
    env.update(
        {
            "LMCACHE_CI_REPO_ROOT": str(repo_root),
            "LMCACHE_CI_LOG_DIR": str(case_log_dir),
            "LMCACHE_CI_GPUS": ",".join(gpu_ids[: case.gpus]),
            "LMCACHE_CI_STARTUP_TIMEOUT": str(startup_timeout),
            "LMCACHE_CI_REQUEST_TIMEOUT": str(request_timeout),
            "LMCACHE_CI_CLEANUP_TIMEOUT": str(cleanup_timeout),
        }
    )
    if base_model_path:
        env["LMCACHE_CI_MODEL_PATH"] = base_model_path
        env["MODEL_PATH"] = base_model_path
    if served_name:
        env["LMCACHE_CI_SERVED_MODEL_NAME"] = served_name

    try:
        precheck(case, gpu_ids, base_model_path)
        proc = subprocess.run(
            ["bash", str(script_path)],
            cwd=examples_dir,
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            return CaseResult(
                name=case.name,
                level=case.level,
                status="failed",
                duration_s=time.time() - start,
                message=f"exit code {proc.returncode}",
                log_dir=str(case_log_dir),
            )
        return CaseResult(
            name=case.name,
            level=case.level,
            status="passed",
            duration_s=time.time() - start,
            log_dir=str(case_log_dir),
        )
    except RunnerError as exc:
        return CaseResult(
            name=case.name,
            level=case.level,
            status="failed",
            duration_s=time.time() - start,
            message=str(exc),
            log_dir=str(case_log_dir),
        )


def print_summary(results: list[CaseResult], log_root: Path, result_json: Path) -> None:
    print("\nlmcache-iluvatar integration test summary")
    print("=" * 80)
    for result in results:
        line = (
            f"{result.name:40} {result.level:3} {result.status:8} "
            f"{result.duration_s:7.1f}s"
        )
        if result.message:
            line += f"  {result.message}"
        print(line)
        if result.log_dir:
            print(f"  log: {result.log_dir}")
    print("=" * 80)
    print(f"Logs saved under: {log_root}")
    print(f"Result JSON: {result_json}")


def write_result_json(results: list[CaseResult], path: Path) -> None:
    payload = [
        {"name": result.name, "level": result.level, "pass": result.pass_value}
        for result in results
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    manifest = load_manifest(MANIFEST_PATH)
    all_cases = [c.name for c in manifest]
    enabled_cases = [c.name for c in manifest if c.enabled]
    parser = argparse.ArgumentParser(
        description="Run lmcache-iluvatar GPU integration cases from examples/ci/manifest.json."
    )
    parser.add_argument(
        "--cases",
        default="all",
        help=(
            "Comma-separated case names or 'all' (enabled only). "
            "Registered: "
            + ", ".join(all_cases)
            + ". Enabled: "
            + (", ".join(enabled_cases) if enabled_cases else "(none)")
        ),
    )
    parser.add_argument("--base-model-path", default="", help="LLM path for cases that need a model.")
    parser.add_argument("--served-model-name", default="", help="Override vLLM served model name.")
    parser.add_argument("--gpus", default="", help="Comma-separated GPU ids, e.g. 0,1.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed case.")
    parser.add_argument("--log-dir", default="", help="Root directory for case logs.")
    parser.add_argument(
        "--result-json",
        default="",
        help="Path to write per-case results JSON (name/level/pass). "
        "pass is true/false/\"skipped\". Default: <log-dir>/results.json.",
    )
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument(
        "--cleanup-timeout",
        type=int,
        default=120,
        help="Seconds to wait for ports to close after each case cleanup.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
    except RunnerError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    examples_dir = Path(__file__).resolve().parent
    repo_root = examples_dir.parent
    manifest = load_manifest(MANIFEST_PATH)

    if args.log_dir:
        log_root = Path(args.log_dir).expanduser().resolve()
    else:
        log_root = (examples_dir / "logs" / time.strftime("%Y%m%d-%H%M%S")).resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    if args.result_json:
        result_json = Path(args.result_json).expanduser().resolve()
    else:
        result_json = (log_root / "results.json").resolve()

    gpu_ids = resolve_gpu_ids(args.gpus or None)
    served = served_model_name(args.base_model_path, args.served_model_name or None) if args.base_model_path else None

    try:
        selected = parse_cases(args.cases, manifest)
    except RunnerError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    results: list[CaseResult] = []
    for index, case in enumerate(selected):
        if len(gpu_ids) < case.gpus:
            if args.cases == "all":
                print(
                    f"[skipped] {case.name} requires {case.gpus} GPU(s), "
                    f"only {len(gpu_ids)} configured via --gpus / CUDA_VISIBLE_DEVICES"
                )
                results.append(
                    CaseResult(
                        name=case.name,
                        level=case.level,
                        status="skipped",
                        duration_s=0.0,
                        message=f"needs {case.gpus} GPU(s), have {len(gpu_ids)}",
                    )
                )
                continue

        result = run_case(
            case,
            examples_dir=examples_dir,
            repo_root=repo_root,
            log_root=log_root,
            gpu_ids=gpu_ids,
            base_model_path=args.base_model_path,
            served_name=served,
            startup_timeout=args.startup_timeout,
            request_timeout=args.request_timeout,
            cleanup_timeout=args.cleanup_timeout,
        )
        results.append(result)
        print(
            f"[{result.status}] {result.name} ({result.duration_s:.1f}s)"
            + (f" - {result.message}" if result.message else "")
        )
        if result.status == "failed" and not args.keep_going:
            for leftover in selected[index + 1 :]:
                results.append(
                    CaseResult(
                        name=leftover.name,
                        level=leftover.level,
                        status="skipped",
                        duration_s=0.0,
                        message="not run after earlier failure",
                    )
                )
            break
        if index + 1 < len(selected):
            time.sleep(3)

    write_result_json(results, result_json)
    print_summary(results, log_root, result_json)
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
