# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
import zipfile

from packaging.requirements import Requirement
import pytest


ROOT = Path(__file__).resolve().parents[1]
LMCACHE_PINNED_VERSION = "0.5.3"
PTH_HOOK_FILES = ("lmcache_iluvatar.pth", "_lmcache_iluvatar_hook.py")


def _load_upstream_pin_module() -> ModuleType:
    path = ROOT / "lmcache_iluvatar" / "_upstream_pin.py"
    spec = importlib.util.spec_from_file_location(
        "lmcache_iluvatar__upstream_pin", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_requirements_txt(path: Path) -> list[str]:
    specs: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        specs.append(line)
    return specs


def _read_pyproject_dependencies(path: Path) -> list[str]:
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # pragma: no cover - py3.9
        import tomli as tomllib  # type: ignore[no-redef]

    data = tomllib.loads(path.read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    assert isinstance(deps, list)
    return [str(item) for item in deps]


def _canonical_requirement(spec: str) -> str:
    return str(Requirement(spec))


def test_requirements_txt_matches_pyproject_dependencies():
    txt_specs = _read_requirements_txt(ROOT / "requirements.txt")
    pyproject_specs = _read_pyproject_dependencies(ROOT / "pyproject.toml")

    txt_canonical = {_canonical_requirement(spec) for spec in txt_specs}
    pyproject_canonical = {_canonical_requirement(spec) for spec in pyproject_specs}
    assert txt_canonical == pyproject_canonical


def test_lmcache_pin_matches_upstream_pin_json():
    pin = json.loads((ROOT / "upstream_pin.json").read_text(encoding="utf-8"))[
        "lmcache"
    ]
    assert pin == {"pypi_version": LMCACHE_PINNED_VERSION}

    txt_specs = _read_requirements_txt(ROOT / "requirements.txt")
    pyproject_specs = _read_pyproject_dependencies(ROOT / "pyproject.toml")
    assert all(Requirement(spec).name != "lmcache" for spec in txt_specs)
    assert all(Requirement(spec).name != "lmcache" for spec in pyproject_specs)

    gitmodules = (ROOT / ".gitmodules").read_text(encoding="utf-8")
    assert 'path = third_party/LMCache' in gitmodules
    assert (ROOT / "third_party" / "LMCache" / "pyproject.toml").is_file()


def test_documented_lmcache_pin_matches_upstream_pin_json():
    pin = json.loads((ROOT / "upstream_pin.json").read_text(encoding="utf-8"))[
        "lmcache"
    ]["pypi_version"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert f"lmcache=={pin}" in readme
    assert "third_party/LMCache" in readme
    assert "--recurse-submodules" in readme


def test_install_dependencies_builds_lmcache_from_submodule():
    script = (ROOT / "scripts" / "install_dependencies.sh").read_text(encoding="utf-8")
    assert 'LMCACHE_ROOT="${ROOT}/third_party/LMCache"' in script
    assert "NO_NATIVE_EXT=1" in script
    assert "pip install --no-deps" in script


def test_hybrid_group_edit_docs_describe_required_targets():
    development = (ROOT / "docs" / "development.md").read_text(encoding="utf-8")
    hybrid_table_row = next(
        line for line in development.splitlines() if "| Hybrid KV group edits |" in line
    )

    assert "required targets" in hybrid_table_row
    assert "required=False" not in hybrid_table_row
    assert "skip" not in hybrid_table_row


def test_upstream_setup_snapshot_is_documented():
    assert (ROOT / "setup_upstream.py").is_file()

    upstream_doc = (ROOT / "csrc" / "UPSTREAM.md").read_text()
    pin = json.loads((ROOT / "upstream_pin.json").read_text(encoding="utf-8"))[
        "lmcache"
    ]

    assert "upstream_pin.json" in upstream_doc
    assert "LMCache/LMCache" in upstream_doc
    assert pin["pypi_version"] in upstream_doc
    assert "setup_upstream.py" in upstream_doc
    assert "reference-only" in upstream_doc
    assert "setup.py" in upstream_doc


def test_build_py_stages_pth_into_the_wheel():
    """Guard against moving the .pth write back into a build-time side effect.

    Writing to site-packages during build only works on the build host; the
    wheel then ships without the auto-activation hook.
    """
    build_py_source = (
        (ROOT / "setup.py")
        .read_text(encoding="utf-8")
        .split("class build_py(_build_py):", 1)[1]
    )

    assert "self._stage_pth_into_build_lib()" in build_py_source
    assert 'getattr(self, "editable_mode", False)' in build_py_source


def test_build_py_stages_root_upstream_pin_into_wheel():
    """A stale or read-only source-package copy must not enter the wheel."""

    setup_source = (ROOT / "setup.py").read_text(encoding="utf-8")
    build_py_source = setup_source.split("class build_py(_build_py):", 1)[1]
    run_body = build_py_source.split("def run(self) -> None:", 1)[1].split(
        "def _stage_pth_into_build_lib", 1
    )[0]

    assert "_stage_upstream_pin_into_build_lib(Path(self.build_lib))" in run_body
    assert run_body.index("_build_py.run(self)") < run_body.index(
        "_stage_upstream_pin_into_build_lib"
    )
    assert 'build_lib / "lmcache_iluvatar" / "upstream_pin.json"' in setup_source
    assert 'ROOT_DIR / "lmcache_iluvatar" / "upstream_pin.json"' not in setup_source


def test_wheel_ships_pth_import_hook():
    wheels = sorted((ROOT / "dist").glob("lmcache_iluvatar-*.whl"))
    if not wheels:
        pytest.skip("no built wheel under dist/; run scripts/build_wheel.sh first")

    with zipfile.ZipFile(wheels[-1]) as archive:
        names = set(archive.namelist())
        record_name = next(name for name in names if name.endswith(".dist-info/RECORD"))
        recorded = {
            line.split(",", 1)[0]
            for line in archive.read(record_name).decode("utf-8").splitlines()
            if line
        }

    for filename in PTH_HOOK_FILES:
        assert filename in names, f"{filename} missing from {wheels[-1].name}"
        assert filename in recorded, f"{filename} missing from wheel RECORD"


def test_upstream_pin_json_is_single_source_of_truth():
    pin_path = ROOT / "upstream_pin.json"
    assert pin_path.is_file()

    raw = json.loads(pin_path.read_text(encoding="utf-8"))
    upstream_pin = _load_upstream_pin_module()
    assert raw["lmcache"]["pypi_version"] == upstream_pin.pinned_lmcache_pypi_version()
    assert upstream_pin.load_upstream_pin() == raw
