# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Active build entry for the lmcache-iluvatar plugin.

``setup_upstream.py`` is kept as the LMCache v0.5.3 reference snapshot. This
file intentionally reuses only the upstream ``lmcache.c_ops`` source list and
compile settings needed to build the package-scoped Iluvatar compatibility
module.
"""

from __future__ import annotations

from pathlib import Path
import os
import shlex
import shutil
import subprocess
import sys

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py
import site

ROOT_DIR = Path(__file__).parent

_BASE_VERSION = "0.1.0"


def _package_version() -> str:
    """Base version from pyproject.toml; optional PEP 440 local suffix from env."""
    local = os.environ.get("LOCAL_VERSION_IDENTIFIER", "").strip()
    if local:
        return f"{_BASE_VERSION}+{local}"
    return _BASE_VERSION


BUILDING_SDIST = "sdist" in sys.argv or os.environ.get("NO_CUDA_EXT", "0") == "1"
MOONCAKE_ENABLE_ENV = "LMCACHE_ILUVATAR_ENABLE_MOONCAKE"

C_OPS_SOURCES = [
    "csrc/pybind.cpp",
    "csrc/mem_kernels.cu",
    "csrc/mp_mem_kernels.cu",
    "csrc/blend_kernels.cu",
    "csrc/cal_cdf.cu",
    "csrc/ac_enc.cu",
    "csrc/ac_dec.cu",
    "csrc/pos_kernels.cu",
    "csrc/mem_alloc.cpp",
    "csrc/utils.cpp",
    "csrc/event_recorder.cpp",
    "csrc/completion_recorder.cpp",
]

NATIVE_STORAGE_OPS_SOURCES = [
    "csrc/storage_manager/bitmap.cpp",
    "csrc/storage_manager/fold.cpp",
    "csrc/storage_manager/periodic_event_notifier.cpp",
    "csrc/storage_manager/pybind.cpp",
    "csrc/storage_manager/ttl_lock.cpp",
    "csrc/storage_manager/utils.cpp",
]

REDIS_SOURCES = [
    "csrc/storage_backends/redis/pybind.cpp",
    "csrc/storage_backends/redis/connector.cpp",
]

FS_SOURCES = [
    "csrc/storage_backends/fs/pybind.cpp",
    "csrc/storage_backends/fs/connector.cpp",
]

MOONCAKE_SOURCES = [
    "csrc/storage_backends/mooncake/pybind.cpp",
    "csrc/storage_backends/mooncake/connector.cpp",
]


def _split_env_paths(value: str) -> list[str]:
    return [item for item in value.split(";") if item]


def _split_env_words(value: str) -> list[str]:
    return shlex.split(value) if value else []


def _append_unique(target: list[str], values: list[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def _run_mooncake_pkg_config() -> list[str]:
    package_name = os.environ.get("MOONCAKE_PKG_CONFIG_NAME", "mooncake-store")
    try:
        completed = subprocess.run(
            ["pkg-config", "--cflags", "--libs", package_name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        message = getattr(exc, "stderr", "") or str(exc)
        raise RuntimeError(
            f"pkg-config could not resolve {package_name!r}: {message.strip()}"
        ) from exc

    return shlex.split(completed.stdout)


def _parse_mooncake_build_tokens(tokens: list[str]) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {
        "include_dirs": [],
        "library_dirs": [],
        "runtime_library_dirs": [],
        "libraries": [],
        "extra_compile_args": [],
        "extra_link_args": [],
    }

    for token in tokens:
        if token.startswith("-I") and len(token) > 2:
            _append_unique(parsed["include_dirs"], [token[2:]])
        elif token.startswith("-L") and len(token) > 2:
            library_dir = token[2:]
            _append_unique(parsed["library_dirs"], [library_dir])
            _append_unique(parsed["runtime_library_dirs"], [library_dir])
        elif token.startswith("-l") and len(token) > 2:
            _append_unique(parsed["libraries"], [token[2:]])
        elif token.startswith("-Wl,"):
            parsed["extra_link_args"].append(token)
        elif token.startswith("-"):
            parsed["extra_compile_args"].append(token)
        else:
            parsed["extra_link_args"].append(token)

    return parsed


def _find_mooncake_sdk_config(include_dirs: list[str]) -> Path | None:
    for include_dir in include_dirs:
        base = Path(include_dir)
        candidates = [
            base / "mooncake-store" / "mooncake_sdk_config.h",
            base / "mooncake_sdk_config.h",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    return None


def _read_mooncake_cxx11_abi(include_dirs: list[str]) -> int:
    env_value = os.environ.get("MOONCAKE_ENABLE_CXX11_ABI")
    if env_value is not None:
        if env_value in {"0", "1"}:
            return int(env_value)
        raise RuntimeError("MOONCAKE_ENABLE_CXX11_ABI must be set to 0 or 1")

    config_header = _find_mooncake_sdk_config(include_dirs)
    if config_header is None:
        raise RuntimeError(
            "Mooncake native extension is enabled, but "
            "mooncake_sdk_config.h was not found in pkg-config or "
            "MOONCAKE_INCLUDE_DIR include paths. Install the Mooncake C++ SDK "
            "or set MOONCAKE_ENABLE_CXX11_ABI=0/1 explicitly."
        )

    for line in config_header.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if (
            len(parts) == 3
            and parts[0] == "#define"
            and parts[1] == "MOONCAKE_SDK_GLIBCXX_USE_CXX11_ABI"
            and parts[2] in {"0", "1"}
        ):
            return int(parts[2])

    raise RuntimeError(
        f"{config_header} does not define MOONCAKE_SDK_GLIBCXX_USE_CXX11_ABI as 0 or 1"
    )


def _mooncake_extension(cpp_extension) -> list:
    """Build the optional Mooncake CppExtension."""
    if os.environ.get(MOONCAKE_ENABLE_ENV) != "1":
        return []

    mc_include = os.environ.get("MOONCAKE_INCLUDE_DIR", "")
    mc_lib = os.environ.get("MOONCAKE_LIB_DIR", "")
    mc_extra_libs = os.environ.get("MOONCAKE_EXTRA_LIBS", "")
    use_pkg_config = os.environ.get("MOONCAKE_USE_PKG_CONFIG", "1") != "0"
    mc_include_dirs = [
        "csrc/storage_backends",
        "csrc/storage_backends/mooncake",
    ]
    mc_library_dirs: list[str] = []
    mc_runtime_library_dirs: list[str] = []
    mc_libraries: list[str] = []
    mc_cxx_flags = ["-O3", "-std=c++20", "-DYLT_ENABLE_IBV"]
    mc_link_args: list[str] = []

    pkg_config_error: RuntimeError | None = None
    if use_pkg_config:
        try:
            pkg_config = _parse_mooncake_build_tokens(_run_mooncake_pkg_config())
        except RuntimeError as exc:
            pkg_config_error = exc
        else:
            _append_unique(mc_include_dirs, pkg_config["include_dirs"])
            _append_unique(mc_library_dirs, pkg_config["library_dirs"])
            _append_unique(mc_runtime_library_dirs, pkg_config["runtime_library_dirs"])
            _append_unique(mc_libraries, pkg_config["libraries"])
            mc_cxx_flags.extend(pkg_config["extra_compile_args"])
            mc_link_args.extend(pkg_config["extra_link_args"])

    if mc_include:
        _append_unique(mc_include_dirs, _split_env_paths(mc_include))
    if mc_lib:
        _append_unique(mc_library_dirs, _split_env_paths(mc_lib))
        _append_unique(mc_runtime_library_dirs, _split_env_paths(mc_lib))
    if mc_extra_libs:
        _append_unique(mc_libraries, [lib for lib in mc_extra_libs.split(";") if lib])
    mc_cxx_flags.extend(
        _split_env_words(os.environ.get("MOONCAKE_EXTRA_CXX_FLAGS", ""))
    )
    mc_link_args.extend(
        _split_env_words(os.environ.get("MOONCAKE_EXTRA_LINK_ARGS", ""))
    )

    if not mc_libraries:
        if mc_include or mc_lib:
            mc_libraries.append("mooncake_store")
        else:
            detail = f" {pkg_config_error}" if pkg_config_error is not None else ""
            raise RuntimeError(
                "Mooncake native extension is enabled, but pkg-config did not "
                "provide libraries and MOONCAKE_INCLUDE_DIR/MOONCAKE_LIB_DIR "
                f"fallbacks are not set.{detail}"
            )

    mooncake_cxx11_abi = _read_mooncake_cxx11_abi(mc_include_dirs)
    extension = cpp_extension.CppExtension(
        "lmcache_iluvatar.lmcache_mooncake",
        sources=MOONCAKE_SOURCES,
        include_dirs=mc_include_dirs,
        library_dirs=mc_library_dirs,
        libraries=mc_libraries,
        runtime_library_dirs=mc_runtime_library_dirs,
        extra_compile_args={
            "cxx": mc_cxx_flags,
        },
        extra_link_args=mc_link_args,
    )
    extension._iluvatar_mooncake_cxx11_abi = mooncake_cxx11_abi
    return [extension]


def _native_extensions() -> tuple[list, dict]:
    if BUILDING_SDIST:
        print("Not building lmcache-iluvatar native extensions for sdist/NO_CUDA_EXT")
        return [], {}

    import torch  # type: ignore[import-not-found]  # noqa: F401
    from torch.utils import cpp_extension  # type: ignore[import-not-found]

    cxx_flags = ["-std=c++17"]
    ext_modules = [
        cpp_extension.CUDAExtension(
            "lmcache_iluvatar.c_ops",
            sources=C_OPS_SOURCES,
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": [],
            },
        ),
        cpp_extension.CppExtension(
            "lmcache_iluvatar.native_storage_ops",
            sources=NATIVE_STORAGE_OPS_SOURCES,
            include_dirs=["csrc/storage_manager"],
            extra_compile_args={
                "cxx": cxx_flags + ["-O3"],
            },
        ),
        cpp_extension.CppExtension(
            "lmcache_iluvatar.lmcache_redis",
            sources=REDIS_SOURCES,
            include_dirs=["csrc/storage_backends", "csrc/storage_backends/redis"],
            extra_compile_args={
                "cxx": cxx_flags + ["-O3"],
            },
        ),
        cpp_extension.CppExtension(
            "lmcache_iluvatar.lmcache_fs",
            sources=FS_SOURCES,
            include_dirs=["csrc/storage_backends", "csrc/storage_backends/fs"],
            extra_compile_args={
                "cxx": cxx_flags + ["-O3"],
            },
        ),
    ]
    ext_modules.extend(_mooncake_extension(cpp_extension))

    class BuildExtension(cpp_extension.BuildExtension):
        """Use the Mooncake SDK ABI only for the Mooncake extension."""

        def _add_gnu_cpp_abi_flag(self, extension):
            mooncake_abi = getattr(extension, "_iluvatar_mooncake_cxx11_abi", None)
            if mooncake_abi is None:
                super()._add_gnu_cpp_abi_flag(extension)
                return
            self._add_compile_flag(
                extension,
                f"-D_GLIBCXX_USE_CXX11_ABI={int(mooncake_abi)}",
            )

    return ext_modules, {"build_ext": BuildExtension}


PTH_FILENAME = "lmcache_iluvatar.pth"
HOOK_MODULE = "_lmcache_iluvatar_hook"


def _install_lmcache_iluvatar_pth() -> None:
    """Copy ``lmcache_iluvatar.pth`` and hook module to site-packages."""
    files = [
        ROOT_DIR / PTH_FILENAME,
        ROOT_DIR / f"{HOOK_MODULE}.py",
    ]
    for src in files:
        if not src.is_file():
            print(f"Warning: {src} not found; skipping.")
            continue
        for sp_dir in site.getsitepackages():
            dst = Path(sp_dir) / src.name
            try:
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                print(f"Installed {src.name} → {dst}")
                break
            except OSError:
                continue
        else:
            print(f"Warning: could not install {src.name} to any site-packages.")


def _stage_upstream_pin_into_build_lib(build_lib: Path) -> None:
    """Stage the repo-root upstream pin directly into the wheel build tree."""
    src = ROOT_DIR / "upstream_pin.json"
    if not src.is_file():
        raise FileNotFoundError(f"{src} is required to build a compatible wheel")
    dst = build_lib / "lmcache_iluvatar" / "upstream_pin.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"Staged upstream pin → {dst}")


class build_py(_build_py):
    """build_py command that also deploys the .pth import hook.

    For wheel builds the .pth is staged into ``build_lib`` so it lands in the
    wheel RECORD and is installed (and uninstalled) by pip on any machine.
    Writing it straight to site-packages here would only be a build-time side
    effect on the build host, invisible to anyone installing the archive.

    PEP-660 editable installs have no wheel archive to carry the file, so they
    keep the direct site-packages write.
    """

    def run(self) -> None:
        _build_py.run(self)
        _stage_upstream_pin_into_build_lib(Path(self.build_lib))
        if getattr(self, "editable_mode", False):
            _install_lmcache_iluvatar_pth()
        else:
            self._stage_pth_into_build_lib()

    def _stage_pth_into_build_lib(self) -> None:
        src = ROOT_DIR / PTH_FILENAME
        if not src.is_file():
            # A wheel without the .pth silently loses plugin auto-activation for
            # every non-vLLM entry point, so fail the build instead.
            raise FileNotFoundError(f"{src} is required to build a usable wheel")
        build_lib = Path(self.build_lib)
        build_lib.mkdir(parents=True, exist_ok=True)
        dst = build_lib / PTH_FILENAME
        shutil.copy2(src, dst)
        print(f"Staged {PTH_FILENAME} → {dst}")


ext_modules, _cmdclass = _native_extensions()
_cmdclass["build_py"] = build_py

setup(
    name="lmcache-iluvatar",
    version=_package_version(),
    description="Non-invasive LMCache vLLM plugin for Iluvatar devices.",
    long_description=(ROOT_DIR / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("csrc", "tests", "tests.*")),
    py_modules=[HOOK_MODULE],
    include_package_data=True,
    package_data={"lmcache_iluvatar": ["upstream_pin.json"]},
    ext_modules=ext_modules,
    cmdclass=_cmdclass,
)
