# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Console-script entry points for lmcache-iluvatar.

These override the upstream ``lmcache``, ``lmcache_server``, and
``lmcache_controller`` console_scripts registered in the upstream
``pyproject.toml``. Each entry imports the plugin (triggering native
redirects) before delegating to the corresponding upstream CLI.
"""

from __future__ import annotations


def main() -> None:
    """Override ``lmcache`` CLI: patch then delegate to upstream."""
    import lmcache_iluvatar  # noqa: F401 — triggers native redirects

    from lmcache.cli.main import main as _upstream_main

    _upstream_main()


def server_main() -> None:
    """Override ``lmcache_server`` CLI: patch then delegate to upstream."""
    import lmcache_iluvatar  # noqa: F401 — triggers native redirects

    from lmcache.v1.server.__main__ import main as _upstream_main

    _upstream_main()


def controller_main() -> None:
    """Override ``lmcache_controller`` CLI: patch then delegate to upstream."""
    import lmcache_iluvatar  # noqa: F401 — triggers native redirects

    from lmcache.v1.api_server.__main__ import main as _upstream_main

    _upstream_main()
