"""Scaffold smoke test: package imports cleanly and sub-packages are reachable."""

from __future__ import annotations

import importlib

import local_splitter


def test_version_present() -> None:
    assert isinstance(local_splitter.__version__, str)
    assert local_splitter.__version__.count(".") >= 2


def test_subpackages_import() -> None:
    for name in (
        "local_splitter.config",
        "local_splitter.transport",
        "local_splitter.models",
        "local_splitter.pipeline",
        "local_splitter.evals",
        "local_splitter.cli",
    ):
        importlib.import_module(name)
