"""Smoke test: every module in src/ imports cleanly."""
import importlib

import pytest

MODULES = ["dynamics", "solvers", "data", "model", "train", "evaluate", "benchmark", "validate", "surrogate"]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(f"src.{name}")
