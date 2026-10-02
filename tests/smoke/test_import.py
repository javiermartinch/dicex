# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Smoke test for the installed package version."""

from importlib.metadata import version

import dicex


def test_import() -> None:
    """Verify that ``dicex.__version__`` matches the installed distribution metadata."""
    assert isinstance(dicex.__version__, str)
    assert dicex.__version__ == version("dicex")
