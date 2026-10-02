# Copyright (c) 2026 Javier Martín-Chávez
# SPDX-License-Identifier: MIT
"""Smoke test that runs the Python examples of the top-level README."""

from pathlib import Path
import re

import pytest

README = Path(__file__).resolve().parents[2] / "README.md"


def test_readme_examples_run() -> None:
    """Execute every ``python`` code block of README.md so that the documented usage stays valid."""
    if not README.exists():
        pytest.skip("README.md is not available (tests run outside the source tree).")
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), flags=re.DOTALL)
    assert blocks, "README.md has no python example"
    for block in blocks:
        exec(compile(block, str(README), "exec"), {"__name__": "__readme__"})  # noqa: S102
