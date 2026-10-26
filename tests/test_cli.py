"""CLI wiring."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from odtp.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]


def test_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "odtp", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "analyze" in result.stdout
    assert "build" in result.stdout
    assert "train" in result.stdout


def test_analyze_parser_requires_gene() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["analyze"])


def test_analyze_parser_accepts_gene() -> None:
    args = build_parser().parse_args(["analyze", "EGFR"])
    assert args.gene == "EGFR"
    assert args.command == "analyze"
