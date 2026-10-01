"""Contributor examples run in fresh processes to isolate registry mutations."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_example(*arguments):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )


def test_minimal_example():
    assert "Final batch loss:" in run_example("-m", "examples.minimal").stdout


@pytest.mark.parametrize(
    "plugin,model,credit",
    [
        ("examples.plugin", "toy_rnn", "zero_credit"),
        ("examples.cosmos_placeholder", "cosmos_example", "bptt"),
    ],
)
def test_cli_plugin_example(plugin, model, credit):
    result = run_example(
        "-m",
        "archcredit.cli",
        "run",
        "--plugin",
        plugin,
        "--model",
        model,
        "--credit",
        credit,
        "--task",
        "delayed-xor",
        "--steps",
        "1",
        "--batch-size",
        "2",
        "--horizon",
        "4",
    )
    assert result.stdout
