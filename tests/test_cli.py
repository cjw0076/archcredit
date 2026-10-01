import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from torch import nn

from archcredit.cli import main


@pytest.mark.parametrize(
    "model,credit", [("rnn", "bptt"), ("rnn", "eprop"), ("rnn", "hebbian"), ("transformer", "bptt")]
)
@pytest.mark.parametrize("task", ["delayed-xor", "copy", "adding"])
def test_supported_runs(model, credit, task, capsys):
    main(
        [
            "run",
            "--model",
            model,
            "--credit",
            credit,
            "--task",
            task,
            "--steps",
            "2",
            "--batch-size",
            "4",
            "--hidden-size",
            "4",
            "--horizon",
            "2",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["train_loss"] >= 0
    assert result["eval_loss"] >= 0
    assert result["parameters"] > 0
    assert result["sequence_length"] == (11 if task == "copy" else 5)
    if credit == "bptt":
        assert result["rho"] == pytest.approx(1)


def test_module_cli_output(tmp_path):
    output = tmp_path / "result.json"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "archcredit.cli",
            "run",
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--horizon",
            "2",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(process.stdout) == json.loads(output.read_text())


def test_unsupported_combination_has_helpful_error(capsys):
    with pytest.raises(SystemExit) as error:
        main(["run", "--model", "transformer", "--credit", "eprop", "--steps", "1"])
    assert error.value.code == 2
    assert "RNN" in capsys.readouterr().err


def test_seeded_run_reproducibility(capsys):
    arguments = ["run", "--steps", "2", "--batch-size", "4", "--horizon", "2"]
    main(arguments)
    first = json.loads(capsys.readouterr().out)
    main(arguments)
    second = json.loads(capsys.readouterr().out)
    first.pop("elapsed_seconds")
    second.pop("elapsed_seconds")
    assert first == second


def test_plugin_registration(capsys):
    main(
        [
            "run",
            "--plugin",
            "examples.plugin",
            "--model",
            "toy_rnn",
            "--credit",
            "zero_credit",
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--horizon",
            "2",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["model"] == "toy_rnn"
    assert result["rho"] is None


def test_installed_console_loads_checkout_plugin():
    scripts = Path(sys.executable).parent
    candidates = [scripts / "archbench.exe", scripts / "Scripts" / "archbench.exe"]
    executable = next((str(path) for path in candidates if path.exists()), None)
    executable = executable or shutil.which("archbench")
    if executable is None:
        pytest.skip("installed archbench console entry point unavailable")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    process = subprocess.run(
        [
            executable,
            "run",
            "--plugin",
            "examples.plugin",
            "--model",
            "toy_rnn",
            "--credit",
            "zero_credit",
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--horizon",
            "2",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(process.stdout)
    assert result["model"] == "toy_rnn"
    assert result["rho"] is None


def test_cosmos_placeholder_has_helpful_error(capsys):
    with pytest.raises(SystemExit) as error:
        main(["run", "--model", "cosmos", "--steps", "1"])
    assert error.value.code == 2
    assert "experimental" in capsys.readouterr().err.lower()


def test_bptt_alignment_replays_dropout_randomness(monkeypatch, capsys):
    def stochastic_model(name, config):
        return nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size),
            nn.Dropout(0.5),
            nn.Linear(config.hidden_size, config.output_size),
        )

    monkeypatch.setattr("archcredit.cli.create_architecture", stochastic_model)
    main(["run", "--credit", "bptt", "--steps", "2", "--batch-size", "4", "--horizon", "2"])
    assert json.loads(capsys.readouterr().out)["rho"] == pytest.approx(1)
