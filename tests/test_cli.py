import json
from pathlib import Path

import pytest

from src import cli


def _write_config(path: Path, output_dir: Path, target_name: str = "DemoTarget"):
    path.write_text(
        f"""
target_name: "{target_name}"
stock_code: "000001"
target_type: "company"
output_dir: "{output_dir.as_posix()}"
language: "zh"
custom_collect_tasks:
  - "Collect revenue"
custom_analysis_tasks:
  - "Analyze profitability"
llm_config_list:
  - model_name: "fake-model"
    api_key: "fake-key"
    base_url: "https://example.com"
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_cli_help_smoke(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert "QAssistant" in capsys.readouterr().out

    with pytest.raises(SystemExit) as exc:
        cli.main(["report", "run", "--help"])
    assert exc.value.code == 0
    assert "report pipeline" in capsys.readouterr().out

    with pytest.raises(SystemExit) as exc:
        cli.main(["reproduce", "--help"])
    assert exc.value.code == 0
    assert "PDF" in capsys.readouterr().out


def test_config_validate_accepts_config_and_legacy_tasks(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    tasks_path = tmp_path / "tasks.json"
    _write_config(config_path, tmp_path / "outputs")
    tasks_path.write_text(
        json.dumps({
            "collect_tasks": [{"id": "c1", "type": "collect", "content": "Collect share price"}],
            "analysis_tasks": [{"id": "a1", "type": "analyze", "content": "Analyze valuation"}],
        }),
        encoding="utf-8",
    )

    exit_code = cli.main(["config", "validate", "--config", str(config_path), "--tasks-file", str(tasks_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Config status: valid" in output
    assert "1 collect, 1 analysis" in output


def test_config_validate_reports_missing_required_key(tmp_path, capsys):
    config_path = tmp_path / "bad_config.yaml"
    config_path.write_text("output_dir: ./outputs\nllm_config_list: []\n", encoding="utf-8")

    exit_code = cli.main(["config", "validate", "--config", str(config_path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "Config status: invalid" in output
    assert "Missing required config key: target_name" in output


def test_outputs_list_json_uses_configured_target_dir(tmp_path, capsys):
    output_root = tmp_path / "outputs"
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, output_root, target_name="DemoTarget")
    target_dir = output_root / "DemoTarget"
    target_dir.mkdir(parents=True)
    (target_dir / "report.md").write_text("# Report\n", encoding="utf-8")
    reproduction_dir = target_dir / "report_reproduction" / "sample"
    reproduction_dir.mkdir(parents=True)
    (reproduction_dir / "sample_strategy.py").write_text("PARAMS = {}\n", encoding="utf-8")

    exit_code = cli.main(["outputs", "list", "--config", str(config_path), "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["exists"] is True
    assert payload["count"] == 2
    assert {item["name"] for item in payload["artifacts"]} == {"report.md", "sample_strategy.py"}


def test_pyproject_registers_qassistant_script():
    import tomllib

    payload = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert payload["project"]["scripts"]["QAssistant"] == "src.cli:main"
