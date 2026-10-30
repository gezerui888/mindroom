"""Tests for `mindroom config new-agent`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml
from typer.testing import CliRunner

from mindroom.cli import config as config_cli
from mindroom.cli.main import app
from mindroom.config.main import Config
from mindroom.constants import resolve_runtime_paths

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()
pytestmark = pytest.mark.usefixtures("wide_cli_consoles")


@pytest.fixture(autouse=True)
def _clear_runtime_path_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINDROOM_CONFIG_PATH", raising=False)
    monkeypatch.delenv("MINDROOM_STORAGE_PATH", raising=False)


def _new_agent(config_path: Path, name: str, template: str) -> str:
    result = runner.invoke(app, ["config", "new-agent", name, "--template", template, "--path", str(config_path)])
    assert result.exit_code == 0, result.output
    return result.output


def test_project_template_seeds_the_shared_workspace_and_keeps_edits(tmp_path: Path) -> None:
    """A project agent gets its workspace seeded, and a rerun restores only missing files."""
    config_path = tmp_path / "config.yaml"
    workspace = tmp_path / "mindroom_data" / "agents" / "apollo" / "workspace"

    output = _new_agent(config_path, "apollo", "project")

    assert (workspace / "AGENTS.md").is_file()
    assert (workspace / "BOOTSTRAP.md").is_file()
    assert (workspace / "memory" / "decisions.md").is_file()
    assert (workspace / "skills" / "status-report" / "SKILL.md").is_file()
    assert "Created 14 files" in output
    assert "apollo:" in output
    assert "- apollo" in output

    (workspace / "AGENTS.md").write_text("edited\n", encoding="utf-8")
    (workspace / "SOUL.md").unlink()
    output = _new_agent(config_path, "apollo", "project")

    assert (workspace / "AGENTS.md").read_text(encoding="utf-8") == "edited\n"
    assert (workspace / "SOUL.md").is_file()
    assert "Created 1 file" in output
    assert "existing files were kept" in output


def test_work_template_writes_a_template_dir_the_printed_block_points_at(tmp_path: Path) -> None:
    """A work agent gets a template directory next to the config that its printed block names."""
    config_path = tmp_path / "config.yaml"

    output = _new_agent(config_path, "helper", "work")

    template_dir = tmp_path / "templates" / "helper"
    assert (template_dir / "USER.md").is_file()
    assert (template_dir / ".gitignore").is_file()
    assert (template_dir / "skills" / "weekly-review" / "SKILL.md").is_file()
    assert "Created 10 files" in output
    assert "template_dir: templates/helper" in output
    assert f"mindroom config validate --path {config_path}" in output
    assert not (tmp_path / "mindroom_data" / "agents" / "helper").exists()


@pytest.mark.parametrize("template", ["project", "work"])
def test_agent_block_validates_in_the_starter_config(tmp_path: Path, template: str) -> None:
    """Each printed block is valid config once pasted into the starter config."""
    config_path = tmp_path / "config.yaml"
    init = runner.invoke(app, ["config", "init", "--path", str(config_path), "--provider", "anthropic", "--no-input"])
    assert init.exit_code == 0, init.output
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _new_agent(config_path, "apollo", template)

    block = yaml.safe_load(config_cli._agent_template_block(template, "apollo"))
    data["agents"].update(block["agents"])

    config = Config.validate_with_runtime(data, resolve_runtime_paths(config_path=config_path))
    agent = config.agents["apollo"]
    assert agent.delegate_to == ["apollo"]
    assert (agent.private is None) == (template == "project")


def test_rejects_an_invalid_agent_name(tmp_path: Path) -> None:
    """A name outside the agent naming rules exits before writing anything."""
    result = runner.invoke(
        app,
        ["config", "new-agent", "my-agent", "--template", "project", "--path", str(tmp_path / "config.yaml")],
    )

    assert result.exit_code == 1
    assert "letters, digits, and underscores" in result.output
    assert not (tmp_path / "mindroom_data").exists()
