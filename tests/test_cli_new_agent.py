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
    assert '"apollo":' in output
    assert '- "apollo"' in output

    (workspace / "AGENTS.md").write_text("edited\n", encoding="utf-8")
    (workspace / "SOUL.md").unlink()
    output = _new_agent(config_path, "apollo", "project")

    assert (workspace / "AGENTS.md").read_text(encoding="utf-8") == "edited\n"
    assert (workspace / "SOUL.md").is_file()
    assert "Created 1 file in" in output


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
@pytest.mark.parametrize("name", ["apollo", "on", "123", "null"])
def test_agent_block_validates_in_the_starter_config(tmp_path: Path, template: str, name: str) -> None:
    """Each printed block is valid config once pasted into the starter config, and its context files exist."""
    config_path = tmp_path / "config.yaml"
    init = runner.invoke(app, ["config", "init", "--path", str(config_path), "--provider", "anthropic", "--no-input"])
    assert init.exit_code == 0, init.output
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _new_agent(config_path, name, template)

    block = yaml.safe_load(config_cli._agent_template_block(template, name))
    data["agents"].update(block["agents"])

    config = Config.validate_with_runtime(data, resolve_runtime_paths(config_path=config_path))
    agent = config.agents[name]
    assert agent.delegate_to == [name]
    if agent.private is None:
        written, context_files = tmp_path / "mindroom_data" / "agents" / name / "workspace", agent.context_files
    else:
        written, context_files = tmp_path / "templates" / name, agent.private.context_files or []
    assert context_files
    assert all((written / context_file).is_file() for context_file in context_files)
    assert (agent.private is None) == (template == "project")


@pytest.mark.parametrize(
    ("name", "message"),
    [("my-agent", "letters, digits, and underscores"), ("router", "reserved"), ("_shared", "reserved")],
)
def test_rejects_an_unusable_agent_name(tmp_path: Path, name: str, message: str) -> None:
    """A name config validation would reject exits before writing anything."""
    result = runner.invoke(
        app,
        ["config", "new-agent", name, "--template", "project", "--path", str(tmp_path / "config.yaml")],
    )

    assert result.exit_code == 1
    assert message in result.output
    assert not (tmp_path / "mindroom_data").exists()


def test_reports_a_planted_symlink_without_following_it(tmp_path: Path) -> None:
    """A workspace entry that points outside it fails with one line and writes nothing there."""
    config_path = tmp_path / "config.yaml"
    workspace = tmp_path / "mindroom_data" / "agents" / "apollo" / "workspace"
    outside = tmp_path / "outside"
    outside.mkdir()
    workspace.mkdir(parents=True)
    (workspace / "skills").symlink_to(outside)

    result = runner.invoke(app, ["config", "new-agent", "apollo", "--template", "project", "--path", str(config_path)])

    assert result.exit_code == 1
    assert "Could not create the template files" in result.output
    assert "Traceback" not in result.output
    assert list(outside.iterdir()) == []
