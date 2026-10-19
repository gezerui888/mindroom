"""Skill scripts run as commands of the agent's own shell tool, wherever and however that shell runs."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from agno.run import RunContext
from agno.run.agent import RunOutput
from agno.session import AgentSession
from agno.tools.function import Function, FunctionCall

from mindroom.agents import create_agent
from mindroom.background_tasks import wait_for_background_tasks
from mindroom.config.main import Config
from mindroom.config.plugin import PluginEntryConfig
from mindroom.constants import resolve_runtime_paths
from mindroom.hooks import (
    EVENT_TOOL_AFTER_CALL,
    EVENT_TOOL_BEFORE_CALL,
    HookRegistry,
    ToolAfterCallContext,
    ToolBeforeCallContext,
    hook,
)
from mindroom.tool_system import sandbox_proxy
from mindroom.tool_system.dynamic_toolkits import load_tool_for_session, save_loaded_tools_for_session
from mindroom.tool_system.skills import clear_skill_cache, get_plugin_skill_roots, set_plugin_skill_roots
from mindroom.tool_system.worker_routing import agent_workspace_root_path
from tests.conftest import SETTLE_TIMEOUT_SECONDS
from tests.identity_helpers import persist_entity_accounts

if TYPE_CHECKING:
    from collections.abc import Iterator

    from agno.agent import Agent

_DOCKER_WORKER_ENV = {
    "MINDROOM_WORKER_BACKEND": "docker",
    "MINDROOM_DOCKER_WORKER_IMAGE": "mindroom-worker:test",
    "MINDROOM_DOCKER_WORKER_STORAGE_MOUNT_PATH": "/srv/worker-storage",
    "MINDROOM_SANDBOX_PROXY_TOKEN": "test-token",
}
# Leaves a marker beside itself, so a test sees whether it ran on this host.
_SCRIPT = '#!/bin/sh\ntouch "$0.ran"\necho "checked $1 in $PWD"\n'
_ORIGINS = ("workspace", "plugin", "plugin_below_storage")
_SESSION_ID = "session"


@pytest.fixture(autouse=True)
def _restore_plugin_skill_roots() -> Iterator[None]:
    """Plugin skill roots are process-wide, so put them back after each test."""
    original = get_plugin_skill_roots()
    yield
    set_plugin_skill_roots(original)
    clear_skill_cache()


@pytest.fixture
def worker_commands(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, dict[str, object]]]:
    """Record each call that reaches a worker instead of starting one; tool construction and proxy wrappers stay real."""
    commands: list[tuple[str, str, dict[str, object]]] = []

    def dispatch(*, tool_name: str, function_name: str, kwargs: dict[str, object], **_kwargs: object) -> str:
        commands.append((tool_name, function_name, kwargs))
        return "ran in the worker"

    monkeypatch.setattr(sandbox_proxy, "_call_proxy_sync", dispatch)
    return commands


def _storage(tmp_path: Path) -> Path:
    return (tmp_path / "storage").resolve()


@dataclass(frozen=True)
class _Skill:
    """Skill ``xl_check`` with its one script, and the config ``plugins`` entry a plugin skill needs."""

    script: Path
    plugins: list[str]

    @property
    def ran_here(self) -> bool:
        return self.script.with_name("check.sh.ran").exists()


def _write_skill(tmp_path: Path, origin: str) -> _Skill:
    plugin_dir = {
        "workspace": None,
        "plugin": tmp_path.resolve() / "ops",
        # Like a config bundle copied onto the storage volume.
        "plugin_below_storage": _storage(tmp_path) / "config-bundle" / "plugins" / "ops",
    }[origin]
    if plugin_dir is None:
        skills_root = agent_workspace_root_path(_storage(tmp_path), "general") / "skills"
    else:
        skills_root = plugin_dir / "skills"
        skills_root.mkdir(parents=True)
        (plugin_dir / "mindroom.plugin.json").write_text(json.dumps({"name": "ops", "skills": ["skills"]}))
    skill_dir = skills_root / "xl_check"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: xl_check\ndescription: Check a workbook\n---\n\nRun check.sh.\n")
    script = skill_dir / "scripts" / "check.sh"
    script.write_text(_SCRIPT)
    return _Skill(script=script, plugins=[] if plugin_dir is None else [str(plugin_dir)])


def _create_agent(
    tmp_path: Path,
    skill: _Skill,
    *,
    tools: list[object],
    process_env: dict[str, str] | None = None,
    agent_settings: dict[str, object] | None = None,
    config_settings: dict[str, object] | None = None,
    hook_registry: HookRegistry | None = None,
    loaded_tools: tuple[str, ...] = (),
) -> Agent:
    runtime_paths = resolve_runtime_paths(
        config_path=tmp_path / "config.yaml",
        storage_path=_storage(tmp_path),
        process_env=process_env or {},
    )
    config = Config.validate_with_runtime(
        {
            "models": {"default": {"provider": "ollama", "id": "test-model"}},
            "agents": {
                "general": {
                    "display_name": "General",
                    "tools": tools,
                    "include_default_tools": False,
                    # File memory gives the agent a workspace, where its shell runs.
                    "memory_backend": "file",
                    "skills": ["xl_check"],
                    **(agent_settings or {}),
                },
            },
            "plugins": skill.plugins,
            **(config_settings or {}),
        },
        runtime_paths,
    )
    persist_entity_accounts(config, runtime_paths, usernames={"router": "router", "general": "general"})
    for tool_name in loaded_tools:
        # As load_tool does before the agent is built again.
        load_tool_for_session(agent_name="general", config=config, session_id=_SESSION_ID, tool_name=tool_name)
    try:
        return create_agent(
            "general",
            config,
            runtime_paths,
            execution_identity=None,
            session_id=_SESSION_ID,
            include_interactive_questions=False,
            persist_runtime_state=False,
            supports_native_tool_approval=True,
            hook_registry=hook_registry,
        )
    finally:
        # Loaded tools are kept per session for the whole process.
        save_loaded_tools_for_session(agent_name="general", session_id=_SESSION_ID, loaded_tools=[], config=config)


async def _get_skill_script(agent: Agent, *, execute: bool, script_path: str = "check.sh") -> str:
    """Call ``get_skill_script`` the way a model's tool call runs it, with the agent's hooks."""
    tools = await agent.aget_tools(
        RunOutput(run_id="run", agent_id="general", agent_name="General", session_id=_SESSION_ID),
        RunContext(run_id="run", session_id=_SESSION_ID),
        AgentSession(session_id=_SESSION_ID, agent_id="general", created_at=1, updated_at=1),
    )
    [function] = [tool for tool in tools if isinstance(tool, Function) and tool.name == "get_skill_script"]
    arguments = {"skill_name": "xl_check", "script_path": script_path, "execute": execute, "args": ["book.xlsx"]}
    execution = await FunctionCall(function=function, arguments=arguments, call_id="call-1").aexecute()
    assert execution.status == "success", execution.error
    return str(execution.result)


@pytest.mark.parametrize("origin", _ORIGINS)
@pytest.mark.asyncio
async def test_skill_scripts_run_in_the_worker_of_the_agents_shell(
    tmp_path: Path,
    worker_commands: list[tuple[str, str, dict[str, object]]],
    origin: str,
) -> None:
    """A script runs as a shell command in the agent's worker, at the path the worker sees, and never on this host."""
    skill = _write_skill(tmp_path, origin)
    agent = _create_agent(
        tmp_path,
        skill,
        tools=["shell"],
        process_env=_DOCKER_WORKER_ENV,
        agent_settings={"worker_tools": ["shell"], "worker_scope": "shared"},
    )

    result = await _get_skill_script(agent, execute=True)

    # The worker mounts the storage root at its own path and sees other paths where this host does.
    storage = _storage(tmp_path)
    worker_script = (
        Path("/srv/worker-storage") / skill.script.relative_to(storage)
        if skill.script.is_relative_to(storage)
        else skill.script
    )
    command = shlex.join(["/bin/sh", str(worker_script), "book.xlsx"])
    assert worker_commands == [("shell", "run_shell_command", {"args": command, "timeout": 30})]
    assert result == "ran in the worker"
    assert not skill.ran_here


@pytest.mark.parametrize("origin", ["workspace", "plugin"])
@pytest.mark.asyncio
async def test_an_unlisted_skill_script_never_reaches_the_shell(
    tmp_path: Path,
    worker_commands: list[tuple[str, str, dict[str, object]]],
    origin: str,
) -> None:
    """A script name the skill does not list gets Agno's not-found answer, and no command runs."""
    skill = _write_skill(tmp_path, origin)
    agent = _create_agent(
        tmp_path,
        skill,
        tools=["shell"],
        process_env=_DOCKER_WORKER_ENV,
        agent_settings={"worker_tools": ["shell"], "worker_scope": "shared"},
    )

    result = json.loads(await _get_skill_script(agent, execute=True, script_path="../SKILL.md"))

    assert "not found" in result["error"]
    assert result["available_scripts"] == ["check.sh"]
    assert worker_commands == []


@pytest.mark.parametrize("origin", ["workspace", "plugin"])
@pytest.mark.asyncio
async def test_skill_scripts_run_on_this_host_when_the_agents_shell_does(tmp_path: Path, origin: str) -> None:
    """With a local shell, a script runs on this host as the agent's shell commands do, in the agent's workspace."""
    skill = _write_skill(tmp_path, origin)
    agent = _create_agent(tmp_path, skill, tools=["shell"])

    result = await _get_skill_script(agent, execute=True)

    workspace = agent_workspace_root_path(_storage(tmp_path), "general")
    assert result.rstrip("\n") == f"[cwd: {workspace}]\nchecked book.xlsx in {workspace}"
    assert skill.ran_here


@pytest.mark.asyncio
async def test_running_a_workspace_skill_script_counts_as_using_the_skill(tmp_path: Path) -> None:
    """A run, like a read, keeps a learned workspace skill from being archived as unused."""
    skill = _write_skill(tmp_path, "workspace")
    agent = _create_agent(tmp_path, skill, tools=["shell"])

    await _get_skill_script(agent, execute=True)
    assert await wait_for_background_tasks(SETTLE_TIMEOUT_SECONDS)

    usage = json.loads((skill.script.parents[2] / ".usage.json").read_text())
    assert "last_used_at" in usage["xl_check"]


@pytest.mark.parametrize("origin", ["workspace", "plugin"])
@pytest.mark.asyncio
async def test_skill_scripts_need_the_agents_shell(tmp_path: Path, origin: str) -> None:
    """An agent without a shell tool can read a skill script but not run it."""
    skill = _write_skill(tmp_path, origin)
    agent = _create_agent(tmp_path, skill, tools=["calculator"])

    refused = json.loads(await _get_skill_script(agent, execute=True))
    read = json.loads(await _get_skill_script(agent, execute=False))

    assert "shell tool" in refused["error"]
    assert "execute=false" in refused["error"]
    assert not skill.ran_here
    assert read["content"] == _SCRIPT


@pytest.mark.parametrize("loaded", [False, True], ids=["deferred", "loaded"])
@pytest.mark.asyncio
async def test_a_deferred_shell_runs_skill_scripts_once_the_agent_loads_it(tmp_path: Path, loaded: bool) -> None:
    """An agent has a shell for its scripts exactly when it has one for its own commands."""
    skill = _write_skill(tmp_path, "plugin")
    agent = _create_agent(
        tmp_path,
        skill,
        tools=[{"shell": {"defer": True}}],
        loaded_tools=("shell",) if loaded else (),
    )

    result = await _get_skill_script(agent, execute=True)

    workspace = agent_workspace_root_path(_storage(tmp_path), "general")
    assert (result.rstrip("\n") == f"[cwd: {workspace}]\nchecked book.xlsx in {workspace}") is loaded
    assert skill.ran_here is loaded


@pytest.mark.parametrize("jobs", [False, True], ids=["paused-run", "job-approval"])
@pytest.mark.asyncio
async def test_skill_scripts_are_refused_while_the_agents_shell_needs_approval(tmp_path: Path, jobs: bool) -> None:
    """A skill function cannot wait for approval, so a script whose shell commands need it is left to the shell."""
    skill = _write_skill(tmp_path, "plugin")
    agent = _create_agent(
        tmp_path,
        skill,
        tools=["shell"],
        config_settings={
            "tool_approval": {"rules": [{"match": "run_shell_command", "action": "require_approval"}]},
            "background_tool_jobs": {"enabled": jobs},
        },
    )

    refused = json.loads(await _get_skill_script(agent, execute=True))

    assert "approval" in refused["error"]
    assert shlex.join(["/bin/sh", str(skill.script), "book.xlsx"]) in refused["error"]
    assert not skill.ran_here


@pytest.mark.asyncio
async def test_tool_hooks_see_a_skill_script_as_the_shell_call_that_runs_it(tmp_path: Path) -> None:
    """A plugin's shell policy applies to scripts too: its hooks see the shell call and can decline it."""
    before_calls: list[tuple[str, dict[str, object]]] = []
    after_calls: list[tuple[str, bool]] = []

    @hook(EVENT_TOOL_BEFORE_CALL)
    async def gate(ctx: ToolBeforeCallContext) -> None:
        before_calls.append((ctx.tool_name, ctx.arguments))
        if ctx.tool_name == "run_shell_command":
            ctx.decline("no shell commands today")

    @hook(EVENT_TOOL_AFTER_CALL)
    async def audit(ctx: ToolAfterCallContext) -> None:
        after_calls.append((ctx.tool_name, ctx.blocked))

    plugin = SimpleNamespace(
        name="shell-policy",
        entry_config=PluginEntryConfig(path="shell-policy"),
        plugin_order=0,
        discovered_hooks=(gate, audit),
        discovered_automations=(),
    )
    skill = _write_skill(tmp_path, "plugin")
    agent = _create_agent(tmp_path, skill, tools=["shell"], hook_registry=HookRegistry.from_plugins([plugin]))

    result = await _get_skill_script(agent, execute=True)

    assert [name for name, _ in before_calls] == ["get_skill_script", "run_shell_command"]
    assert before_calls[1][1] == {"args": shlex.join(["/bin/sh", str(skill.script), "book.xlsx"]), "timeout": 30}
    assert after_calls == [("run_shell_command", True), ("get_skill_script", False)]
    assert "[TOOL CALL DECLINED]" in result
    assert "no shell commands today" in result
    assert not skill.ran_here
