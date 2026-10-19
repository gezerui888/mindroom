"""Skill scripts run as commands of the agent's own shell tool, wherever and however that shell runs."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import yaml
from agno.run import RunContext
from agno.run.agent import RunOutput
from agno.session import AgentSession
from agno.tools.function import Function, FunctionCall

from mindroom.agents import create_agent
from mindroom.agno_compat_prepared_tools import prepare_agent_tools
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
from mindroom.tool_system import skills as skills_module
from mindroom.tool_system.dynamic_toolkits import load_tool_for_session, save_loaded_tools_for_session
from mindroom.tool_system.skills import clear_skill_cache, get_plugin_skill_roots, set_plugin_skill_roots
from mindroom.tool_system.worker_routing import ToolExecutionIdentity, agent_workspace_root_path
from mindroom.workers.models import WorkerSpec
from mindroom.workers.runtime import serialized_kubernetes_worker_config_snapshot
from tests.conftest import SETTLE_TIMEOUT_SECONDS
from tests.identity_helpers import persist_entity_accounts
from tests.test_docker_worker_backend import _backend as docker_backend
from tests.test_kubernetes_worker_backend import _backend as kubernetes_backend

if TYPE_CHECKING:
    from collections.abc import Iterator

    from agno.agent import Agent

    from mindroom.constants import RuntimePaths
    from mindroom.tool_system.worker_routing import ResolvedWorkerTarget

# Both backend test helpers mount the worker storage here.
_WORKER_STORAGE = PurePosixPath("/app/worker")
_PROXY_TOKEN = {"MINDROOM_SANDBOX_PROXY_TOKEN": "test-token"}
_WORKER_ENVS = {
    "docker": {
        "MINDROOM_WORKER_BACKEND": "docker",
        "MINDROOM_DOCKER_WORKER_IMAGE": "mindroom-worker:test",
        "MINDROOM_DOCKER_WORKER_STORAGE_MOUNT_PATH": str(_WORKER_STORAGE),
        **_PROXY_TOKEN,
    },
    "kubernetes": {
        "MINDROOM_WORKER_BACKEND": "kubernetes",
        "MINDROOM_KUBERNETES_WORKER_IMAGE": "mindroom-worker:test",
        "MINDROOM_KUBERNETES_WORKER_STORAGE_PVC_NAME": "mindroom-storage",
        "MINDROOM_KUBERNETES_WORKER_STORAGE_MOUNT_PATH": str(_WORKER_STORAGE),
        **_PROXY_TOKEN,
    },
    # The shared runner, which sees the agents' workspaces where the primary does but no plugin or user skill.
    "static_runner": {"MINDROOM_SANDBOX_PROXY_URL": "http://sandbox-runner:8766", **_PROXY_TOKEN},
}
# A shared worker is resolved from who is talking to the agent where.
_IDENTITY = ToolExecutionIdentity(
    channel="matrix",
    agent_name="general",
    requester_id="@user:example.org",
    room_id="!room:example.org",
    thread_id=None,
    resolved_thread_id=None,
    session_id=None,
    tenant_id="tenant",
)
# Leaves a marker beside itself, so a test sees whether it ran on this host.
_SCRIPT = '#!/bin/sh\ntouch "$0.ran"\necho "checked $1 in $PWD"\n'
_SESSION_ID = "session"


@pytest.fixture(autouse=True)
def _restore_plugin_skill_roots() -> Iterator[None]:
    """Plugin skill roots are process-wide, so put them back after each test."""
    original = get_plugin_skill_roots()
    yield
    set_plugin_skill_roots(original)
    clear_skill_cache()


@dataclass(frozen=True)
class _WorkerCall:
    """One call that reached a worker, and the worker key it was routed to."""

    tool_name: str
    function_name: str
    kwargs: dict[str, object]
    worker_key: str | None


@pytest.fixture
def worker_calls(monkeypatch: pytest.MonkeyPatch) -> list[_WorkerCall]:
    """Record each call that reaches a worker instead of starting one; tool construction and proxy wrappers stay real."""
    calls: list[_WorkerCall] = []

    def dispatch(
        *,
        tool_name: str,
        function_name: str,
        kwargs: dict[str, object],
        worker_target: ResolvedWorkerTarget | None = None,
        **_kwargs: object,
    ) -> str:
        worker_key = None if worker_target is None else worker_target.worker_key
        calls.append(_WorkerCall(tool_name, function_name, kwargs, worker_key))
        return "ran in the worker"

    monkeypatch.setattr(sandbox_proxy, "_call_proxy_sync", dispatch)
    return calls


def _storage(tmp_path: Path) -> Path:
    return (tmp_path / "storage").resolve()


@dataclass(frozen=True)
class _Skill:
    """Skill ``xl_check`` with its one script, and the config ``plugins`` entry a plugin skill needs."""

    script: Path
    plugins: list[str]

    @property
    def directory(self) -> Path:
        return self.script.parents[1]

    @property
    def ran_here(self) -> bool:
        return self.script.with_name("check.sh.ran").exists()


def _write_skill(tmp_path: Path, origin: str, monkeypatch: pytest.MonkeyPatch | None = None) -> _Skill:
    plugin_dir = {
        "workspace": None,
        "bundled": None,
        "plugin": tmp_path.resolve() / "ops",
        # Like a config bundle copied onto the storage volume.
        "plugin_below_storage": _storage(tmp_path) / "config-bundle" / "plugins" / "ops",
    }[origin]
    if origin == "workspace":
        skills_root = agent_workspace_root_path(_storage(tmp_path), "general") / "skills"
    elif origin == "bundled":
        # Bundled skills ship with MindRoom itself, outside the storage root.
        assert monkeypatch is not None
        skills_root = tmp_path.resolve() / "mindroom" / "skills"
        monkeypatch.setattr(skills_module, "_get_bundled_skills_dir", lambda: skills_root)
    else:
        assert plugin_dir is not None
        skills_root = plugin_dir / "skills"
        skills_root.mkdir(parents=True)
        (plugin_dir / "mindroom.plugin.json").write_text(json.dumps({"name": "ops", "skills": ["skills"]}))
    skill_dir = skills_root / "xl_check"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: xl_check\ndescription: Check a workbook\n---\n\nRun check.sh.\n")
    script = skill_dir / "scripts" / "check.sh"
    script.write_text(_SCRIPT)
    return _Skill(script=script, plugins=[] if plugin_dir is None else [str(plugin_dir)])


@dataclass(frozen=True)
class _Setup:
    """An agent built from a config, and that config as the primary and its worker backends read it."""

    agent: Agent
    config: Config
    runtime_paths: RuntimePaths
    authored: dict[str, object]


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
    execution_identity: ToolExecutionIdentity | None = None,
) -> _Setup:
    runtime_paths = resolve_runtime_paths(
        config_path=tmp_path / "config.yaml",
        storage_path=_storage(tmp_path),
        process_env=process_env or {},
    )
    authored = {
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
    }
    config = Config.validate_with_runtime(authored, runtime_paths)
    persist_entity_accounts(config, runtime_paths, usernames={"router": "router", "general": "general"})
    for tool_name in loaded_tools:
        # As load_tool does before the agent is built again.
        load_tool_for_session(agent_name="general", config=config, session_id=_SESSION_ID, tool_name=tool_name)
    try:
        agent = create_agent(
            "general",
            config,
            runtime_paths,
            execution_identity=execution_identity,
            session_id=_SESSION_ID,
            include_interactive_questions=False,
            persist_runtime_state=False,
            supports_native_tool_approval=True,
            hook_registry=hook_registry,
        )
    finally:
        # Loaded tools are kept per session for the whole process.
        save_loaded_tools_for_session(agent_name="general", session_id=_SESSION_ID, loaded_tools=[], config=config)
    return _Setup(agent=agent, config=config, runtime_paths=runtime_paths, authored=authored)


def _create_worker_agent(tmp_path: Path, skill: _Skill, backend_name: str) -> _Setup:
    """Build the agent with its shell routed to one shared worker of ``backend_name``."""
    return _create_agent(
        tmp_path,
        skill,
        tools=["shell"],
        process_env=_WORKER_ENVS[backend_name],
        agent_settings={"worker_tools": ["shell"], "worker_scope": "shared"},
        execution_identity=_IDENTITY,
    )


async def _get_skill_script(agent: Agent, *, execute: bool, script_path: str = "check.sh") -> str:
    """Call ``get_skill_script`` the way a model's tool call runs it, prepared for the run with the agent's hooks."""
    run_output = RunOutput(run_id="run", agent_id="general", agent_name="General", session_id=_SESSION_ID)
    run_context = RunContext(run_id="run", session_id=_SESSION_ID)
    session = AgentSession(session_id=_SESSION_ID, agent_id="general", created_at=1, updated_at=1)
    tools = prepare_agent_tools(
        agent,
        processed_tools=await agent.aget_tools(run_output, run_context, session),
        run_response=run_output,
        run_context=run_context,
        session=session,
    )
    [function] = [tool for tool in tools if isinstance(tool, Function) and tool.name == "get_skill_script"]
    arguments = {"skill_name": "xl_check", "script_path": script_path, "execute": execute, "args": ["book.xlsx"]}
    execution = await FunctionCall(function=function, arguments=arguments, call_id="call-1").aexecute()
    assert execution.status == "success", execution.error
    return str(execution.result)


@dataclass(frozen=True)
class _Mount:
    source: Path
    target: PurePosixPath
    read_only: bool


def _worker_mounts(
    backend_name: str,
    setup: _Setup,
    worker_key: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> list[_Mount]:
    """Start the worker with the real backend, faking only the container runtime, and return its storage mounts."""
    storage = setup.runtime_paths.storage_root
    if backend_name == "docker":
        backend, client, _credential_syncs = docker_backend(
            monkeypatch,
            tmp_path,
            config_text=yaml.safe_dump(setup.authored),
            runtime_paths=setup.runtime_paths,
            storage_path=storage,
        )
        backend.ensure_worker(WorkerSpec(worker_key), now=10.0)
        volumes = client.containers.run_calls[0]["volumes"]
        assert isinstance(volumes, list)
        return [
            _Mount(Path(source), PurePosixPath(target), mode == "ro")
            for source, target, mode in (str(volume).rsplit(":", 2) for volume in volumes)
        ]
    backend, apps_api, _core_api = kubernetes_backend(
        runtime_paths=setup.runtime_paths,
        config_snapshot=serialized_kubernetes_worker_config_snapshot(setup.config),
    )
    backend.ensure_worker(WorkerSpec(worker_key), now=10.0)
    [container] = apps_api.created_bodies[0]["spec"]["template"]["spec"]["containers"]
    return [
        _Mount(storage / mount["subPath"], PurePosixPath(mount["mountPath"]), mount.get("readOnly", False))
        for mount in container["volumeMounts"]
        if mount["name"] == "worker-storage"
    ]


@pytest.mark.parametrize("backend_name", ["docker", "kubernetes"])
@pytest.mark.parametrize("origin", ["workspace", "plugin_below_storage"])
@pytest.mark.asyncio
async def test_skill_scripts_run_in_the_worker_from_its_own_mounts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_calls: list[_WorkerCall],
    backend_name: str,
    origin: str,
) -> None:
    """A script runs as a shell command in the agent's worker, at a path the worker mounts, and never on this host.

    A workspace skill is in the writable workspace mount, and a configured skill in a read-only mount of its own
    directory, as the real backend starts the worker the call is routed to.
    """
    skill = _write_skill(tmp_path, origin)
    setup = _create_worker_agent(tmp_path, skill, backend_name)

    result = await _get_skill_script(setup.agent, execute=True)

    [call] = worker_calls
    assert (call.tool_name, call.function_name, call.kwargs["timeout"]) == ("shell", "run_shell_command", 30)
    interpreter, script, *args = shlex.split(str(call.kwargs["args"]))
    assert (interpreter, args) == ("/bin/sh", ["book.xlsx"])
    assert result == "ran in the worker"
    assert not skill.ran_here
    assert call.worker_key is not None
    mounts = _worker_mounts(backend_name, setup, call.worker_key, monkeypatch, tmp_path)
    worker_script = PurePosixPath(script)
    # The innermost mount holding the script decides what the worker finds there.
    mount = max(
        (mount for mount in mounts if worker_script.is_relative_to(mount.target)),
        key=lambda mount: len(mount.target.parts),
        default=None,
    )
    storage = setup.runtime_paths.storage_root
    workspace = agent_workspace_root_path(storage, "general")
    expected = (
        _Mount(workspace, _WORKER_STORAGE / workspace.relative_to(storage), read_only=False)
        if origin == "workspace"
        else _Mount(skill.directory, _WORKER_STORAGE / skill.directory.relative_to(storage), read_only=True)
    )
    assert mount == expected
    assert worker_script == mount.target / skill.script.relative_to(mount.source)


@pytest.mark.parametrize(
    ("backend_name", "origin"),
    [
        ("docker", "plugin"),
        ("kubernetes", "plugin"),
        ("static_runner", "plugin"),
        # The shared runner does not see a configured skill below the storage root either.
        ("static_runner", "plugin_below_storage"),
    ],
)
@pytest.mark.asyncio
async def test_skill_scripts_a_worker_cannot_see_are_refused(
    tmp_path: Path,
    worker_calls: list[_WorkerCall],
    backend_name: str,
    origin: str,
) -> None:
    """A script whose directory the agent's worker does not mount runs nothing there, and can still be read."""
    skill = _write_skill(tmp_path, origin)
    setup = _create_worker_agent(tmp_path, skill, backend_name)

    refused = json.loads(await _get_skill_script(setup.agent, execute=True))
    read = json.loads(await _get_skill_script(setup.agent, execute=False))

    assert "cannot see" in refused["error"]
    assert "execute=false" in refused["error"]
    assert worker_calls == []
    assert not skill.ran_here
    assert read["content"] == _SCRIPT


@pytest.mark.parametrize(
    ("backend_name", "origin"),
    [
        ("docker", "bundled"),
        ("kubernetes", "bundled"),
        ("static_runner", "bundled"),
        ("static_runner", "workspace"),
    ],
)
@pytest.mark.asyncio
async def test_skill_scripts_run_where_every_worker_sees_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_calls: list[_WorkerCall],
    backend_name: str,
    origin: str,
) -> None:
    """Worker images ship the bundled skills, and the shared runner sees workspaces, where this host has them."""
    skill = _write_skill(tmp_path, origin, monkeypatch)
    setup = _create_worker_agent(tmp_path, skill, backend_name)

    await _get_skill_script(setup.agent, execute=True)

    assert [call.kwargs["args"] for call in worker_calls] == [shlex.join(["/bin/sh", str(skill.script), "book.xlsx"])]
    assert not skill.ran_here


@pytest.mark.parametrize("origin", ["workspace", "plugin"])
@pytest.mark.asyncio
async def test_an_unlisted_skill_script_never_reaches_the_shell(
    tmp_path: Path,
    worker_calls: list[_WorkerCall],
    origin: str,
) -> None:
    """A script name the skill does not list gets Agno's not-found answer, and no command runs."""
    skill = _write_skill(tmp_path, origin)
    setup = _create_worker_agent(tmp_path, skill, "docker")

    result = json.loads(await _get_skill_script(setup.agent, execute=True, script_path="../SKILL.md"))

    assert "not found" in result["error"]
    assert result["available_scripts"] == ["check.sh"]
    assert worker_calls == []


@pytest.mark.parametrize("origin", ["workspace", "plugin"])
@pytest.mark.asyncio
async def test_skill_scripts_run_on_this_host_when_the_agents_shell_does(tmp_path: Path, origin: str) -> None:
    """With a local shell, a script runs on this host as the agent's shell commands do, in the agent's workspace."""
    skill = _write_skill(tmp_path, origin)
    setup = _create_agent(tmp_path, skill, tools=["shell"])

    result = await _get_skill_script(setup.agent, execute=True)

    workspace = agent_workspace_root_path(_storage(tmp_path), "general")
    assert result.rstrip("\n") == f"[cwd: {workspace}]\nchecked book.xlsx in {workspace}"
    assert skill.ran_here


@pytest.mark.asyncio
async def test_a_large_script_output_is_saved_once(tmp_path: Path) -> None:
    """The shell saves a large output to a workspace file, and the skill function returns that file's receipt as is."""
    skill = _write_skill(tmp_path, "plugin")
    skill.script.write_text("#!/bin/sh\nfor line in $(seq 1 100); do printf '%050d\\n' \"$line\"; done\n")
    setup = _create_agent(
        tmp_path,
        skill,
        tools=["shell"],
        config_settings={"defaults": {"tool_output_auto_save_threshold_bytes": 1024}},
    )

    result = await _get_skill_script(setup.agent, execute=True)

    workspace = agent_workspace_root_path(_storage(tmp_path), "general")
    [saved] = (workspace / "mindroom_tool_outputs").iterdir()
    assert saved.name.startswith("run_shell_command-")
    assert f"mindroom_tool_outputs/{saved.name}" in result
    assert saved.read_text().rstrip().endswith(f"{100:050d}")


@pytest.mark.asyncio
async def test_running_a_workspace_skill_script_counts_as_using_the_skill(tmp_path: Path) -> None:
    """A run, like a read, keeps a learned workspace skill from being archived as unused."""
    skill = _write_skill(tmp_path, "workspace")
    setup = _create_agent(tmp_path, skill, tools=["shell"])

    await _get_skill_script(setup.agent, execute=True)
    assert await wait_for_background_tasks(SETTLE_TIMEOUT_SECONDS)

    usage = json.loads((skill.script.parents[2] / ".usage.json").read_text())
    assert "last_used_at" in usage["xl_check"]


@pytest.mark.parametrize("origin", ["workspace", "plugin"])
@pytest.mark.asyncio
async def test_skill_scripts_need_the_agents_shell(tmp_path: Path, origin: str) -> None:
    """An agent without a shell tool can read a skill script but not run it."""
    skill = _write_skill(tmp_path, origin)
    setup = _create_agent(tmp_path, skill, tools=["calculator"])

    refused = json.loads(await _get_skill_script(setup.agent, execute=True))
    read = json.loads(await _get_skill_script(setup.agent, execute=False))

    assert "shell tool" in refused["error"]
    assert "execute=false" in refused["error"]
    assert not skill.ran_here
    assert read["content"] == _SCRIPT


@pytest.mark.parametrize("loaded", [False, True], ids=["deferred", "loaded"])
@pytest.mark.asyncio
async def test_a_deferred_shell_runs_skill_scripts_once_the_agent_loads_it(tmp_path: Path, loaded: bool) -> None:
    """An agent has a shell for its scripts exactly when it has one for its own commands."""
    skill = _write_skill(tmp_path, "plugin")
    setup = _create_agent(
        tmp_path,
        skill,
        tools=[{"shell": {"defer": True}}],
        loaded_tools=("shell",) if loaded else (),
    )

    result = await _get_skill_script(setup.agent, execute=True)

    workspace = agent_workspace_root_path(_storage(tmp_path), "general")
    assert (result.rstrip("\n") == f"[cwd: {workspace}]\nchecked book.xlsx in {workspace}") is loaded
    assert skill.ran_here is loaded


@pytest.mark.parametrize("jobs", [False, True], ids=["paused-run", "job-approval"])
@pytest.mark.asyncio
async def test_skill_scripts_are_refused_while_the_agents_shell_needs_approval(tmp_path: Path, jobs: bool) -> None:
    """A skill function cannot wait for approval, so a script whose shell commands need it is left to the shell."""
    skill = _write_skill(tmp_path, "plugin")
    setup = _create_agent(
        tmp_path,
        skill,
        tools=["shell"],
        config_settings={
            "tool_approval": {"rules": [{"match": "run_shell_command", "action": "require_approval"}]},
            "background_tool_jobs": {"enabled": jobs},
        },
    )

    refused = json.loads(await _get_skill_script(setup.agent, execute=True))

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
    setup = _create_agent(tmp_path, skill, tools=["shell"], hook_registry=HookRegistry.from_plugins([plugin]))

    result = await _get_skill_script(setup.agent, execute=True)

    assert [name for name, _ in before_calls] == ["get_skill_script", "run_shell_command"]
    assert before_calls[1][1] == {"args": shlex.join(["/bin/sh", str(skill.script), "book.xlsx"]), "timeout": 30}
    assert after_calls == [("run_shell_command", True), ("get_skill_script", False)]
    assert "[TOOL CALL DECLINED]" in result
    assert "no shell commands today" in result
    assert not skill.ran_here
