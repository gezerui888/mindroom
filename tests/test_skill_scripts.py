"""get_skill_script gives the agent a command for its own shell tool, which runs it wherever that shell runs."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
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
from mindroom.constants import resolve_runtime_paths
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


async def _call(agent: Agent, function_name: str, arguments: dict[str, object]) -> str:
    """Call one of the agent's functions the way a model's tool call runs it, prepared for the run with its hooks."""
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
    [function] = [tool for tool in tools if isinstance(tool, Function) and tool.name == function_name]
    execution = await FunctionCall(function=function, arguments=arguments, call_id="call-1").aexecute()
    assert execution.status == "success", execution.error
    return str(execution.result)


async def _get_skill_script(agent: Agent, *, execute: bool, script_path: str = "check.sh") -> dict[str, object]:
    arguments = {"skill_name": "xl_check", "script_path": script_path, "execute": execute, "args": ["book.xlsx"]}
    return json.loads(await _call(agent, "get_skill_script", arguments))


async def _run_script(agent: Agent) -> tuple[list[str], str]:
    """Ask for the script's command and run it with the agent's own shell tool, as the reply tells the model to."""
    reply = await _get_skill_script(agent, execute=True)
    assert "run_shell_command" in str(reply["instructions"])
    command = str(reply["command"])
    output = await _call(agent, "run_shell_command", {"args": command, "timeout": reply["timeout"]})
    return shlex.split(command), output


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
async def test_skill_scripts_get_a_command_for_the_worker_that_mounts_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_calls: list[_WorkerCall],
    backend_name: str,
    origin: str,
) -> None:
    """The command names the script where the agent's worker mounts it, and runs there, never on this host.

    A workspace skill is in the writable workspace mount, and a configured skill in a read-only mount of its own
    directory, as the real backend starts the worker the agent's shell command is routed to.
    """
    skill = _write_skill(tmp_path, origin)
    setup = _create_worker_agent(tmp_path, skill, backend_name)

    (interpreter, script, *args), output = await _run_script(setup.agent)

    [call] = worker_calls
    assert (call.tool_name, call.function_name, call.kwargs["timeout"]) == ("shell", "run_shell_command", 30)
    assert (interpreter, args) == ("/bin/sh", ["book.xlsx"])
    assert output == "ran in the worker"
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

    refused = await _get_skill_script(setup.agent, execute=True)
    read = await _get_skill_script(setup.agent, execute=False)

    assert "cannot see" in str(refused["error"])
    assert "execute=false" in str(refused["error"])
    assert "command" not in refused
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
async def test_skill_scripts_get_a_command_where_every_worker_sees_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_calls: list[_WorkerCall],
    backend_name: str,
    origin: str,
) -> None:
    """Worker images ship the bundled skills, and the shared runner sees workspaces, where this host has them."""
    skill = _write_skill(tmp_path, origin, monkeypatch)
    setup = _create_worker_agent(tmp_path, skill, backend_name)

    reply = await _get_skill_script(setup.agent, execute=True)

    assert reply["command"] == shlex.join(["/bin/sh", str(skill.script), "book.xlsx"])
    assert worker_calls == []
    assert not skill.ran_here


@pytest.mark.parametrize("origin", ["workspace", "plugin"])
@pytest.mark.asyncio
async def test_an_unlisted_skill_script_gets_no_command(
    tmp_path: Path,
    worker_calls: list[_WorkerCall],
    origin: str,
) -> None:
    """A script name the skill does not list gets Agno's not-found answer instead of a command."""
    skill = _write_skill(tmp_path, origin)
    setup = _create_worker_agent(tmp_path, skill, "docker")

    result = await _get_skill_script(setup.agent, execute=True, script_path="../SKILL.md")

    assert "not found" in str(result["error"])
    assert result["available_scripts"] == ["check.sh"]
    assert worker_calls == []


@pytest.mark.parametrize("origin", ["workspace", "plugin"])
@pytest.mark.asyncio
async def test_a_local_shell_runs_the_command_on_this_host(tmp_path: Path, origin: str) -> None:
    """With a local shell, the command runs on this host as the agent's shell commands do, in the agent's workspace."""
    skill = _write_skill(tmp_path, origin)
    setup = _create_agent(tmp_path, skill, tools=["shell"])

    command, output = await _run_script(setup.agent)

    workspace = agent_workspace_root_path(_storage(tmp_path), "general")
    assert command == ["/bin/sh", str(skill.script), "book.xlsx"]
    assert output.rstrip("\n") == f"[cwd: {workspace}]\nchecked book.xlsx in {workspace}"
    assert skill.ran_here


@pytest.mark.asyncio
async def test_asking_for_a_workspace_skill_scripts_command_counts_as_using_the_skill(tmp_path: Path) -> None:
    """A command, like a read, keeps a learned workspace skill from being archived as unused."""
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

    refused = await _get_skill_script(setup.agent, execute=True)
    read = await _get_skill_script(setup.agent, execute=False)

    assert "shell tool" in str(refused["error"])
    assert "execute=false" in str(refused["error"])
    assert "command" not in refused
    assert not skill.ran_here
    assert read["content"] == _SCRIPT


@pytest.mark.parametrize("loaded", [False, True], ids=["deferred", "loaded"])
@pytest.mark.asyncio
async def test_a_deferred_shell_gets_the_command_once_the_agent_loads_it(tmp_path: Path, loaded: bool) -> None:
    """An agent gets commands for its scripts exactly when it has a shell loaded to run its own commands."""
    skill = _write_skill(tmp_path, "plugin")
    setup = _create_agent(
        tmp_path,
        skill,
        tools=[{"shell": {"defer": True}}],
        loaded_tools=("shell",) if loaded else (),
    )

    reply = await _get_skill_script(setup.agent, execute=True)

    assert (reply.get("command") == shlex.join(["/bin/sh", str(skill.script), "book.xlsx"])) is loaded
    assert ("not have loaded" in str(reply.get("error"))) is not loaded
    assert not skill.ran_here


@pytest.mark.asyncio
async def test_a_shell_that_needs_approval_still_gets_the_command(tmp_path: Path) -> None:
    """Approval applies when the agent runs the command with its shell tool, so asking for it needs none."""
    skill = _write_skill(tmp_path, "plugin")
    setup = _create_agent(
        tmp_path,
        skill,
        tools=["shell"],
        config_settings={"tool_approval": {"rules": [{"match": "run_shell_command", "action": "require_approval"}]}},
    )

    reply = await _get_skill_script(setup.agent, execute=True)

    assert reply["command"] == shlex.join(["/bin/sh", str(skill.script), "book.xlsx"])
    assert not skill.ran_here
