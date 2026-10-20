"""Tests for moving workers and credentials keyed by an agent's name, and the worker moves shared with account renames."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime
from functools import partial
from typing import TYPE_CHECKING, cast

import pytest

from mindroom.agent_rename import workers as worker_moves
from mindroom.agent_rename.account_files import account_file_steps
from mindroom.agent_rename.entity_files import entity_credential_steps, rewrite_worker_key_agent
from mindroom.agent_rename.steps import (
    DropPath,
    FileRewrite,
    FinalAction,
    MovePath,
    RenamePlan,
    SqliteEdit,
    apply_plan,
)
from mindroom.agent_rename.workers import DestinationExistsError, worker_move_steps
from mindroom.config.main import Config
from mindroom.credentials.manager import (
    PRIMARY_RUNTIME_SCOPED_CREDENTIALS_DIRNAME,
    get_runtime_credentials_manager,
    scoped_credentials_dir_part,
)
from mindroom.oauth.credential_lifecycle import OAuthCredentialContext
from mindroom.oauth.credential_store import oauth_credential_reader, oauth_credential_transaction
from mindroom.tool_system.worker_routing import (
    ToolExecutionIdentity,
    WorkerScope,
    resolve_unscoped_worker_key,
    resolve_worker_key,
    resolve_worker_target,
    worker_dir_name,
)
from mindroom.workers.backends.docker import WORKER_CONTROL_DIRNAME
from mindroom.workers.backends.docker_projection import PROJECTED_CONFIGS_DIRNAME
from mindroom.workers.backends.local import local_worker_state_paths_for_root
from tests.agent_rename_helpers import tree_snapshot
from tests.conftest import test_runtime_paths

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from mindroom.agent_rename.steps import Step
    from mindroom.constants import RuntimePaths
    from mindroom.oauth.providers import OAuthProvider

OLD = "code"
NEW = "coder"
# Its name extends the renamed agent's, so a substring match would wrongly touch it.
PREFIX = "code_team"
ALICE = "@alice:x"
NOW = datetime(2026, 10, 20, 9, 30, 0, tzinfo=UTC)

rewrite = partial(rewrite_worker_key_agent, old_name=OLD, new_name=NEW)


def _identity(agent: str, requester_id: str = ALICE) -> ToolExecutionIdentity:
    return ToolExecutionIdentity(
        channel="matrix",
        agent_name=agent,
        requester_id=requester_id,
        room_id=None,
        thread_id=None,
        resolved_thread_id=None,
        session_id=None,
    )


def _key(scope: WorkerScope, agent: str, requester_id: str = ALICE) -> str:
    key = resolve_worker_key(scope, _identity(agent, requester_id))
    assert key is not None
    return key


def _docker(runtime_paths: RuntimePaths) -> RuntimePaths:
    return replace(runtime_paths, process_env={**runtime_paths.process_env, "MINDROOM_WORKER_BACKEND": "docker"})


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _oauth_store(path: Path, worker_key: str) -> None:
    """Create a minimal OAuth credential store bound to ``worker_key``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            "CREATE TABLE oauth_credential_state "
            "(singleton INTEGER PRIMARY KEY, provider_id TEXT, worker_key TEXT, credential_present INTEGER)",
        )
        connection.execute("INSERT INTO oauth_credential_state VALUES (1, 'demo', ?, 1)", (worker_key,))
        connection.commit()


def _stored_worker_key(path: Path) -> str:
    with closing(sqlite3.connect(path)) as connection:
        return connection.execute("SELECT worker_key FROM oauth_credential_state").fetchone()[0]


def _worker_root(runtime_paths: RuntimePaths, worker_key: str) -> Path:
    return runtime_paths.storage_root / "workers" / worker_dir_name(worker_key)


def _write_worker(runtime_paths: RuntimePaths, worker_key: str) -> Path:
    """Write a worker root with metadata, a workspace file and a venv whose script names the root."""
    root = _worker_root(runtime_paths, worker_key)
    _write_json(
        root / "metadata" / "worker.json",
        {"status": "idle", "worker_id": worker_dir_name(worker_key), "worker_key": worker_key},
    )
    (root / "workspace").mkdir()
    (root / "workspace" / "notes.md").write_text(f"notes of {worker_key}", encoding="utf-8")
    (root / "venv" / "bin").mkdir(parents=True)
    (root / "venv" / "bin" / "pip").write_text(f"#!{root}/venv/bin/python\n", encoding="utf-8")
    return root


def _worker_steps(runtime_paths: RuntimePaths) -> tuple[list[Step], list[str]]:
    return worker_move_steps(runtime_paths, config=Config(), rewrite=rewrite, discard_new_state=False)


def _apply(runtime_paths: RuntimePaths, steps: list[Step], reports: list[str] | None = None) -> Path:
    plan = RenamePlan(storage_root=runtime_paths.storage_root, steps=tuple(steps), reports=tuple(reports or ()))
    result = apply_plan(plan, kind="rename", now=NOW)
    assert result.warnings == ()
    return result.backup_dir


class _Provider:
    id = "demo_provider"
    credential_service = "demo_oauth"
    requester_scoped_credentials = True


def _oauth_context(runtime_paths: RuntimePaths, scope: WorkerScope, agent: str) -> OAuthCredentialContext:
    return OAuthCredentialContext(
        provider=cast("OAuthProvider", _Provider()),
        runtime_paths=runtime_paths,
        credentials_manager=get_runtime_credentials_manager(runtime_paths),
        worker_target=resolve_worker_target(scope, agent, _identity(agent)),
    )


async def _publish(runtime_paths: RuntimePaths, scope: WorkerScope, agent: str) -> None:
    async with oauth_credential_transaction(_oauth_context(runtime_paths, scope, agent)) as transaction:
        await transaction.publish({"token": f"{scope}-{agent}"}, advance_connection_generation=True)
        await transaction.commit()


async def _read(runtime_paths: RuntimePaths, scope: WorkerScope, agent: str) -> object:
    async with oauth_credential_reader(_oauth_context(runtime_paths, scope, agent)) as reader:
        snapshot = await reader.snapshot()
    return snapshot.credentials


class _FakeDockerManager:
    backend_name = "docker"

    def __init__(self, on_retire: Callable[[], None]) -> None:
        self.on_retire = on_retire
        self.retired: list[str] = []

    def retire_worker(self, worker_key: str) -> None:
        self.on_retire()
        self.retired.append(worker_key)


class _FakeLease:
    def __init__(self, manager: _FakeDockerManager) -> None:
        self.manager = manager
        self.released = False

    def __enter__(self) -> _FakeDockerManager:
        return self.manager

    def __exit__(self, *_exc: object) -> bool:
        self.released = True
        return False


def test_rewrite_worker_key_agent_scopes() -> None:
    """The agent component of shared, unscoped and requester-agent keys changes; every other key gives None."""
    for scope in ("shared", "user_agent"):
        assert rewrite(_key(scope, OLD)) == _key(scope, NEW)
    assert rewrite(resolve_unscoped_worker_key(OLD)) == resolve_unscoped_worker_key(NEW)
    assert rewrite("v1:acme:shared:code") == "v1:acme:shared:coder"
    assert rewrite("v1:acme:unscoped:code") == "v1:acme:unscoped:coder"
    # Requesters hold colons, and OAuth stores from before the `~` prefix keep the raw ID; the agent is last either way.
    assert rewrite("v1:default:user_agent:@alice:x:code") == "v1:default:user_agent:@alice:x:coder"
    assert rewrite("v1:default:user_agent:~@alice:x:8448:code") == "v1:default:user_agent:~@alice:x:8448:coder"
    # A `user` key names no agent, even when the requester's server is called like it.
    assert rewrite(_key("user", OLD)) is None
    assert rewrite("v1:default:user:~@alice:code") is None
    for other in ("v1:default:shared:other", "v2:default:shared:code", "v1:default:shared:x:code", "code", ""):
        assert rewrite(other) is None


def test_rewrite_ignores_prefix_agent() -> None:
    """Renaming `code` leaves the keys of `code_team` alone, and renaming `code_team` leaves those of `code`."""
    for scope in ("shared", "user_agent"):
        assert rewrite(_key(scope, PREFIX)) is None
        assert rewrite_worker_key_agent(_key(scope, OLD), old_name=PREFIX, new_name="crew") is None
    assert rewrite(resolve_unscoped_worker_key(PREFIX)) is None
    # A requester whose localpart is the old name does not make the key the old agent's.
    assert rewrite(_key("user_agent", PREFIX, requester_id="@code:x")) is None


def test_shared_worker_moved_and_venv_dropped(tmp_path: Path) -> None:
    """Every worker of the old agent moves to its new key's root; the prefix agent's workers stay exactly as they were.

    The shared and unscoped workers are known from their metadata, the requester-agent worker only from the
    requester's OAuth store. The moved venv goes to the backup, and MindRoom rebuilds it on the worker's next use.
    """
    runtime_paths = test_runtime_paths(tmp_path)
    shared, unscoped, requester = _key("shared", OLD), resolve_unscoped_worker_key(OLD), _key("user_agent", OLD)
    shared_root = _write_worker(runtime_paths, shared)
    _oauth_store(shared_root / "credentials" / "demo_oauth_credentials.sqlite3", shared)
    _write_worker(runtime_paths, unscoped)
    (_worker_root(runtime_paths, requester) / "credentials").mkdir(parents=True)
    oauth = runtime_paths.storage_root / PRIMARY_RUNTIME_SCOPED_CREDENTIALS_DIRNAME
    requester_oauth = oauth / scoped_credentials_dir_part(ALICE) / scoped_credentials_dir_part(OLD)
    _oauth_store(requester_oauth / "demo_oauth_credentials.sqlite3", requester)
    prefix_roots = [
        _write_worker(runtime_paths, key)
        for key in (_key("shared", PREFIX), _key("user_agent", PREFIX), resolve_unscoped_worker_key(PREFIX))
    ]
    prefix_before = {root: tree_snapshot(root) for root in prefix_roots}

    steps, reports = _worker_steps(runtime_paths)

    assert reports == []
    assert not [step for step in steps if isinstance(step, FinalAction)]
    backup = _apply(runtime_paths, steps)

    for old_key in (shared, unscoped, requester):
        assert not _worker_root(runtime_paths, old_key).exists()
    for old_key in (shared, unscoped):
        new_key = rewrite(old_key)
        assert new_key is not None
        new_root = _worker_root(runtime_paths, new_key)
        assert json.loads((new_root / "metadata" / "worker.json").read_text()) == {
            "status": "idle",
            "worker_id": worker_dir_name(new_key),
            "worker_key": new_key,
        }
        assert (new_root / "workspace" / "notes.md").read_text() == f"notes of {old_key}"
        assert not local_worker_state_paths_for_root(new_root).venv_dir.exists()
    assert (_worker_root(runtime_paths, _key("user_agent", NEW)) / "credentials").is_dir()
    new_shared = _worker_root(runtime_paths, _key("shared", NEW))
    assert _stored_worker_key(new_shared / "credentials" / "demo_oauth_credentials.sqlite3") == _key("shared", NEW)
    saved_pips = sorted(path.read_text() for path in backup.rglob("venv/bin/pip"))
    assert saved_pips == sorted(f"#!{_worker_root(runtime_paths, key)}/venv/bin/python\n" for key in (shared, unscoped))
    assert {root: tree_snapshot(root) for root in prefix_roots} == prefix_before


def test_docker_worker_retired_after_move(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On Docker the old container is retired only after apply moved its state, and the moved root has no venv.

    The Docker records of the old key are dropped and the agent's mount point inside the root follows the new name.
    The prefix agent's Docker worker is neither moved nor retired.
    """
    runtime_paths = _docker(test_runtime_paths(tmp_path))
    workers = runtime_paths.storage_root / "workers"
    old_key, new_key, prefix_key = _key("shared", OLD), _key("shared", NEW), _key("shared", PREFIX)
    old_root, new_root = _worker_root(runtime_paths, old_key), _worker_root(runtime_paths, new_key)
    for key in (old_key, prefix_key):
        root = _worker_root(runtime_paths, key)
        for name in ("credentials", ".shared_credentials", "venv/bin"):
            (root / name).mkdir(parents=True)
        _write_json(workers / WORKER_CONTROL_DIRNAME / root.name / "metadata" / "worker.json", {"worker_key": key})
        (workers / PROJECTED_CONFIGS_DIRNAME / root.name).mkdir(parents=True)
    (old_root / "agents" / OLD).mkdir(parents=True)
    (_worker_root(runtime_paths, prefix_key) / "agents" / PREFIX).mkdir(parents=True)
    prefix_paths = [
        _worker_root(runtime_paths, prefix_key),
        workers / WORKER_CONTROL_DIRNAME / worker_dir_name(prefix_key),
        workers / PROJECTED_CONFIGS_DIRNAME / worker_dir_name(prefix_key),
    ]
    prefix_before = {path: tree_snapshot(path) for path in prefix_paths}

    def state_already_moved() -> None:
        assert not old_root.exists()
        assert (new_root / "credentials").is_dir()

    manager = _FakeDockerManager(state_already_moved)
    leases: list[_FakeLease] = []

    def lease(paths: RuntimePaths, *, runtime_config: Config | None) -> _FakeLease:
        assert paths == runtime_paths
        assert runtime_config is not None
        leases.append(_FakeLease(manager))
        return leases[-1]

    monkeypatch.setattr(worker_moves, "lease_configured_primary_worker_manager", lease)
    before = tree_snapshot(tmp_path)

    steps, _reports = _worker_steps(runtime_paths)

    assert tree_snapshot(tmp_path) == before
    assert leases == []
    assert [type(step) for step in steps].count(FinalAction) == 1
    _apply(runtime_paths, steps)

    assert manager.retired == [old_key]
    assert all(lease.released for lease in leases)
    assert not (workers / WORKER_CONTROL_DIRNAME / old_root.name).exists()
    assert not (workers / PROJECTED_CONFIGS_DIRNAME / old_root.name).exists()
    assert not (new_root / "venv").exists()
    assert (new_root / "agents" / NEW).is_dir()
    assert not (new_root / "agents" / OLD).exists()
    assert {path: tree_snapshot(path) for path in prefix_paths} == prefix_before


def test_existing_destination_refuses(tmp_path: Path) -> None:
    """A worker root or OAuth directory that already exists under the new name refuses before any write."""
    runtime_paths = test_runtime_paths(tmp_path)
    _write_worker(runtime_paths, _key("shared", OLD))
    taken = _worker_root(runtime_paths, _key("shared", NEW))
    taken.mkdir(parents=True)
    agents_oauth = runtime_paths.storage_root / PRIMARY_RUNTIME_SCOPED_CREDENTIALS_DIRNAME / "_agents"
    (agents_oauth / scoped_credentials_dir_part(OLD)).mkdir(parents=True)
    taken_oauth = agents_oauth / scoped_credentials_dir_part(NEW)
    taken_oauth.mkdir()

    with pytest.raises(DestinationExistsError, match=f"{taken} already exists"):
        _worker_steps(runtime_paths)
    with pytest.raises(DestinationExistsError, match=f"{taken_oauth} already exists"):
        entity_credential_steps(runtime_paths, old_name=OLD, new_name=NEW)


@pytest.mark.asyncio
async def test_agent_oauth_dirs_moved_and_bindings_validate(tmp_path: Path) -> None:
    """The agent's own and requester-agent OAuth stores move to the new name and open as the new agent.

    The prefix agent's stores and the requester-wide store stay byte for byte.
    """
    runtime_paths = test_runtime_paths(tmp_path)
    for agent in (OLD, PREFIX):
        for scope in ("shared", "user_agent"):
            await _publish(runtime_paths, scope, agent)
    await _publish(runtime_paths, "user", OLD)
    oauth = runtime_paths.storage_root / PRIMARY_RUNTIME_SCOPED_CREDENTIALS_DIRNAME
    scopes = (oauth / "_agents", oauth / scoped_credentials_dir_part(ALICE))
    untouched = [scope / scoped_credentials_dir_part(PREFIX) for scope in scopes]
    untouched.append(oauth / scoped_credentials_dir_part(ALICE) / scoped_credentials_dir_part("_shared"))
    untouched_before = {path: tree_snapshot(path) for path in untouched}

    steps = entity_credential_steps(runtime_paths, old_name=OLD, new_name=NEW)

    assert [type(step) for step in steps] == [MovePath, SqliteEdit] * 2
    _apply(runtime_paths, steps)

    for scope in scopes:
        assert not (scope / scoped_credentials_dir_part(OLD)).exists()
        assert (scope / scoped_credentials_dir_part(NEW)).is_dir()
    for scope in ("shared", "user_agent"):
        assert await _read(runtime_paths, scope, NEW) == {"token": f"{scope}-{OLD}"}
        assert await _read(runtime_paths, scope, PREFIX) == {"token": f"{scope}-{PREFIX}"}
    assert await _read(runtime_paths, "user", NEW) == {"token": f"user-{OLD}"}
    assert {path: tree_snapshot(path) for path in untouched} == untouched_before


def test_account_rename_behavior_unchanged(tmp_path: Path) -> None:
    """An account rename plans exactly the steps it planned before the worker moves were shared with agent renames.

    The agent-keyed shared worker and OAuth store of the same agent are left alone.
    The expected order is the one ``account_file_steps`` produced before the extraction;
    tests/test_agent_rename_account_files.py and tests/test_cli_agents_rename_account.py cover its behavior.
    """
    runtime_paths = _docker(test_runtime_paths(tmp_path))
    storage = runtime_paths.storage_root
    workers = storage / "workers"
    old_user, new_user = "@old:x", "@new:x"
    user_old, user_new = _key("user", OLD, old_user), _key("user", OLD, new_user)
    agent_old, agent_new = _key("user_agent", OLD, old_user), _key("user_agent", OLD, new_user)
    oauth = storage / PRIMARY_RUNTIME_SCOPED_CREDENTIALS_DIRNAME
    store = "demo_oauth_credentials.sqlite3"
    _oauth_store(oauth / scoped_credentials_dir_part(old_user) / scoped_credentials_dir_part(OLD) / store, agent_old)
    _oauth_store(oauth / "_agents" / scoped_credentials_dir_part(OLD) / store, _key("shared", OLD))
    user_root = _write_worker(runtime_paths, user_old)
    _oauth_store(user_root / "credentials" / store, user_old)
    _write_json(
        workers / WORKER_CONTROL_DIRNAME / user_root.name / "metadata" / "worker.json",
        {"worker_key": user_old},
    )
    (workers / PROJECTED_CONFIGS_DIRNAME / user_root.name).mkdir(parents=True)
    (_worker_root(runtime_paths, agent_old) / "credentials").mkdir(parents=True)
    _write_worker(runtime_paths, _key("shared", OLD))
    unknown = workers / "v1_default_user_agent_@old_x_legacy-0123456789abcdef"
    unknown.mkdir()

    steps, reports = account_file_steps(
        runtime_paths,
        agent="harry",
        old_user_id=old_user,
        new_user_id=new_user,
        config=Config(),
        discard_new_identity_state=False,
    )

    def shape(step: Step) -> tuple[type, Path | None, Path | None]:
        if isinstance(step, MovePath):
            return MovePath, step.source, step.destination
        if isinstance(step, SqliteEdit):
            return SqliteEdit, step.database, None
        if isinstance(step, FileRewrite | DropPath):
            return type(step), step.path, None
        return type(step), None, None

    new_oauth = oauth / scoped_credentials_dir_part(new_user)
    new_user_root = _worker_root(runtime_paths, user_new)
    assert [shape(step) for step in steps] == [
        (MovePath, oauth / scoped_credentials_dir_part(old_user), new_oauth),
        (SqliteEdit, new_oauth / scoped_credentials_dir_part(OLD) / store, None),
        (MovePath, user_root, new_user_root),
        (DropPath, new_user_root / "venv", None),
        (FileRewrite, new_user_root / "metadata" / "worker.json", None),
        (SqliteEdit, new_user_root / "credentials" / store, None),
        (DropPath, workers / WORKER_CONTROL_DIRNAME / user_root.name, None),
        (DropPath, workers / PROJECTED_CONFIGS_DIRNAME / user_root.name, None),
        (FinalAction, None, None),
        (MovePath, _worker_root(runtime_paths, agent_old), _worker_root(runtime_paths, agent_new)),
        (FinalAction, None, None),
    ]
    assert reports == [
        f"{unknown} looks like a worker of {old_user}, but its worker key is unknown, so it stays in place.",
        f"Records of past tool jobs and scheduled runs outside the event journal keep {old_user} as history.",
    ]
