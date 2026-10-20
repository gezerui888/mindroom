"""Tests for `mindroom agents rename-account`, run through the CLI against a synthetic stopped install."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import closing
from functools import partial
from typing import TYPE_CHECKING

import httpx
import pytest
from typer.testing import CliRunner

from mindroom.agent_rename import account_files
from mindroom.agent_rename import steps as steps_module
from mindroom.agent_rename.guard import stopped_runtime
from mindroom.agent_rename.steps import RenameUndoError
from mindroom.cli.config import activate_cli_runtime
from mindroom.cli.main import app
from mindroom.constants import matrix_state_file, tracking_dir
from mindroom.credentials.manager import PRIMARY_RUNTIME_SCOPED_CREDENTIALS_DIRNAME, scoped_credentials_dir_part
from mindroom.event_journal import EventJournalStore, EventKind, encode_thread_id
from mindroom.event_journal.opening import event_journal_sqlite_path
from mindroom.file_locks import release_file_lock, try_exclusive_file_lock
from mindroom.matrix.state import MatrixState
from mindroom.tool_system.worker_routing import ToolExecutionIdentity, WorkerScope, resolve_worker_key, worker_dir_name
from mindroom.workers import runtime as workers_runtime
from mindroom.yaml_io import safe_load
from tests.agent_rename_helpers import tree_snapshot
from tests.conftest import TEST_ACCESS_TOKEN

if TYPE_CHECKING:
    from pathlib import Path

    from mindroom.agent_rename.steps import RenamePlan
    from mindroom.config.main import Config
    from mindroom.constants import RuntimePaths

AGENT = "harry"
OLD = "@old:x"
NEW = "@new:x"
OLD_PRINCIPAL = "harry@@old:x"
NEW_PRINCIPAL = "harry@@new:x"
ROUTER = "router@@router:x"
PASSWORD = "stored-secret"  # noqa: S105
CONFIG = "agents:\n  harry:\n    display_name: Harry\nbot_accounts:\n  - '@old:x'   # harry himself\n"

runner = CliRunner()


class _MatrixServer:
    """A fake homeserver for the login check that records every request."""

    def __init__(self, login_status: int = 200) -> None:
        self.login_status = login_status
        self.requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/_matrix/client/v3/login":
            if self.login_status != 200:
                return httpx.Response(self.login_status, json={"errcode": "M_FORBIDDEN", "error": "Invalid password"})
            return httpx.Response(200, json={"user_id": NEW, "access_token": "check-token", "device_id": "CHECK"})
        return httpx.Response(200, json={})


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> _MatrixServer:
    """Answer the login check from a fake homeserver instead of the network."""
    fake = _MatrixServer()
    monkeypatch.setattr(httpx, "Client", partial(httpx.Client, transport=httpx.MockTransport(fake.handle)))
    return fake


def _identity(requester_id: str) -> ToolExecutionIdentity:
    return ToolExecutionIdentity(
        channel="matrix",
        agent_name="code",
        requester_id=requester_id,
        room_id=None,
        thread_id=None,
        resolved_thread_id=None,
        session_id=None,
    )


def _key(scope: WorkerScope, requester_id: str) -> str:
    key = resolve_worker_key(scope, _identity(requester_id), agent_name="code")
    assert key is not None
    return key


def _insert(connection: sqlite3.Connection, table: str, **values: object) -> None:
    columns = ", ".join(values)
    marks = ", ".join("?" for _ in values)
    connection.execute(f"INSERT INTO {table} ({columns}) VALUES ({marks})", tuple(values.values()))  # noqa: S608


def _event(connection: sqlite3.Connection, principal: str, *, sender: str) -> None:
    _insert(
        connection,
        "journal_events",
        principal_id=principal,
        event_id="$from-harry",
        room_id="!room:x",
        thread_id=encode_thread_id(None),
        kind=EventKind.MESSAGE.value,
        sender=sender,
        origin_server_ts=10,
        source_json=json.dumps({"sender": sender}),
        membership_epoch=0,
        state="settled",
    )


def _journal(storage_root: Path, *, new_principal_rows: bool = False) -> Path:
    """Build a journal with the real schema: harry's event seen by harry and the router, and harry's sync consumer."""
    database = event_journal_sqlite_path(storage_root)
    asyncio.run(EventJournalStore.open_sqlite(database).close())
    with closing(sqlite3.connect(database)) as connection, connection:
        _event(connection, OLD_PRINCIPAL, sender=OLD)
        _event(connection, ROUTER, sender=OLD)
        _insert(
            connection,
            "matrix_sync_consumers",
            principal_id=OLD_PRINCIPAL,
            consumer_generation="generation-1",
            stream_id="stream-1",
            next_sequence=5,
        )
        if new_principal_rows:
            # MindRoom ran after the server rename and bound a sync stream under the new principal.
            _insert(
                connection,
                "matrix_sync_consumers",
                principal_id=NEW_PRINCIPAL,
                consumer_generation="generation-2",
                stream_id="stream-2",
                next_sequence=1,
            )
    return database


def _oauth_store(path: Path, worker_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "CREATE TABLE oauth_credential_state "
            "(singleton INTEGER PRIMARY KEY, provider_id TEXT, worker_key TEXT, credential_present INTEGER)",
        )
        connection.execute("INSERT INTO oauth_credential_state VALUES (1, 'demo', ?, 1)", (worker_key,))


def _stored_worker_key(path: Path) -> str:
    with closing(sqlite3.connect(path)) as connection:
        return connection.execute("SELECT worker_key FROM oauth_credential_state").fetchone()[0]


def _oauth_path(storage_root: Path, requester_id: str) -> Path:
    oauth_root = storage_root / PRIMARY_RUNTIME_SCOPED_CREDENTIALS_DIRNAME
    return (
        oauth_root
        / scoped_credentials_dir_part(requester_id)
        / scoped_credentials_dir_part("code")
        / "demo_oauth_credentials.sqlite3"
    )


def _install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, new_principal_rows: bool = False) -> RuntimePaths:
    """Build a stopped install whose server already renamed harry from @old:x to @new:x."""
    for name in ("MINDROOM_WORKER_BACKEND", "MATRIX_SERVER_NAME", "MATRIX_SSL_VERIFY", "MINDROOM_CONFIG_TEMPLATE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MATRIX_HOMESERVER", "http://localhost:8008")
    monkeypatch.setenv("MINDROOM_NAMESPACE", "")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    runtime_paths = activate_cli_runtime(config_path, storage_path=tmp_path / "mindroom_data")
    storage = runtime_paths.storage_root

    state = MatrixState()
    state.add_account(
        f"agent_{AGENT}",
        "old",
        PASSWORD,
        domain="x",
        device_id="OLDDEVICE",
        access_token=TEST_ACCESS_TOKEN,
    )
    state.add_account("agent_router", "router", "router-secret", domain="x")
    state.save(runtime_paths)

    _journal(storage, new_principal_rows=new_principal_rows)
    _oauth_store(_oauth_path(storage, OLD), _key("user_agent", OLD))
    worker = storage / "workers" / worker_dir_name(_key("user", OLD))
    (worker / "metadata").mkdir(parents=True)
    (worker / "metadata" / "worker.json").write_text(json.dumps({"worker_key": _key("user", OLD)}), encoding="utf-8")
    (worker / "workspace").mkdir()
    (worker / "workspace" / "notes.md").write_text("notes", encoding="utf-8")
    # A MindRoom that ran here left its lock files behind.
    with stopped_runtime(runtime_paths):
        pass
    return runtime_paths


def _invoke(runtime_paths: RuntimePaths, *args: str, new_user_id: str = NEW, agent: str = AGENT) -> object:
    return runner.invoke(
        app,
        [
            "agents",
            "rename-account",
            agent,
            new_user_id,
            "--config",
            str(runtime_paths.config_path),
            "--storage-path",
            str(runtime_paths.storage_root),
            *args,
        ],
    )


def _journal_rows(runtime_paths: RuntimePaths, sql: str) -> list[tuple[object, ...]]:
    with closing(sqlite3.connect(event_journal_sqlite_path(runtime_paths.storage_root))) as connection:
        return connection.execute(sql).fetchall()


def test_dry_run_changes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server: _MatrixServer) -> None:
    """Without --apply the plan is printed and nothing on disk changes, not even the lock files."""
    runtime_paths = _install(tmp_path, monkeypatch)
    before = tree_snapshot(tmp_path)

    result = _invoke(runtime_paths)

    assert result.exit_code == 0, result.output
    assert tree_snapshot(tmp_path) == before
    # Rich would print the ":x:" of "@old:x:" as an emoji.
    assert f"Plan to adopt {NEW} for agent {AGENT}, whose stored account is {OLD}:" in result.output
    assert f"Set the username of agent_{AGENT}" in result.output
    assert "Rewrite the event journal" in result.output
    assert f"Replace {OLD} with {NEW} in {runtime_paths.config_path}" in result.output
    assert "Dry run: nothing was changed." in result.output
    assert "only show up when --apply rewrites the journal" in result.output
    # The login check belongs to --apply.
    assert server.requests == []


def test_apply_adopts_account(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server: _MatrixServer) -> None:
    """--apply checks the login, moves every store to the new ID, keeps a backup, and finds no leftovers."""
    runtime_paths = _install(tmp_path, monkeypatch)
    storage = runtime_paths.storage_root

    result = _invoke(runtime_paths, "--apply", "--yes")

    assert result.exit_code == 0, result.output
    login, logout = server.requests
    assert login.url == "http://localhost:8008/_matrix/client/v3/login"
    assert json.loads(login.content) == {
        "type": "m.login.password",
        "identifier": {"type": "m.id.user", "user": NEW},
        "password": PASSWORD,
        "initial_device_display_name": "mindroom rename check",
    }
    assert logout.url == "http://localhost:8008/_matrix/client/v3/logout"
    assert logout.headers["Authorization"] == "Bearer check-token"
    assert PASSWORD not in result.output

    account = safe_load(matrix_state_file(runtime_paths).read_bytes())["accounts"][f"agent_{AGENT}"]
    assert (account["username"], account["device_id"], account["access_token"]) == ("new", None, None)
    assert _journal_rows(runtime_paths, "SELECT principal_id, sender FROM journal_events ORDER BY principal_id") == [
        (NEW_PRINCIPAL, NEW),
        (ROUTER, NEW),
    ]
    assert _journal_rows(runtime_paths, "SELECT * FROM matrix_sync_consumers") == []
    assert not _oauth_path(storage, OLD).exists()
    assert _stored_worker_key(_oauth_path(storage, NEW)) == _key("user_agent", NEW)
    assert not (storage / "workers" / worker_dir_name(_key("user", OLD))).exists()
    new_worker = storage / "workers" / worker_dir_name(_key("user", NEW))
    assert (new_worker / "workspace" / "notes.md").read_text() == "notes"
    assert runtime_paths.config_path.read_text() == CONFIG.replace(OLD, NEW)

    (backup_dir,) = (storage / "agent-rename-backups").iterdir()
    assert backup_dir.name.endswith("-rename-account")
    assert (backup_dir / "manifest.json").is_file()
    assert str(backup_dir) in result.output
    assert f"Leftover scan: nothing in {storage} names {OLD} any more." in result.output


def test_refuses_while_runtime_lock_held(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    server: _MatrixServer,
) -> None:
    """A running MindRoom holds runtime.lock, so --apply refuses before the login check or any write."""
    runtime_paths = _install(tmp_path, monkeypatch)
    runtime_lock_path = tracking_dir(runtime_paths) / "runtime.lock"
    runtime_lock = try_exclusive_file_lock(runtime_lock_path)
    assert runtime_lock is not None
    before = tree_snapshot(tmp_path)
    try:
        result = _invoke(runtime_paths, "--apply", "--yes")
    finally:
        release_file_lock(runtime_lock)

    assert result.exit_code == 1, result.output
    assert f"Stop MindRoom first: another process holds {runtime_lock_path}." in result.output
    assert tree_snapshot(tmp_path) == before
    assert server.requests == []


def test_refuses_when_login_check_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    server: _MatrixServer,
) -> None:
    """A server that still rejects the new ID means it was not renamed there, so nothing changes."""
    runtime_paths = _install(tmp_path, monkeypatch)
    server.login_status = 403
    before = tree_snapshot(tmp_path)

    result = _invoke(runtime_paths, "--apply", "--yes")

    assert result.exit_code == 1, result.output
    assert (
        f"The server does not accept {NEW} with the stored password; rename the account on the server first."
        in result.output
    )
    assert tree_snapshot(tmp_path) == before
    assert [request.url.path for request in server.requests] == ["/_matrix/client/v3/login"]


@pytest.mark.parametrize(
    ("agent", "new_user_id", "reason"),
    [
        ("ghost", NEW, "ghost is not an agent in"),
        (AGENT, "@new:y", f"@new:y is not on x, the server of {OLD}"),
        (AGENT, "new", "new is not a Matrix user ID"),
        (AGENT, "@router:x", "@router:x is already the stored account agent_router"),
    ],
)
def test_refuses_unknown_agent_and_foreign_domain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    server: _MatrixServer,
    agent: str,
    new_user_id: str,
    reason: str,
) -> None:
    """Requests the command cannot honor refuse with the reason and change nothing, with or without --apply."""
    runtime_paths = _install(tmp_path, monkeypatch)
    before = tree_snapshot(tmp_path)

    for args in ((), ("--apply", "--yes")):
        result = _invoke(runtime_paths, *args, agent=agent, new_user_id=new_user_id)

        assert result.exit_code == 1, result.output
        assert reason in result.output
    assert tree_snapshot(tmp_path) == before
    assert server.requests == []


def test_rerun_after_apply_is_a_no_op(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server: _MatrixServer) -> None:
    """Once the stored account holds the new ID, running again says so and changes nothing."""
    runtime_paths = _install(tmp_path, monkeypatch)
    assert _invoke(runtime_paths, "--apply", "--yes").exit_code == 0
    requests = len(server.requests)
    after_apply = tree_snapshot(tmp_path)

    for args in ((), ("--apply", "--yes")):
        result = _invoke(runtime_paths, *args)

        assert result.exit_code == 0, result.output
        assert "Already adopted; nothing to do." in result.output
    assert tree_snapshot(tmp_path) == after_apply
    assert len(server.requests) == requests


def test_discard_new_principal_reaches_the_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    server: _MatrixServer,
) -> None:
    """Rows MindRoom wrote under the new ID refuse the rename until --discard-new-principal deletes them."""
    runtime_paths = _install(tmp_path, monkeypatch, new_principal_rows=True)
    before = tree_snapshot(tmp_path)

    refused = _invoke(runtime_paths, "--apply", "--yes", "--skip-login-check")

    assert refused.exit_code == 1, refused.output
    assert "Rerun with --discard-new-principal" in refused.output
    assert tree_snapshot(tmp_path) == before

    applied = _invoke(runtime_paths, "--apply", "--yes", "--skip-login-check", "--discard-new-principal")

    assert applied.exit_code == 0, applied.output
    assert _journal_rows(runtime_paths, "SELECT principal_id FROM matrix_sync_consumers") == []
    assert server.requests == []


def test_declining_the_confirmation_changes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    server: _MatrixServer,
) -> None:
    """Without --yes the operator is asked, and answering no exits 1 with nothing changed."""
    runtime_paths = _install(tmp_path, monkeypatch)
    before = tree_snapshot(tmp_path)

    result = runner.invoke(
        app,
        [
            "agents",
            "rename-account",
            AGENT,
            NEW,
            "--config",
            str(runtime_paths.config_path),
            "--storage-path",
            str(runtime_paths.storage_root),
            "--apply",
        ],
        input="n\n",
    )

    assert result.exit_code == 1, result.output
    assert "Apply this plan?" in result.output
    assert tree_snapshot(tmp_path) == before
    assert len(server.requests) == 2


def test_failed_undo_exits_2_naming_the_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    server: _MatrixServer,
) -> None:
    """When undoing a failed step fails too, the command exits 2 and points at the manifest for a manual restore."""
    runtime_paths = _install(tmp_path, monkeypatch)
    manifest = runtime_paths.storage_root / "agent-rename-backups" / "x-rename-account" / "manifest.json"

    def fail_undo(plan: RenamePlan, *, kind: str, now: object) -> None:
        assert plan.steps
        assert kind == "rename-account"
        assert now is not None
        raise RenameUndoError(manifest, "step 2 failed, and undoing step 1 failed")

    monkeypatch.setattr(steps_module, "apply_plan", fail_undo)

    result = _invoke(runtime_paths, "--apply", "--yes")

    assert result.exit_code == 2, result.output
    assert "step 2 failed, and undoing step 1 failed" in result.output
    assert str(manifest) in result.output
    assert len(server.requests) == 2


class _RetiringManager:
    backend_name = "docker"

    def __init__(self) -> None:
        self.retired: list[str] = []

    def __enter__(self) -> _RetiringManager:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def retire_worker(self, worker_key: str) -> None:
        self.retired.append(worker_key)


def test_apply_retires_old_docker_workers_then_shuts_the_worker_manager_down(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    server: _MatrixServer,
) -> None:
    """On Docker the old worker's container is retired after the move, and the leased manager is shut down."""
    runtime_paths = _install(tmp_path, monkeypatch)
    monkeypatch.setenv("MINDROOM_WORKER_BACKEND", "docker")
    manager = _RetiringManager()
    events: list[str] = []

    def lease(_runtime_paths: RuntimePaths, *, runtime_config: Config | None) -> _RetiringManager:
        assert runtime_config is not None
        events.append("lease")
        return manager

    monkeypatch.setattr(account_files, "lease_configured_primary_worker_manager", lease)
    monkeypatch.setattr(workers_runtime, "shutdown_primary_worker_manager", lambda: events.append("shutdown"))

    result = _invoke(runtime_paths, "--apply", "--yes")

    assert result.exit_code == 0, result.output
    assert manager.retired == [_key("user", OLD)]
    assert events == ["lease", "shutdown"]
    assert len(server.requests) == 2
