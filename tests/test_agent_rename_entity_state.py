"""Tests for moving an agent's state root and renaming its session and learning databases with its config key."""

from __future__ import annotations

import json
import shutil
import sqlite3
import time
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from agno.db.sqlite import SqliteDb
from agno.metrics import RunMetrics
from agno.models.message import Message
from agno.run.agent import RunOutput, RunStartedEvent
from agno.run.base import RunStatus
from agno.session.agent import AgentSession

from mindroom import constants
from mindroom.agent_rename.entity_state import entity_state_steps, matrix_state_key_step
from mindroom.agent_rename.steps import (
    BACKUPS_DIRECTORY_NAME,
    FileRewrite,
    MovePath,
    RenameApplyError,
    RenamePlan,
    RenameRefusedError,
    SqliteEdit,
    apply_plan,
)
from mindroom.config.agent import AgentConfig
from mindroom.config.main import Config
from mindroom.constants import matrix_state_file
from mindroom.history import summary_rows
from mindroom.history.replay import current_summary_text, scope_visible_runs
from mindroom.history.storage import (
    add_pending_force_compaction_scope,
    has_queued_force_compaction,
    read_scope_seen_event_ids,
    update_scope_seen_event_ids,
)
from mindroom.history.types import HistoryScope
from mindroom.matrix.state import MatrixState
from mindroom.session_storage.agent_storage import (
    create_session_storage,
    create_state_storage,
    get_agent_session,
)
from mindroom.tool_system.worker_routing import agent_state_root_path
from mindroom.usage.stats import collect_admin_usage
from tests.conftest import TEST_ACCESS_TOKEN, seed_session, test_runtime_paths

if TYPE_CHECKING:
    from pathlib import Path

    from agno.db.base import BaseDb

    from mindroom.agent_rename.steps import Step
    from mindroom.constants import RuntimePaths

OLD = "code"
NEW = "coder"
# Its name extends the renamed agent's, so a substring match would wrongly touch it.
PREFIX = "code_team"
ALICE = "@alice:x"
SESSION = "s1"
NOW = datetime(2026, 10, 20, 9, 30, 0, tzinfo=UTC)

# The compaction archive every release from #2351 to v2026.10.340 created beside the sessions table.
_ARCHIVE_DDL = (
    "CREATE TABLE code_sessions_compactions (id INTEGER PRIMARY KEY, "
    "session_id TEXT NOT NULL REFERENCES code_sessions(session_id) ON DELETE CASCADE, "
    "scope_key TEXT NOT NULL, summary TEXT, summary_model TEXT, "
    "legacy INTEGER NOT NULL DEFAULT 0, legacy_event_ids TEXT, created_at INTEGER NOT NULL)",
    "CREATE TABLE code_sessions_compacted_runs (id INTEGER PRIMARY KEY, "
    "compaction_id INTEGER NOT NULL REFERENCES code_sessions_compactions(id) ON DELETE CASCADE, "
    "session_id TEXT NOT NULL, run_id TEXT NOT NULL, run_type TEXT, event_ids TEXT NOT NULL, "
    "seen_event_ids TEXT NOT NULL DEFAULT '[]', run_data TEXT, UNIQUE(session_id, run_id))",
)


def _root(runtime_paths: RuntimePaths, agent: str) -> Path:
    return agent_state_root_path(runtime_paths.storage_root, agent)


def _sessions(runtime_paths: RuntimePaths, agent: str) -> BaseDb:
    return create_state_storage(
        agent,
        _root(runtime_paths, agent),
        subdir="sessions",
        session_table=f"{agent}_sessions",
    )


def _learning(runtime_paths: RuntimePaths, agent: str) -> SqliteDb:
    storage = create_state_storage(
        agent,
        _root(runtime_paths, agent),
        subdir="learning",
        session_table=f"{agent}_learning_sessions",
    )
    assert isinstance(storage, SqliteDb)
    return storage


def _run(agent: str, run_id: str) -> RunOutput:
    return RunOutput(
        run_id=run_id,
        agent_id=agent,
        agent_name=agent.title(),
        session_id=SESSION,
        user_id=ALICE,
        status=RunStatus.completed,
        model="test-model",
        model_provider="test-provider",
        content=f"answer {run_id}",
        messages=[Message(role="user", content=f"question {run_id}"), Message(role="assistant", content="answer")],
        metrics=RunMetrics(input_tokens=2, output_tokens=3, total_tokens=5),
        events=[RunStartedEvent(agent_id=agent, run_id=run_id, session_id=SESSION)],
        metadata={constants.MATRIX_SEEN_EVENT_IDS_METADATA_KEY: [f"${run_id}"]},
    )


def _seed_agent(runtime_paths: RuntimePaths, agent: str) -> None:
    """Write an agent's history and learnings the way MindRoom does: two runs, the first summarized."""
    scope = HistoryScope(kind="agent", scope_id=agent)
    session = AgentSession(
        session_id=SESSION,
        agent_id=agent,
        user_id=ALICE,
        # Written like agno/agent/_storage.py's get_agent_data.
        agent_data={"name": agent.title(), "agent_id": agent},
        session_data={"session_state": add_pending_force_compaction_scope(None, scope)},
        metadata={},
        runs=[_run(agent, "r1"), _run(agent, "r2")],
    )
    update_scope_seen_event_ids(session, scope, ["$preserved"])
    # Another scope whose name extends this agent's keeps its own state.
    update_scope_seen_event_ids(session, HistoryScope(kind="agent", scope_id=f"{agent}_team"), ["$other"])
    storage = _sessions(runtime_paths, agent)
    try:
        seed_session(storage, session)
        covered = summary_rows.covered_runs(storage, session_id=SESSION, run_ids=["r1"])
        summary_rows.append_summary(
            storage,
            session_id=SESSION,
            scope_key=scope.key,
            based_on=None,
            covered=covered,
            summary=f"summary of {agent}'s r1",
            summary_model="test-model",
        )
    finally:
        storage.close()
    learning = _learning(runtime_paths, agent)
    try:
        learning.upsert_learning(
            id=f"user_memory_{agent}",
            learning_type="user_memory",
            content={"memories": [f"{agent} learned this"]},
            user_id=ALICE,
            agent_id=agent,
        )
        # Creates the learning store's sessions table, which the rename covers too.
        learning.upsert_session(AgentSession(session_id=SESSION, agent_id=agent, user_id=ALICE))
    finally:
        learning.close()


def _install(tmp_path: Path) -> RuntimePaths:
    runtime_paths = test_runtime_paths(tmp_path)
    for agent in (OLD, PREFIX):
        _seed_agent(runtime_paths, agent)
        (_root(runtime_paths, agent) / "workspace").mkdir()
        (_root(runtime_paths, agent) / "workspace" / "notes.md").write_text(f"{agent} notes\n", encoding="utf-8")
    return runtime_paths


def _apply(runtime_paths: RuntimePaths, steps: list[Step]) -> None:
    apply_plan(
        RenamePlan(storage_root=runtime_paths.storage_root, steps=tuple(steps), reports=()),
        kind="rename",
        now=NOW,
    )


def _dump(database: Path) -> list[str]:
    with closing(sqlite3.connect(f"{database.absolute().as_uri()}?mode=ro", uri=True)) as connection:
        return list(connection.iterdump())


def _tree(root: Path) -> dict[str, object]:
    """Map every path under ``root`` outside the backups to its bytes, or its rows for a SQLite database."""
    tree: dict[str, object] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts[0] == BACKUPS_DIRECTORY_NAME:
            continue
        if path.is_dir():
            tree[relative.as_posix()] = "dir"
        elif path.suffix == ".db":
            tree[relative.as_posix()] = _dump(path)
        else:
            tree[relative.as_posix()] = path.read_bytes()
    return tree


def _schema(database: Path) -> set[tuple[str, str, str]]:
    with closing(sqlite3.connect(database)) as connection:
        return set(connection.execute("SELECT type, name, tbl_name FROM sqlite_master"))


def _config() -> Config:
    return Config(agents={NEW: AgentConfig(display_name="Code"), PREFIX: AgentConfig(display_name="Code Team")})


def test_history_visible_to_new_name(tmp_path: Path) -> None:
    """After the rename the agent replays its old runs, summary, seen events, queued compaction and learnings."""
    runtime_paths = _install(tmp_path)

    _apply(runtime_paths, entity_state_steps(runtime_paths, old_name=OLD, new_name=NEW))

    scope = HistoryScope(kind="agent", scope_id=NEW)
    storage = create_session_storage(NEW, _config(), runtime_paths, execution_identity=None)
    try:
        session = get_agent_session(storage, SESSION)
        assert session is not None
        # r1 is behind the summary row, so only r2 replays; without the summary's scope renamed both would.
        assert [run.run_id for run in scope_visible_runs(session, scope)] == ["r2"]
        assert current_summary_text(session) == "summary of code's r1"
        assert [message.content for message in session.get_messages(agent_id=NEW)] == ["question r2", "answer"]
        assert read_scope_seen_event_ids(storage, session, scope) == {"$preserved", "$r1", "$r2"}
        assert has_queued_force_compaction(storage, SESSION, scope)
    finally:
        storage.close()
    learning = _learning(runtime_paths, NEW)
    try:
        learnings = learning.get_learnings(agent_id=NEW)
    finally:
        learning.close()
    assert [row["content"] for row in learnings] == [{"memories": ["code learned this"]}]
    assert not _root(runtime_paths, OLD).exists()
    assert (_root(runtime_paths, NEW) / "workspace" / "notes.md").read_text(encoding="utf-8") == "code notes\n"


def test_tables_indexes_and_schema_versions_renamed(tmp_path: Path) -> None:
    """Every table, index and schema-version row of the agent's sessions and learning takes the new name."""
    runtime_paths = _install(tmp_path)
    old_sessions = _root(runtime_paths, OLD) / "sessions" / f"{OLD}.db"
    old_learning = _root(runtime_paths, OLD) / "learning" / f"{OLD}.db"
    expected_sessions = {
        tuple(part.replace("code_sessions", "coder_sessions") for part in entry) for entry in _schema(old_sessions)
    }
    expected_learning = {
        tuple(part.replace("code_learning_sessions", "coder_learning_sessions") for part in entry)
        for entry in _schema(old_learning)
    }

    steps = entity_state_steps(runtime_paths, old_name=OLD, new_name=NEW)
    _apply(runtime_paths, steps)

    sessions = _root(runtime_paths, NEW) / "sessions" / f"{NEW}.db"
    learning = _root(runtime_paths, NEW) / "learning" / f"{NEW}.db"
    assert [type(step) for step in steps] == [MovePath, MovePath, SqliteEdit, MovePath, SqliteEdit]
    assert _schema(sessions) == expected_sessions
    assert {name for _type, name, _table in expected_sessions} >= {
        "coder_sessions_runs",
        "coder_sessions_usage",
        "coder_sessions_summaries",
        "coder_sessions_removed_runs",
        "idx_coder_sessions_runs_session_id_run_index",
        "coder_sessions_summaries_scope",
        "coder_sessions_removed_runs_session",
    }
    assert _schema(learning) == expected_learning
    with closing(sqlite3.connect(sessions)) as connection:
        stored_sql = connection.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL").fetchall()
        assert all("code_sessions" not in sql for (sql,) in stored_sql)
        assert set(connection.execute("SELECT table_name FROM agno_schema_versions")) == {
            ("coder_sessions",),
            ("coder_sessions_runs",),
        }
        assert {row[2] for row in connection.execute("PRAGMA foreign_key_list(coder_sessions_runs)")} == {
            "coder_sessions",
        }
        assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        ((agent_id, agent_data, metadata),) = connection.execute(
            "SELECT agent_id, agent_data, metadata FROM coder_sessions",
        )
        assert (agent_id, json.loads(agent_data)) == (NEW, {"name": "Code", "agent_id": NEW})
        assert set(json.loads(metadata)[constants.MINDROOM_MATRIX_HISTORY_METADATA_KEY]["states"]) == {
            f"agent:{NEW}",
            f"agent:{PREFIX}",
        }
        runs = connection.execute("SELECT agent_id, run_data FROM coder_sessions_runs ORDER BY run_index").fetchall()
        assert [(agent_id, json.loads(run_data)["agent_id"]) for agent_id, run_data in runs] == [(NEW, NEW), (NEW, NEW)]
        assert [json.loads(run_data)["events"][0]["agent_id"] for _agent_id, run_data in runs] == [NEW, NEW]
        assert connection.execute("SELECT scope_key FROM coder_sessions_summaries").fetchall() == [(f"agent:{NEW}",)]
    with closing(sqlite3.connect(learning)) as connection:
        assert set(connection.execute("SELECT table_name FROM agno_schema_versions")) == {
            ("agno_learnings",),
            ("coder_learning_sessions",),
        }
        assert connection.execute("SELECT agent_id FROM agno_learnings").fetchall() == [(NEW,)]
        assert connection.execute("SELECT agent_id FROM coder_learning_sessions").fetchall() == [(NEW,)]


def test_usage_stats_find_new_db(tmp_path: Path) -> None:
    """Usage reports find the renamed agent's database, with the usage recorded under the old name."""
    runtime_paths = _install(tmp_path)

    _apply(runtime_paths, entity_state_steps(runtime_paths, old_name=OLD, new_name=NEW))

    report = collect_admin_usage(config=_config(), runtime_paths=runtime_paths)
    assert {row.key: (row.run_count, row.retained_run_totals.total_tokens) for row in report.breakdown} == {
        NEW: (2, 10),
        PREFIX: (2, 10),
    }
    assert report.coverage.unavailable_sources == 0


def test_prefix_agent_untouched(tmp_path: Path) -> None:
    """Renaming ``code`` plans nothing on ``code_team`` and leaves its state root as it was."""
    runtime_paths = _install(tmp_path)
    prefix_root = _root(runtime_paths, PREFIX)
    before = _tree(prefix_root)

    steps = entity_state_steps(runtime_paths, old_name=OLD, new_name=NEW)
    _apply(runtime_paths, steps)

    touched = [
        path
        for step in steps
        for path in (getattr(step, "source", None), getattr(step, "destination", None), getattr(step, "database", None))
        if path is not None
    ]
    assert not any(path.is_relative_to(prefix_root) for path in touched)
    assert _tree(prefix_root) == before


def test_matrix_state_key_renamed_username_kept(tmp_path: Path) -> None:
    """The stored account moves to the new key with its username, device and token; other accounts stay."""
    runtime_paths = test_runtime_paths(tmp_path / "install")
    state = MatrixState()
    state.add_account(
        "agent_router",
        "mindroom_router",
        "r",
        domain="x",
        device_id="ROUTER",
        access_token=TEST_ACCESS_TOKEN,
    )
    state.add_account(
        f"agent_{OLD}",
        "mindroom_code",
        "c",
        domain="x",
        device_id="CODE",
        access_token=TEST_ACCESS_TOKEN,
    )
    state.add_account(
        f"agent_{PREFIX}",
        "mindroom_code_team",
        "t",
        domain="x",
        device_id="TEAM",
        access_token=TEST_ACCESS_TOKEN,
    )
    state.add_room("lobby", "!lobby:x", "#lobby:x", "Lobby")
    state.save(runtime_paths)

    _apply(runtime_paths, [matrix_state_key_step(runtime_paths, old_name=OLD, new_name=NEW)])

    expected = state.model_copy(deep=True)
    expected.accounts = {
        "agent_router": state.accounts["agent_router"],
        f"agent_{NEW}": state.accounts[f"agent_{OLD}"],
        f"agent_{PREFIX}": state.accounts[f"agent_{PREFIX}"],
    }
    expected_paths = test_runtime_paths(tmp_path / "expected")
    expected.save(expected_paths)
    assert matrix_state_file(runtime_paths).read_bytes() == matrix_state_file(expected_paths).read_bytes()


@pytest.mark.parametrize("accounts", [(), (f"agent_{OLD}", f"agent_{NEW}")])
def test_matrix_state_key_refuses_without_old_or_with_new_account(tmp_path: Path, accounts: tuple[str, ...]) -> None:
    """A missing old account or an existing new one refuses before any write."""
    runtime_paths = test_runtime_paths(tmp_path)
    state = MatrixState()
    for key in accounts:
        state.add_account(key, key, "secret", domain="x")
    state.save(runtime_paths)

    with pytest.raises(RenameRefusedError, match="agent_"):
        matrix_state_key_step(runtime_paths, old_name=OLD, new_name=NEW)


def test_existing_destination_refuses(tmp_path: Path) -> None:
    """A state root or database already under the new name refuses while planning."""
    runtime_paths = _install(tmp_path)
    stale = _root(runtime_paths, OLD) / "sessions" / f"{NEW}.db"
    stale.touch()
    with pytest.raises(RenameRefusedError, match=str(stale)):
        entity_state_steps(runtime_paths, old_name=OLD, new_name=NEW)
    stale.unlink()

    _root(runtime_paths, NEW).mkdir()
    with pytest.raises(RenameRefusedError, match=str(_root(runtime_paths, NEW))):
        entity_state_steps(runtime_paths, old_name=OLD, new_name=NEW)


def test_agent_without_state_plans_nothing(tmp_path: Path) -> None:
    """An agent that never ran has no state root, so there is nothing to move."""
    assert entity_state_steps(test_runtime_paths(tmp_path), old_name=OLD, new_name=NEW) == []


def _fail(_content: bytes) -> bytes:
    msg = "injected failure"
    raise RuntimeError(msg)


def test_failure_after_any_step_restores_the_tree(tmp_path: Path) -> None:
    """A failure after any number of completed steps undoes them, leaving every file and database row as before."""
    step_count = len(entity_state_steps(_install(tmp_path / "count"), old_name=OLD, new_name=NEW))
    for completed in range(step_count + 1):
        runtime_paths = _install(tmp_path / str(completed))
        sentinel = runtime_paths.storage_root / "sentinel"
        sentinel.write_text("unchanged", encoding="utf-8")
        before = _tree(runtime_paths.storage_root)
        steps = entity_state_steps(runtime_paths, old_name=OLD, new_name=NEW)[:completed]

        with pytest.raises(RenameApplyError, match="injected failure"):
            _apply(runtime_paths, [*steps, FileRewrite(sentinel, _fail, "Fail on purpose")])

        assert _tree(runtime_paths.storage_root) == before


def _legacy_install(tmp_path: Path) -> RuntimePaths:
    """Return an install whose session database an older release left: Agno 2 blob runs and a compaction archive."""
    runtime_paths = test_runtime_paths(tmp_path)
    storage = _sessions(runtime_paths, OLD)
    seed_session(storage, AgentSession(session_id=SESSION, agent_id=OLD, user_id=ALICE, runs=[_run(OLD, "r2")]))
    storage.close()
    with closing(sqlite3.connect(_root(runtime_paths, OLD) / "sessions" / f"{OLD}.db")) as connection:
        connection.execute("DROP TABLE code_sessions_summaries")
        connection.execute("DROP TABLE code_sessions_removed_runs")
        for statement in _ARCHIVE_DDL:
            connection.execute(statement)
        connection.execute(
            "INSERT INTO code_sessions_compactions (id, session_id, scope_key, summary, created_at) "
            "VALUES (1, ?, 'agent:code', 'archived summary', ?)",
            (SESSION, int(time.time())),
        )
        connection.execute(
            "INSERT INTO code_sessions_compacted_runs (compaction_id, session_id, run_id, run_type, event_ids, "
            "seen_event_ids, run_data) VALUES (1, ?, 'r1', 'agent', '[\"$r1\"]', '[\"$r1\"]', ?)",
            (SESSION, json.dumps(_run(OLD, "r1").to_dict())),
        )
        # Databases Agno 2 created keep their runs blob, encoded twice.
        connection.execute("ALTER TABLE code_sessions ADD COLUMN runs JSON")
        connection.execute("UPDATE code_sessions SET runs = ?", (json.dumps(json.dumps([_run(OLD, "r0").to_dict()])),))
        connection.commit()
    return runtime_paths


def _replay(runtime_paths: RuntimePaths, agent: str) -> tuple[str | None, list[str | None], set[str]]:
    """Open the agent's sessions as MindRoom does and return its summary, replayed runs and seen events."""
    scope = HistoryScope(kind="agent", scope_id=agent)
    storage = _sessions(runtime_paths, agent)
    try:
        session = get_agent_session(storage, SESSION)
        assert session is not None
        runs = [run.run_id for run in scope_visible_runs(session, scope)]
        return current_summary_text(session), runs, read_scope_seen_event_ids(storage, session, scope)
    finally:
        storage.close()


def test_legacy_archive_and_blob_visible_to_new_name(tmp_path: Path) -> None:
    """A database an older release left replays for the new name exactly as it would have for the old one."""
    renamed = _legacy_install(tmp_path / "renamed")
    _apply(renamed, entity_state_steps(renamed, old_name=OLD, new_name=NEW))

    replayed = _replay(renamed, NEW)

    assert replayed == _replay(_legacy_install(tmp_path / "kept"), OLD)
    assert replayed[0] == "archived summary"
    assert replayed[2] == {"$r0", "$r1", "$r2"}


def test_separate_session_storage_moves_too(tmp_path: Path) -> None:
    """With ``MINDROOM_SESSION_STORAGE_PATH`` set the agent's sessions move inside that root as well."""
    base = test_runtime_paths(tmp_path)
    session_root = tmp_path / "session_state"
    runtime_paths = replace(
        base,
        process_env={**base.process_env, "MINDROOM_SESSION_STORAGE_PATH": str(session_root)},
    )
    storage = create_session_storage(OLD, Config(agents={OLD: AgentConfig(display_name="Code")}), runtime_paths, None)
    seed_session(storage, AgentSession(session_id=SESSION, agent_id=OLD, user_id=ALICE, runs=[_run(OLD, "r1")]))
    storage.close()
    assert (session_root / "agents" / OLD / "sessions" / f"{OLD}.db").is_file()

    _apply(runtime_paths, entity_state_steps(runtime_paths, old_name=OLD, new_name=NEW))

    storage = create_session_storage(NEW, _config(), runtime_paths, None)
    try:
        session = get_agent_session(storage, SESSION)
        assert session is not None
        assert [run.run_id for run in scope_visible_runs(session, HistoryScope(kind="agent", scope_id=NEW))] == ["r1"]
    finally:
        storage.close()
    assert not (session_root / "agents" / OLD).exists()


def test_uncheckpointed_wal_moves_with_its_database(tmp_path: Path) -> None:
    """Commits a crashed writer left in the ``-wal`` file stay with the renamed database."""
    runtime_paths = _install(tmp_path)
    database = _root(runtime_paths, OLD) / "sessions" / f"{OLD}.db"
    crashed = tmp_path / "crashed"
    crashed.mkdir()
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA wal_autocheckpoint = 0")
        connection.execute("UPDATE code_sessions SET user_id = '@wal:x'")
        connection.commit()
        # Copied while the connection is open, as a writer killed now would leave them.
        shutil.copy(database, crashed / database.name)
        shutil.copy(f"{database}-wal", crashed / f"{database.name}-wal")
    shutil.copy(crashed / database.name, database)
    shutil.copy(crashed / f"{database.name}-wal", f"{database}-wal")

    _apply(runtime_paths, entity_state_steps(runtime_paths, old_name=OLD, new_name=NEW))

    with closing(sqlite3.connect(_root(runtime_paths, NEW) / "sessions" / f"{NEW}.db")) as connection:
        assert connection.execute("SELECT agent_id, user_id FROM coder_sessions").fetchall() == [(NEW, "@wal:x")]
