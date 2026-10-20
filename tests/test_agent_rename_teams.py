"""Tests for moving the team history scopes that include a renamed agent."""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from agno.agent import Agent
from agno.metrics import RunMetrics
from agno.models.message import Message
from agno.models.response import ToolExecution
from agno.run.agent import RunOutput, RunStartedEvent
from agno.run.base import RunStatus
from agno.run.requirement import RunRequirement
from agno.run.team import RunStartedEvent as TeamRunStartedEvent
from agno.run.team import TeamRunOutput
from agno.session.team import TeamSession

from mindroom import constants
from mindroom.agent_rename.steps import (
    FileRewrite,
    MovePath,
    RenameApplyError,
    RenamePlan,
    RenameRefusedError,
    SqliteEdit,
    apply_plan,
)
from mindroom.agent_rename.teams import team_steps
from mindroom.config.agent import AgentConfig, AgentPrivateConfig, TeamConfig
from mindroom.config.main import Config
from mindroom.history import summary_rows
from mindroom.history.replay import current_summary_text, scope_visible_runs
from mindroom.history.session_context import (
    TEAM_STATE_ROOT_DIRNAME,
    open_bound_scope_session_context,
    scope_session_storage_name,
)
from mindroom.history.storage import (
    add_pending_force_compaction_scope,
    has_queued_force_compaction,
    read_scope_seen_event_ids,
    read_scope_state,
    set_force_compaction_state,
    update_scope_seen_event_ids,
)
from mindroom.history.team_scope import ad_hoc_team_scope_id
from mindroom.history.types import HistoryScope
from mindroom.tool_system.worker_routing import ToolExecutionIdentity
from tests.agent_rename_helpers import sqlite_schema, tree_rows, tree_snapshot
from tests.conftest import seed_session, test_runtime_paths

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from mindroom.agent_rename.steps import Step
    from mindroom.constants import RuntimePaths
    from mindroom.history.session_context import ScopeSessionContext

OLD = "code"
NEW = "coder"
# Its name extends the renamed agent's, so a substring match would wrongly touch its scopes.
PREFIX = "code_team"
HELPER = "helper"
# A private agent partitions every ad-hoc team it joins by requester.
DIARY = "diary"
ALICE = "@alice:x"
SESSION = "s1"
NOW = datetime(2026, 10, 20, 9, 30, 0, tzinfo=UTC)
# A configured team with the renamed agent, and one without it.
CREW = "crew"
DOCS = "docs"


@dataclass(frozen=True)
class _Team:
    """One team history scope of the fixture: its members, and its configured name if it has one."""

    members: tuple[str, ...]
    team_name: str | None = None

    def renamed(self) -> _Team:
        return _Team(tuple(NEW if member == OLD else member for member in self.members), self.team_name)


AD_HOC = _Team((OLD, HELPER))
PRIVATE_AD_HOC = _Team((OLD, DIARY))
CONFIGURED = _Team((OLD, HELPER), CREW)
PREFIX_AD_HOC = _Team((PREFIX, HELPER))
OTHER_CONFIGURED = _Team((PREFIX, HELPER), DOCS)
_TEAMS = (AD_HOC, PRIVATE_AD_HOC, CONFIGURED, PREFIX_AD_HOC, OTHER_CONFIGURED)


def _config(agent: str) -> Config:
    """Return the config naming the renamed agent ``agent``, as it is before (``OLD``) or after (``NEW``)."""
    return Config(
        agents={
            agent: AgentConfig(display_name="Code"),
            PREFIX: AgentConfig(display_name="Code Team"),
            HELPER: AgentConfig(display_name="Helper"),
            DIARY: AgentConfig(display_name="Diary", private=AgentPrivateConfig(per="user")),
        },
        teams={
            CREW: TeamConfig(display_name="Crew", role="Ships releases", agents=[agent, HELPER]),
            DOCS: TeamConfig(display_name="Docs", role="Writes docs", agents=[PREFIX, HELPER]),
        },
    )


@contextmanager
def _team_context(runtime_paths: RuntimePaths, config: Config, team: _Team) -> Iterator[ScopeSessionContext]:
    """Open a team's history the way a live team run of its members does, scope resolution included."""
    with open_bound_scope_session_context(
        agents=[Agent(id=member, name=member.title()) for member in team.members],
        session_id=SESSION,
        runtime_paths=runtime_paths,
        config=config,
        execution_identity=ToolExecutionIdentity(
            channel="matrix",
            agent_name=team.members[0],
            requester_id=ALICE,
            room_id="!room:x",
            thread_id=None,
            resolved_thread_id=None,
            session_id=SESSION,
        ),
        team_name=team.team_name,
        create_session_if_missing=True,
    ) as context:
        assert context is not None
        yield context


def _member_run(member: str, team_run_id: str) -> RunOutput:
    run_id = f"{team_run_id}-{member}"
    return RunOutput(
        run_id=run_id,
        agent_id=member,
        agent_name=member.title(),
        parent_run_id=team_run_id,
        session_id=SESSION,
        user_id=ALICE,
        status=RunStatus.completed,
        content=f"{member} did {team_run_id}",
        messages=[Message(role="user", content=f"task {team_run_id}"), Message(role="assistant", content="done")],
        events=[RunStartedEvent(agent_id=member, run_id=run_id, session_id=SESSION)],
    )


def _team_run(scope_id: str, team: _Team, run_id: str) -> TeamRunOutput:
    requirement = RunRequirement(tool_execution=ToolExecution(tool_name="approve", tool_call_id=f"{run_id}-call"))
    requirement.member_agent_id = team.members[0]
    return TeamRunOutput(
        run_id=run_id,
        team_id=scope_id,
        team_name=f"Team-{'-'.join(team.members)}",
        session_id=SESSION,
        user_id=ALICE,
        status=RunStatus.completed,
        model="test-model",
        model_provider="test-provider",
        content=f"answer {run_id}",
        messages=[Message(role="user", content=f"question {run_id}"), Message(role="assistant", content="answer")],
        metrics=RunMetrics(input_tokens=2, output_tokens=3, total_tokens=5),
        member_responses=[_member_run(member, run_id) for member in team.members],
        events=[TeamRunStartedEvent(team_id=scope_id, run_id=run_id, session_id=SESSION)],
        requirements=[requirement],
        metadata={constants.MATRIX_SEEN_EVENT_IDS_METADATA_KEY: [f"${run_id}"]},
    )


def _seed_team(runtime_paths: RuntimePaths, config: Config, team: _Team) -> None:
    """Write a team's history the way MindRoom does: two team runs with their member runs, the first summarized."""
    with _team_context(runtime_paths, config, team) as context:
        scope, storage = context.scope, context.storage
        runs: list[RunOutput | TeamRunOutput] = []
        for run_id in ("r1", "r2"):
            team_run = _team_run(scope.scope_id, team, run_id)
            # Agno saves each member's run into the team session as its own run, then the team's run.
            runs += [*team_run.member_responses, team_run]
        session = TeamSession(
            session_id=SESSION,
            team_id=scope.scope_id,
            user_id=ALICE,
            # Written like agno/team/_telemetry.py's get_team_data.
            team_data={"name": f"Team-{'-'.join(team.members)}", "team_id": scope.scope_id},
            session_data={"session_state": add_pending_force_compaction_scope({}, scope)},
            metadata={},
            runs=runs,
        )
        update_scope_seen_event_ids(session, scope, ["$preserved"])
        set_force_compaction_state(session, scope, read_scope_state(session, scope), force=True)
        seed_session(storage, session)
        covered = summary_rows.covered_runs(storage, session_id=SESSION, run_ids=["r1"])
        summary_rows.append_summary(
            storage,
            session_id=SESSION,
            scope_key=scope.key,
            based_on=None,
            covered=covered,
            summary=f"summary of {scope.scope_id}'s r1",
            summary_model="test-model",
        )


def _install(runtime_paths: RuntimePaths) -> RuntimePaths:
    for team in _TEAMS:
        _seed_team(runtime_paths, _config(OLD), team)
    return runtime_paths


def _scope_id(config: Config, team: _Team) -> str:
    if team.team_name is not None:
        return team.team_name
    scope_id = ad_hoc_team_scope_id(team.members, config.agents, requester_user_id=ALICE)
    assert scope_id is not None
    return scope_id


def _store(runtime_paths: RuntimePaths, scope_id: str) -> Path:
    """Return the state root of a team scope's session store."""
    storage_name = scope_session_storage_name(HistoryScope(kind="team", scope_id=scope_id))
    return runtime_paths.storage_root / TEAM_STATE_ROOT_DIRNAME / storage_name


def _apply(runtime_paths: RuntimePaths, steps: list[Step]) -> None:
    apply_plan(
        RenamePlan(storage_root=runtime_paths.storage_root, steps=tuple(steps), reports=()),
        kind="rename",
        now=NOW,
    )


def _plan(runtime_paths: RuntimePaths) -> list[Step]:
    return team_steps(runtime_paths, config=_config(OLD), old_name=OLD, new_name=NEW)


def _touches(step: Step, store: Path) -> bool:
    """Whether a step reads or writes a path inside ``store``."""
    paths = (getattr(step, "source", None), getattr(step, "destination", None), getattr(step, "database", None))
    return any(path is not None and path.is_relative_to(store) for path in paths)


@dataclass(frozen=True)
class _History:
    scope_id: str
    replayed: list[str | None]
    summary: str | None
    seen: set[str]
    queued: bool
    member_runs: list[tuple[str | None, str | None]]
    member_responses: list[list[str | None]]
    requirement_members: list[str | None]
    member_messages: list[str]


def _history(runtime_paths: RuntimePaths, config: Config, team: _Team) -> _History:
    """Load a team's history through MindRoom's history code, under the scope its members resolve to now."""
    with _team_context(runtime_paths, config, team) as context:
        session, scope = context.session, context.scope
        assert context.session_exists
        assert isinstance(session, TeamSession)
        replayed = scope_visible_runs(session, scope)
        return _History(
            scope_id=scope.scope_id,
            replayed=[run.run_id for run in replayed],
            summary=current_summary_text(session),
            seen=read_scope_seen_event_ids(context.storage, session, scope),
            queued=has_queued_force_compaction(context.storage, SESSION, scope),
            member_runs=[(run.agent_id, run.run_id) for run in session.runs or [] if isinstance(run, RunOutput)],
            member_responses=[
                [member.agent_id for member in run.member_responses if isinstance(member, RunOutput)]
                for run in replayed
                if isinstance(run, TeamRunOutput)
            ],
            requirement_members=[
                requirement.member_agent_id
                for run in replayed
                if isinstance(run, TeamRunOutput)
                for requirement in run.requirements or []
            ],
            # Agno's member filter, which finds a member's interactions in a team session by agent id.
            member_messages=[str(message.content) for message in session.get_messages(member_ids=[team.members[0]])],
        )


@pytest.mark.parametrize("team", [AD_HOC, PRIVATE_AD_HOC], ids=["shared", "private"])
def test_ad_hoc_history_visible_under_recomputed_scope(tmp_path: Path, team: _Team) -> None:
    """An ad-hoc team with the agent replays its history under the scope id its members resolve to after the rename."""
    runtime_paths = _install(test_runtime_paths(tmp_path))
    old_scope_id = _scope_id(_config(OLD), team)
    before = tree_snapshot(runtime_paths.storage_root)

    steps = _plan(runtime_paths)

    # Planning reads the stores without writing a byte, side files included.
    assert tree_snapshot(runtime_paths.storage_root) == before
    _apply(runtime_paths, steps)

    history = _history(runtime_paths, _config(NEW), team.renamed())
    renamed = "+".join(sorted(team.renamed().members))
    assert history.scope_id == old_scope_id.replace("+".join(team.members), renamed)
    assert history.scope_id == _scope_id(_config(NEW), team.renamed())
    assert history.replayed == ["r2"]
    assert history.summary == f"summary of {old_scope_id}'s r1"
    assert history.seen == {"$preserved", "$r1", "$r2"}
    assert history.queued
    other = team.members[1]
    # r1 and its member runs are folded behind the summary, so only r2's load.
    assert history.member_runs == [(NEW, f"r2-{OLD}"), (other, f"r2-{other}")]
    assert history.member_responses == [[NEW, other]]
    assert history.requirement_members == [NEW]
    assert history.member_messages == ["task r2", "done"]
    assert not _store(runtime_paths, old_scope_id).exists()
    assert sorted(path.name for path in (_store(runtime_paths, history.scope_id) / "sessions").iterdir()) == [
        f"{_store(runtime_paths, history.scope_id).name}.db",
    ]


def test_configured_team_member_runs_visible_as_new_name(tmp_path: Path) -> None:
    """A configured team keeps its store, and the agent's member runs in it carry the new name."""
    runtime_paths = _install(test_runtime_paths(tmp_path))
    store = _store(runtime_paths, CREW)

    steps = _plan(runtime_paths)
    _apply(runtime_paths, steps)

    assert [type(step) for step in steps if _touches(step, store)] == [SqliteEdit]
    history = _history(runtime_paths, _config(NEW), CONFIGURED.renamed())
    assert history.scope_id == CREW
    assert history.replayed == ["r2"]
    assert history.summary == "summary of crew's r1"
    assert history.seen == {"$preserved", "$r1", "$r2"}
    assert history.queued
    assert history.member_runs == [(NEW, "r2-code"), (HELPER, "r2-helper")]
    assert history.member_responses == [[NEW, HELPER]]
    assert history.requirement_members == [NEW]
    assert history.member_messages == ["task r2", "done"]
    with _team_context(runtime_paths, _config(NEW), CONFIGURED.renamed()) as context:
        assert isinstance(context.session, TeamSession)
        assert context.session.get_messages(member_ids=[OLD]) == []


def test_scopes_without_agent_untouched(tmp_path: Path) -> None:
    """Renaming ``code`` plans nothing on the stores of ``team_code_team+helper`` or the team ``docs``."""
    runtime_paths = _install(test_runtime_paths(tmp_path))
    stores = [_store(runtime_paths, _scope_id(_config(OLD), team)) for team in (PREFIX_AD_HOC, OTHER_CONFIGURED)]
    before = [tree_rows(store) for store in stores]

    steps = _plan(runtime_paths)
    _apply(runtime_paths, steps)

    assert not any(_touches(step, store) for step in steps for store in stores)
    assert [tree_rows(store) for store in stores] == before
    assert _history(runtime_paths, _config(NEW), PREFIX_AD_HOC).replayed == ["r2"]


def test_ad_hoc_store_tables_and_rows_renamed(tmp_path: Path) -> None:
    """The moved store's tables, indexes and schema versions take its new name; its rows the new scope and agent."""
    runtime_paths = _install(test_runtime_paths(tmp_path))
    old_scope_id, new_scope_id = "team_code+helper", "team_coder+helper"
    old_store, new_store = _store(runtime_paths, old_scope_id), _store(runtime_paths, new_scope_id)
    old_name, new_name = old_store.name, new_store.name
    old_database = old_store / "sessions" / f"{old_name}.db"
    expected = {tuple(part.replace(old_name, new_name) for part in entry) for entry in sqlite_schema(old_database)}

    steps = _plan(runtime_paths)
    _apply(runtime_paths, steps)

    database = new_store / "sessions" / f"{new_name}.db"
    assert [type(step) for step in steps if _touches(step, old_store) or _touches(step, new_store)] == [
        MovePath,
        SqliteEdit,
        MovePath,
    ]
    assert sqlite_schema(database) == expected
    with closing(sqlite3.connect(database)) as connection:
        stored_sql = connection.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL").fetchall()
        assert all(old_name not in sql for (sql,) in stored_sql)
        assert set(connection.execute("SELECT table_name FROM agno_schema_versions")) == {
            (f"{new_name}_sessions",),
            (f"{new_name}_sessions_runs",),
        }
        assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        ((team_id, team_data, metadata, session_data),) = connection.execute(
            f'SELECT team_id, team_data, metadata, session_data FROM "{new_name}_sessions"',  # noqa: S608
        )
        assert (team_id, json.loads(team_data)) == (new_scope_id, {"name": "Team-code-helper", "team_id": new_scope_id})
        for key in (constants.MINDROOM_COMPACTION_METADATA_KEY, constants.MINDROOM_MATRIX_HISTORY_METADATA_KEY):
            assert set(json.loads(metadata)[key]["states"]) == {f"team:{new_scope_id}"}
        assert json.loads(session_data)["session_state"] == {
            constants.PENDING_COMPACTION_SCOPE_KEYS_SESSION_STATE_KEY: [f"team:{new_scope_id}"],
        }
        runs = connection.execute(
            f'SELECT run_id, agent_id, team_id, run_data FROM "{new_name}_sessions_runs" ORDER BY run_index',  # noqa: S608
        ).fetchall()
        assert [(run_id, agent_id, team_id) for run_id, agent_id, team_id, _data in runs] == [
            ("r1-code", NEW, None),
            ("r1-helper", HELPER, None),
            ("r1", None, new_scope_id),
            ("r2-code", NEW, None),
            ("r2-helper", HELPER, None),
            ("r2", None, new_scope_id),
        ]
        team_run = json.loads(runs[5][3])
        assert team_run["team_id"] == new_scope_id
        assert [event["team_id"] for event in team_run["events"]] == [new_scope_id]
        assert [member["agent_id"] for member in team_run["member_responses"]] == [NEW, HELPER]
        assert [member["events"][0]["agent_id"] for member in team_run["member_responses"]] == [NEW, HELPER]
        assert [requirement["member_agent_id"] for requirement in team_run["requirements"]] == [NEW]
        assert json.loads(runs[3][3])["agent_id"] == NEW
        assert connection.execute(f'SELECT scope_key FROM "{new_name}_sessions_summaries"').fetchall() == [  # noqa: S608
            (f"team:{new_scope_id}",),
        ]
        usage_query = f'SELECT usage_data FROM "{new_name}_sessions_usage"'  # noqa: S608
        usage = [json.loads(data) for (data,) in connection.execute(usage_query)]
        assert sorted(entry.get("team_id") or "" for entry in usage) == ["", "", "", "", new_scope_id, new_scope_id]


def _fail(_content: bytes) -> bytes:
    msg = "injected failure"
    raise RuntimeError(msg)


def test_failure_after_any_step_restores_the_tree(tmp_path: Path) -> None:
    """A failure after any number of completed steps, the last included, undoes them, leaving every row as before."""
    step_count = len(_plan(_install(test_runtime_paths(tmp_path / "count"))))
    assert step_count == 7
    for completed in range(step_count + 1):
        runtime_paths = _install(test_runtime_paths(tmp_path / str(completed)))
        sentinel = runtime_paths.storage_root / "sentinel"
        sentinel.write_text("unchanged", encoding="utf-8")
        before = tree_rows(runtime_paths.storage_root)
        steps = _plan(runtime_paths)[:completed]

        with pytest.raises(RenameApplyError, match="injected failure"):
            _apply(runtime_paths, [*steps, FileRewrite(sentinel, _fail, "Fail on purpose")])

        assert tree_rows(runtime_paths.storage_root) == before


def test_existing_new_scope_store_refuses(tmp_path: Path) -> None:
    """A store already under the recomputed scope refuses while planning, naming it."""
    runtime_paths = _install(test_runtime_paths(tmp_path))
    existing = _store(runtime_paths, "team_coder+helper")
    existing.mkdir(parents=True)

    with pytest.raises(RenameRefusedError, match=re.escape(str(existing))):
        _plan(runtime_paths)


def test_scope_already_naming_new_name_refuses(tmp_path: Path) -> None:
    """A scope with both names would merge two members into one, so it refuses."""
    runtime_paths = test_runtime_paths(tmp_path)
    config = _config(OLD)
    config.agents[NEW] = AgentConfig(display_name="Coder")
    _seed_team(runtime_paths, config, _Team((OLD, NEW)))

    with pytest.raises(RenameRefusedError, match=re.escape("team_code+coder")):
        _plan(runtime_paths)


def test_separate_session_storage_moves_too(tmp_path: Path) -> None:
    """With ``MINDROOM_SESSION_STORAGE_PATH`` set, the team stores in that root move and are edited there."""
    base = test_runtime_paths(tmp_path)
    session_root = tmp_path / "session_state"
    runtime_paths = replace(
        base,
        process_env={**base.process_env, "MINDROOM_SESSION_STORAGE_PATH": str(session_root)},
    )
    _seed_team(runtime_paths, _config(OLD), AD_HOC)
    old_store = session_root / TEAM_STATE_ROOT_DIRNAME / _store(runtime_paths, "team_code+helper").name
    assert old_store.is_dir()

    _apply(runtime_paths, _plan(runtime_paths))

    history = _history(runtime_paths, _config(NEW), AD_HOC.renamed())
    assert (history.scope_id, history.replayed, history.member_responses) == (
        "team_coder+helper",
        ["r2"],
        [[NEW, HELPER]],
    )
    assert not old_store.exists()


def test_no_team_stores_plans_nothing(tmp_path: Path) -> None:
    """An install without team history has nothing to move."""
    assert _plan(test_runtime_paths(tmp_path)) == []
