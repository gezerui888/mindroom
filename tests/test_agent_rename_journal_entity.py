"""Tests for renaming an agent's journal principal and every journal value keyed by its config name."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import asdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import pytest

from mindroom.agent_rename.journal import journal_entity_steps
from mindroom.agent_rename.steps import RenamePlan, RenameRefusedError, SqliteEdit, apply_plan
from mindroom.delegation.state import DelegationChild
from mindroom.event_journal import (
    DepartureSource,
    EventClass,
    EventJournalStore,
    EventKind,
    InboundEvent,
    IngestionBatchAdmission,
    IngestionRecordAdmission,
    IngestionRecordDisposition,
    PrincipalStore,
    encode_thread_id,
)
from mindroom.event_journal.models import IngestionConsumer
from mindroom.event_journal.opening import event_journal_sqlite_path
from mindroom.handled_turns import HandledTurnLedger, TurnRecord, _reset_handled_turn_ledger_runtime
from mindroom.history.types import HistoryScope
from mindroom.message_target import MessageTarget
from mindroom.tool_jobs.runtime import BackgroundJob
from mindroom.tool_system.worker_routing import ToolExecutionIdentity
from tests.agent_rename_helpers import tree_snapshot

if TYPE_CHECKING:
    from pathlib import Path

OLD = "code"
NEW = "zeta"
# Shares OLD's name as a prefix; renaming OLD must leave all of it alone.
PREFIX = "code_team"
HELPER = "helper"
USER_ID = "@code:x"
# bot.py joins the agent name and the Matrix ID with one "@", and the ID starts with "@".
OLD_PRINCIPAL = "code@@code:x"
NEW_PRINCIPAL = "zeta@@code:x"
PREFIX_PRINCIPAL = "code_team@@code_team:x"
HELPER_PRINCIPAL = "helper@@helper:x"
ROOM = "!room:x"
UNTHREADED = encode_thread_id(None)
NOW = datetime(2026, 10, 20, 12, 0, 0, tzinfo=UTC)
# The scope a requester-partitioned ad-hoc team gets when one member is private (history/team_scope.py).
REQUESTER_SUFFIX = "_requester_0123456789ab"


def _steps(database: Path) -> list[SqliteEdit]:
    return journal_entity_steps(database, old_name=OLD, new_name=NEW, user_id=USER_ID)


def _apply(steps: list[SqliteEdit], storage_root: Path) -> None:
    apply_plan(RenamePlan(storage_root=storage_root, steps=tuple(steps), reports=()), kind="rename", now=NOW)


def _query(database: Path, sql: str, *params: object) -> list[tuple[object, ...]]:
    with closing(sqlite3.connect(database)) as connection:
        return connection.execute(sql, params).fetchall()


def _rows(database: Path) -> dict[str, list[tuple[object, ...]]]:
    """Every row of every table, for comparing a database before and after."""
    with closing(sqlite3.connect(database)) as connection:
        tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        return {table: sorted(connection.execute(f'SELECT * FROM "{table}"'), key=repr) for table in tables}  # noqa: S608


def _holds(row: tuple[object, ...], needle: str) -> bool:
    """Whether a text cell of ``row`` is ``needle`` or holds it as a JSON string."""
    return any(isinstance(value, str) and (value == needle or f'"{needle}"' in value) for value in row)


def _rows_holding(database: Path, *needles: str, without: tuple[str, ...] = ()) -> list[tuple[str, tuple[object, ...]]]:
    """Every row holding one of ``needles`` and none of ``without``."""
    return sorted(
        (table, row)
        for table, rows in _rows(database).items()
        for row in rows
        if any(_holds(row, needle) for needle in needles) and not any(_holds(row, other) for other in without)
    )


def _insert(connection: sqlite3.Connection, table: str, **values: object) -> None:
    columns = ", ".join(values)
    marks = ", ".join("?" for _ in values)
    connection.execute(f"INSERT INTO {table} ({columns}) VALUES ({marks})", tuple(values.values()))  # noqa: S608


def _membership(previous: str | None, current: str, previous_epoch: int, epoch: int) -> IngestionRecordAdmission:
    """One own-membership transition in the room, as the sync stream reports it."""
    return IngestionRecordAdmission(
        IngestionRecordDisposition.ROOM_LIFECYCLE,
        source=DepartureSource.REPORTED,
        room_id=ROOM,
        previous_membership=previous,
        membership=current,
        previous_membership_epoch=previous_epoch,
        membership_epoch=epoch,
    )


def _message(event_id: str) -> IngestionRecordAdmission:
    """One message from alice in the room that the bot may answer."""
    event = InboundEvent(
        event_id=event_id,
        room_id=ROOM,
        thread_id=None,
        kind=EventKind.MESSAGE,
        event_class=EventClass.ACTIONABLE,
        sender="@alice:x",
        origin_server_ts=10,
        source={"event_id": event_id, "sender": "@alice:x"},
    )
    return IngestionRecordAdmission(IngestionRecordDisposition.SEMANTIC_EVENT, event=event)


async def _bound(store: EventJournalStore, principal_id: str, stream: UUID) -> PrincipalStore:
    """Bind ``stream`` to the principal, as a device's first start does."""
    principal = store.principal(principal_id)
    consumer = await principal.load_or_create_ingestion_consumer(new_generation=uuid4())
    await principal.bind_ingestion_stream(generation=consumer.generation, stream_id=stream)
    return principal


async def _synced_journal(database: Path) -> tuple[UUID, UUID]:
    """Sync one batch for the agent and for the prefix agent; return the agent's stream and consumer generation."""
    stream = uuid4()
    store = EventJournalStore.open_sqlite(database)
    try:
        for principal_id, principal_stream in ((OLD_PRINCIPAL, stream), (PREFIX_PRINCIPAL, uuid4())):
            principal = await _bound(store, principal_id, principal_stream)
            batch = (_membership(None, "join", 0, 0), _message("$from-alice"))
            await principal.admit_ingestion_batch(IngestionBatchAdmission(principal_stream, 1, batch))
        consumer = await store.principal(OLD_PRINCIPAL).load_or_create_ingestion_consumer(new_generation=uuid4())
    finally:
        await store.close()
    return stream, consumer.generation


_PRINCIPAL_CELLS = """
    SELECT 'matrix_sync_consumers', principal_id FROM matrix_sync_consumers
    UNION ALL SELECT 'matrix_ingestion_membership', principal_id FROM matrix_ingestion_membership
    UNION ALL SELECT 'journal_events', principal_id FROM journal_events
    UNION ALL SELECT 'room_membership', principal_id FROM room_membership
"""


@pytest.mark.asyncio
async def test_principal_and_its_sync_stream_are_renamed_and_the_stream_continues(tmp_path: Path) -> None:
    """The principal keeps its consumer and producer positions, so the same device's stream goes on admitting.

    The Matrix account and device stay, so the next sync continues the stream at its next sequence; re-delivered
    events deduplicate instead of failing with IngestionBatchIntegrityError.
    """
    database = event_journal_sqlite_path(tmp_path)
    stream, generation = await _synced_journal(database)
    consumer = "SELECT consumer_generation, stream_id, next_sequence FROM matrix_sync_consumers WHERE principal_id = ?"
    producer = "SELECT room_id, membership, membership_epoch FROM matrix_ingestion_membership WHERE principal_id = ?"
    old_consumer, old_producer = _query(database, consumer, OLD_PRINCIPAL), _query(database, producer, OLD_PRINCIPAL)
    prefix_rows = _rows_holding(database, PREFIX_PRINCIPAL)
    assert prefix_rows
    assert old_consumer == [(str(generation), str(stream), 2)]
    assert old_producer == [(ROOM, "join", 0)]

    _apply(_steps(database), tmp_path)

    assert _query(database, consumer, NEW_PRINCIPAL) == old_consumer
    assert _query(database, producer, NEW_PRINCIPAL) == old_producer
    assert _rows_holding(database, OLD_PRINCIPAL) == []
    assert _rows_holding(database, PREFIX_PRINCIPAL) == prefix_rows
    assert {table for table, principal in _query(database, _PRINCIPAL_CELLS) if principal == NEW_PRINCIPAL} == {
        "matrix_sync_consumers",
        "matrix_ingestion_membership",
        "journal_events",
        "room_membership",
    }
    store = EventJournalStore.open_sqlite(database)
    try:
        renamed = store.principal(NEW_PRINCIPAL)
        assert await renamed.load_or_create_ingestion_consumer(new_generation=uuid4()) == IngestionConsumer(
            generation,
            stream,
        )
        await renamed.bind_ingestion_stream(generation=generation, stream_id=stream)
        redelivered = await renamed.admit_ingestion_batch(
            IngestionBatchAdmission(stream, 2, (_message("$from-alice"),)),
        )
        fresh = await renamed.admit_ingestion_batch(IngestionBatchAdmission(stream, 3, (_message("$next"),)))
        history = await renamed.load_event("$from-alice")
    finally:
        await store.close()
    assert redelivered.receipt_new
    assert not redelivered.semantic_event_new
    assert fresh.semantic_event_new
    assert history is not None


def _target(event_id: str) -> MessageTarget:
    return MessageTarget.resolve(ROOM, None, event_id)


async def _record_turn(store: EventJournalStore, agent: str, event_id: str, scope: HistoryScope) -> None:
    """Record ``agent``'s answer to ``event_id`` the way the bot does, through its handled-turn ledger."""
    ledger = HandledTurnLedger(agent, records=store.turn_records(agent))
    await ledger.load()
    await ledger.record_handled_turn(
        TurnRecord.create(
            [event_id],
            response_event_id=f"$answer-{agent}-{event_id}",
            response_owner=agent,
            history_scope=scope,
            conversation_target=_target(event_id),
        ),
    )


async def _turn_record(database: Path, agent: str, event_id: str) -> TurnRecord | None:
    """Read ``agent``'s record for ``event_id`` through a freshly loaded handled-turn ledger."""
    _reset_handled_turn_ledger_runtime()
    store = EventJournalStore.open_sqlite(database)
    try:
        ledger = HandledTurnLedger(agent, records=store.turn_records(agent))
        await ledger.load()
        return ledger.get_turn_record(event_id) if ledger.has_responded(event_id) else None
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.ledger_loads_from_disk
async def test_turn_handled_under_the_old_name_is_handled_under_the_new_name(tmp_path: Path) -> None:
    """The renamed agent's ledger finds the turns it answered as OLD, owned by it and in its renamed scope."""
    database = event_journal_sqlite_path(tmp_path)
    store = EventJournalStore.open_sqlite(database)
    try:
        await _record_turn(store, OLD, "$from-alice", HistoryScope(kind="agent", scope_id=OLD))
        await _record_turn(store, PREFIX, "$from-alice", HistoryScope(kind="agent", scope_id=PREFIX))
    finally:
        await store.close()
    prefix_record = await _turn_record(database, PREFIX, "$from-alice")

    _apply(_steps(database), tmp_path)

    renamed = await _turn_record(database, NEW, "$from-alice")
    assert renamed is not None
    assert renamed.response_owner == NEW
    assert renamed.history_scope == HistoryScope(kind="agent", scope_id=NEW)
    assert renamed.response_event_id == f"$answer-{OLD}-$from-alice"
    assert await _turn_record(database, OLD, "$from-alice") is None
    assert await _turn_record(database, PREFIX, "$from-alice") == prefix_record


@pytest.mark.asyncio
@pytest.mark.ledger_loads_from_disk
async def test_ad_hoc_team_scopes_take_the_recomputed_id(tmp_path: Path) -> None:
    """An ad-hoc team scope with the agent gets the id of the renamed member set, sorted again, suffix kept.

    The member that answered keeps its own name. A configured team's own scope is its name, even one shaped like
    an ad-hoc id, and a scope naming the prefix agent is a different member set.
    """
    team_name = f"team_{OLD}"
    scopes = {
        "$team": (HELPER, f"team_{OLD}+{HELPER}"),
        "$private-team": (HELPER, f"team_{OLD}+{HELPER}{REQUESTER_SUFFIX}"),
        "$prefix-team": (HELPER, f"team_{PREFIX}+{HELPER}"),
        "$configured-team": (team_name, team_name),
    }
    database = event_journal_sqlite_path(tmp_path)
    store = EventJournalStore.open_sqlite(database)
    try:
        for event_id, (agent, scope_id) in scopes.items():
            await _record_turn(store, agent, event_id, HistoryScope(kind="team", scope_id=scope_id))
    finally:
        await store.close()

    _apply(_steps(database), tmp_path)

    renamed_scopes = {}
    for event_id, (agent, _scope_id) in scopes.items():
        record = await _turn_record(database, agent, event_id)
        assert record is not None
        assert record.response_owner == agent
        renamed_scopes[event_id] = record.history_scope
    # `zeta` sorts after `helper`, so the members are sorted again.
    assert renamed_scopes == {
        "$team": HistoryScope(kind="team", scope_id=f"team_{HELPER}+{NEW}"),
        "$private-team": HistoryScope(kind="team", scope_id=f"team_{HELPER}+{NEW}{REQUESTER_SUFFIX}"),
        "$prefix-team": HistoryScope(kind="team", scope_id=f"team_{PREFIX}+{HELPER}"),
        "$configured-team": HistoryScope(kind="team", scope_id=team_name),
    }


def _identity(agent: str, *, transport: str | None = None) -> ToolExecutionIdentity:
    return ToolExecutionIdentity(
        channel="matrix",
        agent_name=agent,
        requester_id="@alice:x",
        room_id=ROOM,
        thread_id=None,
        resolved_thread_id=None,
        session_id="session-1",
        transport_agent_name=transport,
    )


def _bindings(*agents: str) -> dict[str, dict[str, object]]:
    """Delegation storage bindings as delegation/storage.py freezes them."""
    return {agent: {"display_name": agent, "private": None, "worker_scope": None} for agent in agents}


def _context(*, history_scope: dict[str, str], identity: ToolExecutionIdentity) -> str:
    """The parts of an approval continuation's context that name agents, encoded like approval_continuations.py."""
    context = {
        "cli_call": {
            "kind": "agent_cli",
            "requirements": [{"member_agent_id": OLD}, {"member_agent_id": PREFIX}],
            "external_requirement": {"member_agent_id": OLD},
        },
        "delegation_storage_bindings": _bindings(OLD, PREFIX),
        "entity_kind": "team",
        "execution_identity": asdict(identity),
        "history_scope": history_scope,
        "mentioned_agents": [OLD, PREFIX],
        "origin": {"requester_entity_name": None, "requester_id": "@alice:x", "sender_entity_name": OLD},
        "request_body": f"ask {OLD} and {PREFIX}",
        "team_member_model_names": [[OLD, "model-a"], [PREFIX, "model-b"]],
        "team_member_names": [OLD, HELPER, PREFIX],
    }
    return json.dumps(context, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _continuation(connection: sqlite3.Connection, principal: str, approval_id: str, context_json: str) -> None:
    _insert(
        connection,
        "approval_continuations",
        principal_id=principal,
        approval_id=approval_id,
        state="waiting",
        context_json=context_json,
        created_at_ns=1,
    )


def _continuation_call(connection: sqlite3.Connection, principal: str, approval_id: str, invoking_agent: str) -> None:
    _insert(
        connection,
        "approval_continuation_calls",
        principal_id=principal,
        approval_id=approval_id,
        generation=0,
        tool_call_id=f"call-{invoking_agent}",
        call_ordinal={OLD: 0, PREFIX: 1}.get(invoking_agent, 2),
        tool_name="shell",
        invoking_agent=invoking_agent,
        expires_at_ns=1,
    )


def _grant(connection: sqlite3.Connection, principal: str, grant_id: str, *, agent: str, responder: str) -> None:
    _insert(
        connection,
        "approval_grants",
        principal_id=principal,
        grant_id=grant_id,
        scope_key=f"scope-{grant_id}",
        room_id=ROOM,
        thread_id="$thread",
        card_event_id=f"$card-{grant_id}",
        requester_id="@alice:x",
        entity_name=agent,
        invoking_agent=agent,
        operation="shell:run",
        responder_principal_id=responder,
        responder_epoch=0,
        membership_epoch=0,
        expires_at_ns=1,
        resolution_json="{}",
        original_delivery_id=f"$delivery-{grant_id}",
    )


def _grant_card(connection: sqlite3.Connection, principal: str, agent: str) -> None:
    _insert(
        connection,
        "approval_grant_cards",
        principal_id=principal,
        continuation_id="continuation-1",
        continuation_generation=0,
        tool_call_id="call-1",
        delivery_id=f"$card-delivery-{agent}",
        scope_key=f"scope-card-{agent}",
        room_id=ROOM,
        thread_id="$thread",
        requester_id="@alice:x",
        entity_name=agent,
        invoking_agent=agent,
        operation="shell:run",
        responder_principal_id=principal,
        responder_epoch=0,
        membership_epoch=0,
    )


def _reply(connection: sqlite3.Connection, principal: str, agent: str) -> None:
    _insert(
        connection,
        "reply_messages",
        principal_id=principal,
        reply_id=f"reply-{agent}",
        entity_name=agent,
        room_id=ROOM,
        membership_epoch=0,
        state="completed",
        last_span_id="span-1",
        presentation_json="{}",
        revision=1,
        placeholder_only=False,
        reply_sequence=1,
        created_at_ns=1,
        updated_at_ns=1,
    )


def _scheduled_call(connection: sqlite3.Connection, principal: str, agent: str) -> None:
    _insert(
        connection,
        "scheduled_call_approvals",
        principal_id=principal,
        task_id=f"task-{agent}",
        delivery_id=f"$scheduled-{agent}",
        room_id=ROOM,
        thread_id="$thread",
        requester_id="@alice:x",
        agent_name=agent,
        toolkit_name="shell",
        tool_name="run",
        arguments_json="{}",
        workflow_digest="digest",
        execute_at_ns=1,
        membership_epoch=0,
    )


def _job(connection: sqlite3.Connection, job: BackgroundJob) -> None:
    # Saved like tool_jobs/runtime.py's _encoded.
    _insert(connection, "tool_jobs", job_id=job.job_id, job_json=json.dumps(asdict(job)))


def _delegation_job(job_id: str, *, caller: str, child: str) -> BackgroundJob:
    """A background delegation job snapshotting its child like delegation/background.py does."""
    snapshot = DelegationChild(
        delegation_id=job_id,
        parent_tool_call_id="call-1",
        caller_agent_name=caller,
        child_agent_name=child,
        task=f"ask {child}",
        session_id="session-1",
        run_id="run-1",
        model_name="model-a",
        depth=1,
        execution_identity=asdict(_identity(child)),
        storage_bindings=_bindings(caller, child),
    )
    return BackgroundJob(
        job_id=job_id,
        owner=_identity(caller),
        tool_name=child,
        depth=0,
        kind="delegation",
        adapter={"child": asdict(snapshot)},
    )


async def _approval_journal(storage_root: Path) -> Path:
    """Build a journal with the real schema holding approval, reply and job rows for the agent and the prefix agent."""
    database = event_journal_sqlite_path(storage_root)
    await EventJournalStore.open_sqlite(database).close()
    agent_scope = {"kind": "agent", "scope_id": OLD}
    team_scope = {"kind": "team", "scope_id": f"team_{OLD}+{HELPER}"}
    with closing(sqlite3.connect(database)) as connection, connection:
        _continuation(connection, OLD_PRINCIPAL, "own", _context(history_scope=agent_scope, identity=_identity(OLD)))
        # helper answered for an ad-hoc team with the agent, and the agent's team call waits for approval.
        team_identity = _identity(OLD, transport=HELPER)
        _continuation(connection, HELPER_PRINCIPAL, "team", _context(history_scope=team_scope, identity=team_identity))
        _continuation_call(connection, HELPER_PRINCIPAL, "team", OLD)
        _continuation_call(connection, HELPER_PRINCIPAL, "team", PREFIX)
        prefix_scope = {"kind": "agent", "scope_id": PREFIX}
        prefix_context = _context(history_scope=prefix_scope, identity=_identity(PREFIX)).replace(f'"{OLD}"', '"x"')
        _continuation(connection, PREFIX_PRINCIPAL, "prefix", prefix_context)
        for principal, agent in ((OLD_PRINCIPAL, OLD), (PREFIX_PRINCIPAL, PREFIX)):
            _reply(connection, principal, agent)
            _scheduled_call(connection, principal, agent)
            _grant_card(connection, principal, agent)
        _grant(connection, OLD_PRINCIPAL, "own-grant", agent=OLD, responder=OLD_PRINCIPAL)
        _grant(connection, HELPER_PRINCIPAL, "team-grant-for-code", agent=OLD, responder=HELPER_PRINCIPAL)
        _grant(connection, HELPER_PRINCIPAL, "helper-grant", agent=HELPER, responder=HELPER_PRINCIPAL)
        _grant(connection, PREFIX_PRINCIPAL, "prefix-grant", agent=PREFIX, responder=PREFIX_PRINCIPAL)
        _insert(connection, "approval_grant_locks", principal_id=OLD_PRINCIPAL)
        _job(connection, BackgroundJob(job_id="own-job", owner=_identity(OLD), tool_name="shell", depth=0))
        _job(
            connection,
            BackgroundJob(job_id="team-job", owner=_identity(OLD, transport=HELPER), tool_name="shell", depth=0),
        )
        _job(connection, _delegation_job("delegation-job", caller=HELPER, child=OLD))
        _job(connection, _delegation_job("prefix-job", caller=HELPER, child=PREFIX))
    return database


def _without_times(job: dict[str, Any]) -> dict[str, Any]:
    """A saved job without the timestamps it takes when created."""
    return {key: value for key, value in job.items() if key not in {"created_at", "updated_at"}}


def _json_column(database: Path, sql: str, *params: object) -> dict[str, object]:
    ((text,),) = _query(database, sql, *params)
    assert isinstance(text, str)
    return json.loads(text)


@pytest.mark.asyncio
async def test_approval_reply_and_job_rows_follow_the_agent(tmp_path: Path) -> None:
    """Every name column and agent field of the approval, reply and job rows takes the new name; grants go."""
    database = await _approval_journal(tmp_path)
    # Rows naming the agent as well change; every other row naming the prefix agent stays exactly as it is.
    prefix_rows = _rows_holding(database, PREFIX, PREFIX_PRINCIPAL, without=(OLD, OLD_PRINCIPAL))
    assert len(prefix_rows) == 7

    _apply(_steps(database), tmp_path)

    assert _query(database, "SELECT principal_id, entity_name FROM reply_messages ORDER BY entity_name") == [
        (PREFIX_PRINCIPAL, PREFIX),
        (NEW_PRINCIPAL, NEW),
    ]
    assert _query(database, "SELECT principal_id, agent_name FROM scheduled_call_approvals ORDER BY agent_name") == [
        (PREFIX_PRINCIPAL, PREFIX),
        (NEW_PRINCIPAL, NEW),
    ]
    assert _query(database, "SELECT invoking_agent FROM approval_continuation_calls ORDER BY call_ordinal") == [
        (NEW,),
        (PREFIX,),
    ]
    # The card keeps its scope key: the grant a click on it creates still covers the calls it was posted for.
    card = (
        "SELECT principal_id, responder_principal_id, entity_name, invoking_agent, scope_key FROM approval_grant_cards"
    )
    assert sorted(_query(database, card)) == [
        (PREFIX_PRINCIPAL, PREFIX_PRINCIPAL, PREFIX, PREFIX, f"scope-card-{PREFIX}"),
        (NEW_PRINCIPAL, NEW_PRINCIPAL, NEW, NEW, f"scope-card-{OLD}"),
    ]
    # A grant's scope key hashes the invoking agent and the responder principal, so no grant naming either matches again.
    assert _query(database, "SELECT grant_id FROM approval_grants ORDER BY grant_id") == [
        ("helper-grant",),
        ("prefix-grant",),
    ]
    assert _query(database, "SELECT principal_id FROM approval_grant_locks") == [(NEW_PRINCIPAL,)]

    own = _json_column(
        database,
        "SELECT context_json FROM approval_continuations WHERE principal_id = ?",
        NEW_PRINCIPAL,
    )
    assert own["history_scope"] == {"kind": "agent", "scope_id": NEW}
    team = _json_column(database, "SELECT context_json FROM approval_continuations WHERE approval_id = 'team'")
    assert team == {
        "cli_call": {
            "kind": "agent_cli",
            "requirements": [{"member_agent_id": NEW}, {"member_agent_id": PREFIX}],
            "external_requirement": {"member_agent_id": NEW},
        },
        "delegation_storage_bindings": {**_bindings(NEW), **_bindings(PREFIX)},
        "entity_kind": "team",
        "execution_identity": asdict(_identity(NEW, transport=HELPER)),
        "history_scope": {"kind": "team", "scope_id": f"team_{HELPER}+{NEW}"},
        "mentioned_agents": [NEW, PREFIX],
        "origin": {"requester_entity_name": None, "requester_id": "@alice:x", "sender_entity_name": NEW},
        "request_body": f"ask {OLD} and {PREFIX}",
        "team_member_model_names": [[NEW, "model-a"], [PREFIX, "model-b"]],
        "team_member_names": [NEW, HELPER, PREFIX],
    }

    jobs = {
        job_id: _without_times(json.loads(str(text)))
        for job_id, text in _query(database, "SELECT job_id, job_json FROM tool_jobs")
    }
    assert jobs["own-job"]["owner"] == asdict(_identity(NEW))
    assert jobs["team-job"]["owner"] == asdict(_identity(NEW, transport=HELPER))
    delegation = _without_times(asdict(_delegation_job("delegation-job", caller=HELPER, child=NEW)))
    # The task is prose the requester wrote, never rewritten.
    delegation["adapter"]["child"]["task"] = f"ask {OLD}"
    assert jobs["delegation-job"] == delegation

    assert _rows_holding(database, OLD, OLD_PRINCIPAL) == []
    assert set(prefix_rows) <= set(_rows_holding(database, PREFIX, PREFIX_PRINCIPAL))


@pytest.mark.asyncio
@pytest.mark.ledger_loads_from_disk
async def test_journal_naming_only_other_agents_needs_no_step(tmp_path: Path) -> None:
    """A journal without rows for the agent, nor values naming it, is left alone; so is a missing journal."""
    assert _steps(event_journal_sqlite_path(tmp_path)) == []
    database = event_journal_sqlite_path(tmp_path)
    store = EventJournalStore.open_sqlite(database)
    try:
        await _bound(store, PREFIX_PRINCIPAL, uuid4())
        await _record_turn(store, PREFIX, "$from-alice", HistoryScope(kind="team", scope_id=f"team_{PREFIX}+{HELPER}"))
    finally:
        await store.close()

    assert _steps(database) == []


@pytest.mark.asyncio
async def test_planning_writes_nothing_and_describes_the_rename(tmp_path: Path) -> None:
    """Building the step only reads, and its description counts what it changes."""
    database = await _approval_journal(tmp_path)
    _stream, _generation = await _synced_journal(database)
    before = tree_snapshot(tmp_path)

    (step,) = _steps(database)

    assert tree_snapshot(tmp_path) == before
    assert step.description.startswith(f"Rewrite the event journal {database}: rename principal {OLD_PRINCIPAL} to ")
    assert "matrix_ingestion_membership: 1, matrix_sync_consumers: 1" in step.description
    # The principal's grants are dropped, not renamed.
    assert "approval_grants:" not in step.description
    assert "approval_continuations.context_json (2)" in step.description
    assert "drop the approval grants bound to agent code (2)" in step.description


@pytest.mark.asyncio
async def test_rows_already_under_the_new_principal_refuse(tmp_path: Path) -> None:
    """A principal of the new name, under any Matrix ID, means the name already has journal state."""
    database = event_journal_sqlite_path(tmp_path)
    await _synced_journal(database)
    store = EventJournalStore.open_sqlite(database)
    try:
        await _bound(store, "zeta@@zeta:x", uuid4())
    finally:
        await store.close()
    before = _rows(database)

    with pytest.raises(RenameRefusedError, match=r"already has rows for agent zeta \(matrix_sync_consumers: 1\)"):
        _steps(database)

    assert _rows(database) == before


@pytest.mark.asyncio
@pytest.mark.ledger_loads_from_disk
async def test_turn_records_already_under_the_new_name_refuse(tmp_path: Path) -> None:
    """Turn records of the new name mean an agent of that name already answered messages."""
    database = event_journal_sqlite_path(tmp_path)
    store = EventJournalStore.open_sqlite(database)
    try:
        await _record_turn(store, OLD, "$from-alice", HistoryScope(kind="agent", scope_id=OLD))
        await _record_turn(store, NEW, "$other", HistoryScope(kind="agent", scope_id=NEW))
    finally:
        await store.close()

    with pytest.raises(RenameRefusedError, match=r"already has rows for agent zeta \(turn_records: 1\)"):
        _steps(database)


@pytest.mark.asyncio
async def test_new_principal_rows_written_after_planning_refuse_at_apply(tmp_path: Path) -> None:
    """The edit checks the new name again in its transaction and leaves the journal unchanged."""
    database = event_journal_sqlite_path(tmp_path)
    await _synced_journal(database)
    steps = _steps(database)
    store = EventJournalStore.open_sqlite(database)
    try:
        await _bound(store, NEW_PRINCIPAL.replace(USER_ID, "@zeta:x"), uuid4())
    finally:
        await store.close()
    before = _rows(database)

    with pytest.raises(RenameRefusedError, match="already has rows for agent zeta"):
        _apply(steps, tmp_path)

    assert _rows(database) == before


@pytest.mark.asyncio
@pytest.mark.ledger_loads_from_disk
async def test_team_scope_listing_both_names_refuses(tmp_path: Path) -> None:
    """Renaming a member of a scope that already lists the new name would merge two member sets."""
    database = event_journal_sqlite_path(tmp_path)
    store = EventJournalStore.open_sqlite(database)
    try:
        await _record_turn(store, HELPER, "$team", HistoryScope(kind="team", scope_id=f"team_{OLD}+{HELPER}+{NEW}"))
    finally:
        await store.close()

    with pytest.raises(RenameRefusedError, match=r"zeta is already one of its members"):
        _steps(database)
