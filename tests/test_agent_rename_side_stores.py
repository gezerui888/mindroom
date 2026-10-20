"""Tests for moving the tracking records and side stores keyed by an agent's name with its config key."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from agno.run.agent import RunOutput
from agno.run.base import RunStatus
from agno.session.agent import AgentSession

from mindroom.agent_modes import resolve_agent_mode, set_agent_mode
from mindroom.agent_rename.entity_state import entity_state_steps
from mindroom.agent_rename.side_stores import entity_file_steps
from mindroom.agent_rename.steps import (
    BACKUPS_DIRECTORY_NAME,
    DirectoryEdit,
    DropPath,
    FileRewrite,
    MovePath,
    RenameApplyError,
    RenamePlan,
    RenameRefusedError,
    SqliteEdit,
    apply_plan,
)
from mindroom.automations.threads import automations_tracking_root
from mindroom.config.agent import AgentConfig, AgentPrivateConfig
from mindroom.config.knowledge import KnowledgeBaseConfig
from mindroom.config.main import Config
from mindroom.config.models import ModelConfig
from mindroom.constants import resolve_avatar_path, tracking_dir, workspace_avatar_path
from mindroom.custom_tools.matrix_message_idempotency import _digest, _read
from mindroom.custom_tools.todo_poke import _parse_thread_snapshot, _poke_record, _scope_key, _TodoPokeScope
from mindroom.custom_tools.todo_state import locked_update_json, read_json, state_root, todos_path
from mindroom.delegation.records import DelegationMetadata, DelegationRecordOwner
from mindroom.delegation.sessions import SubagentSessionError, load_subagent
from mindroom.delegation.state import DELEGATION_STATE_KEY, DelegationChild, DelegationState
from mindroom.delegation.storage import freeze_delegation_storage
from mindroom.desktop.pairing import DesktopPairingError, claim_desktop_pairing, create_desktop_pairing
from mindroom.durable_write import write_json_file_durable
from mindroom.dynamic_workflows.store import DynamicWorkflowStore
from mindroom.external_triggers.store import ExternalTriggerRecord, ExternalTriggerStore, ExternalTriggerTarget
from mindroom.matrix.invited_rooms_store import invited_rooms_path, load_invited_rooms, save_invited_rooms
from mindroom.matrix.personal_room_store import (
    PersonalRoomRecord,
    personal_room_record_path,
    retained_personal_rooms,
    write_personal_room,
)
from mindroom.matrix.sync_continuity import SyncContinuityStore
from mindroom.mcp_gateway.oauth import GatewayOAuthProvider
from mindroom.mcp_gateway.selection import GatewaySelections
from mindroom.mcp_gateway.types import GatewayOwner
from mindroom.memory.auto_flush import _flush_scopes
from mindroom.oauth.providers import OAuthProviderError
from mindroom.oauth.state import issue_opaque_oauth_state, read_opaque_oauth_state
from mindroom.report_publishing.store import PublishableReport, ReportPublishingStore
from mindroom.script_runs.models import ScriptRunRecord, ScriptRunState, ScriptToolGrant
from mindroom.script_runs.store import ScriptRunStore
from mindroom.session_storage.agent_storage import create_state_storage
from mindroom.shared_directories.store import (
    SharedDirectory,
    add_shared_directory_member,
    create_shared_directory,
    load_shared_directories,
)
from mindroom.thread_models import resolve_thread_model_override, set_thread_model_override
from mindroom.tool_system.worker_routing import (
    ToolExecutionIdentity,
    agent_state_root_path,
    agent_workspace_root_path,
    serialize_tool_execution_identity,
)
from tests.agent_rename_helpers import SQLITE_SIDE_FILES, sqlite_rows, tree_snapshot
from tests.conftest import seed_session, test_runtime_paths

if TYPE_CHECKING:
    from pathlib import Path

    from mindroom.agent_rename.steps import Step
    from mindroom.constants import RuntimePaths
    from mindroom.delegation.records import DelegationRecordHandle

OLD = "code"
NEW = "coder"
# Its name extends the renamed agent's, so a substring match would wrongly touch it.
PREFIX = "code_team"
HELPER = "helper"
ALICE = "@alice:x"
ROOM = "!room:x"
THREAD = "$thread"
NOW = datetime(2026, 10, 20, 9, 30, 0, tzinfo=UTC)
# Desktop pairing tokens live 15 minutes from this creation time.
PAIRED_AT = 1_800_000_000
OWNER = GatewayOwner(ALICE, ALICE)


def _config(agent: str) -> Config:
    """Return the config naming the renamed agent ``agent`` beside ``code_team`` and ``helper``."""
    return Config(
        agents={name: AgentConfig(display_name=name.title()) for name in (agent, PREFIX, HELPER)},
        models={
            "default": ModelConfig(provider="test", id="test-model"),
            "fast": ModelConfig(provider="test", id="fast-model"),
        },
    )


def _identity(agent: str) -> ToolExecutionIdentity:
    return ToolExecutionIdentity(
        channel="matrix",
        agent_name=agent,
        requester_id=ALICE,
        room_id=ROOM,
        thread_id=THREAD,
        resolved_thread_id=THREAD,
        session_id=f"session-{agent}",
    )


def _workflow_spec(workflow_id: str, participant_agent: str) -> dict[str, object]:
    """Return a workflow whose one participant is the configured agent ``participant_agent``."""
    return {
        "schema_version": 1,
        "id": workflow_id,
        "name": f"Workflow {workflow_id}",
        "kind": "workflow",
        "inputs": {"type": "object", "properties": {"topic": {"type": "string"}}},
        "participants": [{"id": "member", "kind": "room_agent", "agent": participant_agent}],
        "workflow": [{"id": "ask", "type": "agent_step", "participant": "member", "prompt": "Answer."}],
        "outputs": [{"id": "report_html", "type": "html_report", "from_step": "ask"}],
        "permissions": {
            "max_runtime_seconds": 600,
            "max_concurrent_agents": 1,
            "max_total_agents": 1,
            "models": [],
            "tools": [],
            "data": {"matrix_history": "none", "attachments": "none", "knowledge_bases": []},
        },
    }


@dataclass(frozen=True)
class _Install:
    """A storage tree holding every side store for ``code``, ``code_team`` and ``helper``, and handles into it."""

    runtime_paths: RuntimePaths
    # Delegation records by (caller, child).
    delegations: dict[tuple[str, str], DelegationRecordHandle]
    pairing_tokens: dict[str, str]
    oauth_tokens: dict[str, str]
    # Subagent handles by (caller, child).
    subagents: dict[tuple[str, str], str]
    # Public report slugs of each workflow owner's run report.
    published: dict[str, str]


def _poke_scope(agent: str, requester_id: str | None) -> _TodoPokeScope:
    return _TodoPokeScope(
        assigned_agent=agent,
        requester_id=requester_id,
        room_id=ROOM,
        thread_id=THREAD,
        actionable_items=(),
        latest_actionable_update=NOW,
        fingerprint="",
    )


def _todo(item_id: str, agent: str) -> dict[str, object]:
    return {
        "id": item_id,
        "title": f"Work for {agent}",
        "status": "open",
        "priority": "medium",
        "depends_on": [],
        "assigned_agent": agent,
        "requester_id": ALICE,
        "created_at": NOW.isoformat(),
        "updated_at": NOW.isoformat(),
        "completed_at": None,
    }


def _trigger(trigger_id: str, *, created_by: str, target: str) -> ExternalTriggerRecord:
    return ExternalTriggerRecord(
        trigger_id=trigger_id,
        uid=f"uid-{trigger_id}",
        version=1,
        auth_epoch=1,
        enabled=True,
        owner_user_id=ALICE,
        created_by_agent_name=created_by,
        created_in_room_id=ROOM,
        target=ExternalTriggerTarget(room_id=ROOM, agent=target),
        auth="capability",
        capability_token_hash="a" * 64,
        delivery_mode="single_use",
        replay_window_seconds=300,
        max_body_bytes=4096,
        created_at=1,
        updated_at=1,
    )


def _subagent(caller: str, child: str, subagent_id: str, config: Config) -> dict[str, object]:
    """Return a subagent handle file as ``reserve_subagent_turn`` writes it."""
    record = DelegationChild(
        delegation_id=f"delegation-{subagent_id}",
        parent_tool_call_id="call",
        caller_agent_name=caller,
        child_agent_name=child,
        task="Investigate",
        session_id=f"session-{child}",
        run_id="run",
        model_name="default",
        depth=1,
        execution_identity=serialize_tool_execution_identity(_identity(child)),
        subagent_id=subagent_id,
        storage_bindings=freeze_delegation_storage(config, [caller, child]),
        status="completed",
    )
    return {"owner": serialize_tool_execution_identity(_identity(caller)), "child": asdict(record)}


async def _write_async_stores(
    runtime_paths: RuntimePaths,
    config: Config,
) -> dict[tuple[str, str], DelegationRecordHandle]:
    """Write a settled delegation record for each pairing of callers and children, and the MCP gateway selections."""
    owner = DelegationRecordOwner(config, runtime_paths)
    handles: dict[tuple[str, str], DelegationRecordHandle] = {}
    for caller, child in ((HELPER, OLD), (OLD, HELPER), (PREFIX, HELPER)):
        metadata = DelegationMetadata(
            caller_agent_name=caller,
            child_agent_name=child,
            requester_id=ALICE,
            parent_run_id="parent-run",
            parent_tool_call_id="parent-call",
            source_room_id=ROOM,
            source_thread_id=THREAD,
            model_name="test-model",
            task=f"{caller} asks {child}",
        )
        handle = await owner.start(
            metadata,
            caller_execution_identity=_identity(caller),
            child_execution_identity=_identity(child),
        )
        await owner.finish(handle, status="completed", output="Done")
        handles[caller, child] = handle
    provider = GatewayOAuthProvider(runtime_paths, public_url="https://example.org")
    await GatewaySelections(provider.store).set(OWNER, {OLD: None, PREFIX: ("calculator",), HELPER: None})
    return handles


def _install(tmp_path: Path) -> _Install:  # noqa: PLR0915
    """Write every side store keyed by an agent's name with MindRoom's own writers, for ``code`` and its neighbors."""
    runtime_paths = test_runtime_paths(tmp_path)
    storage = runtime_paths.storage_root
    config = _config(OLD)
    for agent in (OLD, PREFIX):
        root = agent_state_root_path(storage, agent)
        set_agent_mode(runtime_paths, root, agent, "s1", "minimal")
        save_invited_rooms(invited_rooms_path(runtime_paths, agent), {f"!{agent}-invited:x"})
        write_personal_room(
            personal_room_record_path(runtime_paths, agent, ALICE),
            PersonalRoomRecord(
                user_id=ALICE,
                alias=f"#{agent}-alice:x",
                source_room_id="!lobby:x",
                room_id=f"!{agent}-alice:x",
            ),
        )
        state_dir = automations_tracking_root(runtime_paths) / agent
        state_dir.mkdir(parents=True)
        (state_dir / "dreaming.json").write_text(json.dumps({"latest_run": f"{agent}-run"}))
        SyncContinuityStore(storage, agent).update_join_fences(add=[f"!{agent}-fence:x"])
        (storage / "calls" / agent).mkdir(parents=True)
        (storage / "calls" / agent / "2026-10-20_call.md").write_text(f"# Call with {agent}\n")
        avatar = workspace_avatar_path("agents", agent, runtime_paths)
        avatar.parent.mkdir(parents=True, exist_ok=True)
        avatar.write_bytes(f"png of {agent}".encode())

    set_thread_model_override(
        runtime_paths,
        thread_id="$t1",
        model_name="fast",
        entity_names=[OLD, PREFIX],
        config=config,
    )
    set_thread_model_override(runtime_paths, thread_id="$t2", model_name="fast", entity_names=[PREFIX], config=config)

    flush_keys = {scope.agent_name: scope.key for scope in _flush_scopes(config, runtime_paths, [OLD, PREFIX])}
    conversation = {"updated_at": 5, "marker": [5, "run"], "failures": 0, "failing_since": 0, "retry_at": 0}
    write_json_file_durable(
        tracking_dir(runtime_paths) / "memory_auto_flush.json",
        {
            "started_at": 1,
            "agents": {OLD: 2, PREFIX: 3},
            "scopes": {flush_keys[OLD]: {"s1": conversation}, flush_keys[PREFIX]: {"s2": conversation}},
        },
    )

    create_shared_directory(runtime_paths, owner=OLD, name="notes")
    (storage / "shared_directories" / OLD / "notes" / "plan.md").write_text("plan")
    for owner, name, members in (
        (OLD, "notes", (HELPER, PREFIX)),
        (HELPER, "specs", (OLD, PREFIX)),
        (PREFIX, "drafts", (HELPER,)),
    ):
        if owner != OLD:
            create_shared_directory(runtime_paths, owner=owner, name=name)
        for member in members:
            add_shared_directory_member(runtime_paths, owner=owner, name=name, member=member)

    todo_root = state_root(runtime_paths)
    items = [_todo("a1", OLD), _todo("a2", PREFIX), _todo("a3", HELPER)]
    locked_update_json(
        todos_path(todo_root, ROOM, THREAD),
        lambda data: data.update(room_id=ROOM, thread_id=THREAD, items=items, updated_at=NOW.isoformat()),
    )
    poke = {"last_poked_at": 1.0, "last_fingerprint": "fingerprint", "unchanged_repoke_count": 1}
    scopes = {_scope_key(_poke_scope(agent, requester)): poke for agent in (OLD, PREFIX) for requester in (None, ALICE)}
    locked_update_json(todo_root / "poke_state.json", lambda data: data.update(scopes=scopes))

    store = DynamicWorkflowStore(storage)
    published = {}
    for owner, participant in ((OLD, HELPER), (HELPER, OLD), (PREFIX, HELPER)):
        workflow_id = f"{owner.replace('_', '-')}-flow"
        store.create_workflow(
            spec=_workflow_spec(workflow_id, participant),
            scope="agent",
            owner_id=owner,
            created_by=owner,
        )
        store.update_workflow(
            workflow_id=workflow_id,
            scope="agent",
            owner_id=owner,
            patch={"description": "second revision"},
            updated_by=owner,
            reason="test",
        )
        run = store.start_workflow_run(
            workflow_id=workflow_id,
            scope="agent",
            owner_id=owner,
            input_data={"topic": "x"},
            requested_by=ALICE,
            base_url="https://mindroom.example",
        )
        failed = store.fail_workflow_run(run, error="no input")
        report = PublishableReport(
            source_type="dynamic_workflow_run",
            source={"workflow_id": workflow_id, "run_id": failed.run_id, "scope": "agent"},
            artifact_path=store.run_report_html_artifact_path(failed),
            title=store.run_report_title(failed),
            requested_by=ALICE,
        )
        published[owner] = ReportPublishingStore(storage).publish_report(source=report, published_by=ALICE).slug

    triggers = {
        record.trigger_id: record.model_dump(mode="json")
        for record in (
            _trigger("from-code", created_by=OLD, target=HELPER),
            _trigger("to-code", created_by=HELPER, target=OLD),
            _trigger("from-prefix", created_by=PREFIX, target=PREFIX),
        )
    }
    assert runtime_paths.control_state_root is not None
    write_json_file_durable(
        runtime_paths.control_state_root / "external_triggers" / "triggers.json",
        {"version": 1, "triggers": triggers},
        indent=2,
        sort_keys=True,
    )

    script_runs = ScriptRunStore(runtime_paths)
    for run_id, agent, state in (
        ("run-code", OLD, ScriptRunState.RUNNING),
        ("run-code-done", OLD, ScriptRunState.EXITED),
        ("run-prefix", PREFIX, ScriptRunState.RUNNING),
    ):
        script_runs.create_run(
            ScriptRunRecord(
                run_id=run_id,
                agent_name=agent,
                owner_user_id=ALICE,
                room_id=ROOM,
                source_digest="digest",
                grants=(ScriptToolGrant("website", "read_url"),),
                token_hash=f"token-{run_id}",
            ),
        )
        if state is not ScriptRunState.STARTING:
            script_runs.transition_run(run_id, state=state)

    sends = runtime_paths.control_state_root / "matrix_message_sends"
    sends.mkdir(parents=True)
    for agent in (OLD, PREFIX):
        scope = (ALICE, agent, ROOM)
        (sends / f"{_digest(scope)}.json").write_text(json.dumps({"version": 1, "scope": list(scope), "sends": {}}))

    subagents = {}
    for index, (caller, child) in enumerate(((OLD, HELPER), (HELPER, OLD), (PREFIX, HELPER))):
        subagent_id = f"{index:032x}"
        path = storage / "subagent_sessions" / f"{subagent_id}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(_subagent(caller, child, subagent_id, config)))
        subagents[caller, child] = subagent_id

    pairing_tokens = {
        agent: create_desktop_pairing(runtime_paths, requester_id=ALICE, agent_name=agent, now=PAIRED_AT).token
        for agent in (OLD, PREFIX)
    }
    oauth_tokens = {
        agent: issue_opaque_oauth_state(
            runtime_paths,
            kind="pending_oauth",
            ttl_seconds=600,
            data={"service": "google", "agent_name": agent, "worker_key": f"v1:default:shared:{agent}"},
        )
        for agent in (OLD, PREFIX)
    }
    oauth_tokens["by-key"] = issue_opaque_oauth_state(
        runtime_paths,
        kind="oauth_connect",
        ttl_seconds=600,
        data={"provider": "google", "agent_name": "", "worker_key": f"v1:default:shared:{OLD}"},
    )

    delegations = asyncio.run(_write_async_stores(runtime_paths, config))
    return _Install(runtime_paths, delegations, pairing_tokens, oauth_tokens, subagents, published)


def _plan(runtime_paths: RuntimePaths, *, old: str = OLD, new: str = NEW) -> tuple[list[Step], list[str]]:
    """Plan like `mindroom agents rename`: the state root moves first, then the side stores."""
    steps, reports = entity_file_steps(runtime_paths, config=_config(old), old_name=old, new_name=new)
    return [*entity_state_steps(runtime_paths, old_name=old, new_name=new), *steps], reports


def _apply(runtime_paths: RuntimePaths, steps: list[Step]) -> None:
    apply_plan(
        RenamePlan(storage_root=runtime_paths.storage_root, steps=tuple(steps), reports=()),
        kind="rename",
        now=NOW,
    )


def _renamed(tmp_path: Path) -> tuple[_Install, list[str]]:
    install = _install(tmp_path)
    steps, reports = _plan(install.runtime_paths)
    _apply(install.runtime_paths, steps)
    return install, reports


def _records_dir(runtime_paths: RuntimePaths, agent: str) -> Path:
    return tracking_dir(runtime_paths) / "agents" / agent


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _tree(root: Path) -> dict[str, object]:
    """Map every path under ``root`` to its bytes, or its rows for a SQLite database, which undo restores by rows."""
    tree: dict[str, object] = {}
    for key, value in tree_snapshot(root).items():
        if key.endswith(SQLITE_SIDE_FILES):
            continue
        tree[key] = sqlite_rows(root / key) if key.endswith((".sqlite", ".sqlite3")) else value
    return tree


def _step_paths(step: Step) -> list[Path]:
    if isinstance(step, MovePath):
        return [step.source, step.destination]
    if isinstance(step, SqliteEdit):
        return [step.database]
    if isinstance(step, FileRewrite | DirectoryEdit | DropPath):
        return [step.path]
    msg = f"Unexpected step {step}"
    raise AssertionError(msg)


def test_tracking_records_follow_the_agent(tmp_path: Path) -> None:
    """The agent's room ownership, personal rooms and conversation modes are found under the new name."""
    install, _ = _renamed(tmp_path)
    runtime_paths = install.runtime_paths
    new_root = agent_state_root_path(runtime_paths.storage_root, NEW)

    assert not _records_dir(runtime_paths, OLD).exists()
    assert load_invited_rooms(invited_rooms_path(runtime_paths, NEW)) == {"!code-invited:x"}
    assert retained_personal_rooms(runtime_paths, NEW) == {"!code-alice:x"}
    # The router finds adopted personal rooms of every agent by globbing the records.
    assert retained_personal_rooms(runtime_paths, "router", user_id="@mindroom_router:x") == set()
    assert resolve_agent_mode(runtime_paths, new_root, NEW, "s1") == "minimal"
    modes = _json(_records_dir(runtime_paths, NEW) / "agent_modes.json")
    assert isinstance(modes, dict)
    assert list(modes) == ['["coder","s1"]']
    assert modes['["coder","s1"]']["agent"] == NEW


def test_delegation_records_name_the_new_agent(tmp_path: Path) -> None:
    """Delegation state and its workspace exports name the new agent as caller or child, and keep working."""
    install, _ = _renamed(tmp_path)
    runtime_paths = install.runtime_paths
    storage = runtime_paths.storage_root
    as_child, as_caller = install.delegations[HELPER, OLD], install.delegations[OLD, HELPER]

    child_state = _json(
        _records_dir(runtime_paths, NEW)
        / as_child.state_dir.relative_to(_records_dir(runtime_paths, OLD))
        / "state.json",
    )
    caller_state = _json(as_caller.state_dir / "state.json")
    assert isinstance(child_state, dict)
    assert isinstance(caller_state, dict)
    reference = f"{NEW}:{as_child.scoped_path}"
    assert (child_state["run"]["caller_agent_name"], child_state["run"]["child_agent_name"]) == (HELPER, NEW)
    assert child_state["run"]["record_reference"] == reference
    assert (caller_state["run"]["caller_agent_name"], caller_state["run"]["child_agent_name"]) == (NEW, HELPER)
    assert caller_state["run"]["record_reference"] == f"{HELPER}:{as_caller.scoped_path}"

    # The exports of a settled record are never written again, so they name the new agent only when rewritten.
    child_export = _json(agent_workspace_root_path(storage, NEW) / as_child.scoped_path / "run.json")
    assert isinstance(child_export, dict)
    assert (child_export["child_agent_name"], child_export["record_reference"]) == (NEW, reference)
    receipt = _json(
        as_child.caller_workspace
        / f".mindroom/delegation_receipts/{as_child.locator.started_date}/{as_child.locator.delegation_id}.json",
    )
    assert isinstance(receipt, dict)
    assert (receipt["child_agent_name"], receipt["record_reference"]) == (NEW, reference)
    caller_export = _json(as_caller.child_workspace / as_caller.scoped_path / "run.json")
    assert isinstance(caller_export, dict)
    assert caller_export["caller_agent_name"] == NEW
    caller_receipt = agent_workspace_root_path(storage, NEW) / ".mindroom/delegation_receipts"
    caller_receipt_json = _json(
        caller_receipt / as_caller.locator.started_date / f"{as_caller.locator.delegation_id}.json",
    )
    assert isinstance(caller_receipt_json, dict)
    assert caller_receipt_json["caller_agent_name"] == NEW

    owner = DelegationRecordOwner(_config(NEW), runtime_paths)
    locator = replace(as_child.locator, child_agent_name=NEW, child_execution_identity=_identity(NEW))
    reopened = asyncio.run(owner.reopen(locator))
    assert reopened.state_dir == _records_dir(runtime_paths, NEW) / as_child.state_dir.relative_to(
        _records_dir(runtime_paths, OLD),
    )
    asyncio.run(owner.finish(reopened, status="completed"))
    transcript = (agent_workspace_root_path(storage, NEW) / as_child.scoped_path / "transcript.md").read_text()
    assert f"- Child: {NEW}" in transcript
    assert f"- Record: {reference}" in transcript


def test_tracking_files_name_the_new_agent(tmp_path: Path) -> None:
    """Thread model choices, memory auto-flush progress and shared directories read back under the new name."""
    install, _ = _renamed(tmp_path)
    runtime_paths = install.runtime_paths
    config = _config(NEW)

    assert resolve_thread_model_override(runtime_paths, "$t1", config=config).active == {NEW: "fast", PREFIX: "fast"}
    assert resolve_thread_model_override(runtime_paths, "$t2", config=config).active == {PREFIX: "fast"}

    flush_keys = {scope.agent_name: scope.key for scope in _flush_scopes(config, runtime_paths, [NEW, PREFIX])}
    progress = _json(tracking_dir(runtime_paths) / "memory_auto_flush.json")
    assert isinstance(progress, dict)
    assert progress["agents"] == {NEW: 2, PREFIX: 3}
    assert set(progress["scopes"]) == {flush_keys[NEW], flush_keys[PREFIX]}
    assert list(progress["scopes"][flush_keys[NEW]]) == ["s1"]

    assert load_shared_directories(runtime_paths) == (
        SharedDirectory(owner=PREFIX, name="drafts", members=(HELPER,)),
        SharedDirectory(owner=NEW, name="notes", members=(PREFIX, HELPER)),
        SharedDirectory(owner=HELPER, name="specs", members=(PREFIX, NEW)),
    )
    assert (runtime_paths.storage_root / "shared_directories" / NEW / "notes" / "plan.md").read_text() == "plan"
    assert not (runtime_paths.storage_root / "shared_directories" / OLD).exists()


def test_todos_and_pokes_name_the_new_agent(tmp_path: Path) -> None:
    """Todos assigned to the agent stay assigned to it, and its poke history keeps its cooldown."""
    install, _ = _renamed(tmp_path)
    todo_root = state_root(install.runtime_paths)
    path = todos_path(todo_root, ROOM, THREAD)

    snapshot = _parse_thread_snapshot(read_json(path), path, set())
    assert [(item.item_id, item.assigned_agent) for item in snapshot.items] == [
        ("a1", NEW),
        ("a2", PREFIX),
        ("a3", HELPER),
    ]
    pokes = read_json(todo_root / "poke_state.json")
    for requester in (None, ALICE):
        assert _poke_record(pokes, _poke_scope(NEW, requester)) is not None
        assert _poke_record(pokes, _poke_scope(OLD, requester)) is None
        assert _poke_record(pokes, _poke_scope(PREFIX, requester)) is not None


def test_moved_stores_open_under_the_new_name(tmp_path: Path) -> None:
    """Automation state, join fences, call transcripts and the avatar are where MindRoom looks for the new name."""
    install, reports = _renamed(tmp_path)
    runtime_paths = install.runtime_paths
    storage = runtime_paths.storage_root

    assert _json(automations_tracking_root(runtime_paths) / NEW / "dreaming.json") == {"latest_run": "code-run"}
    assert SyncContinuityStore(storage, NEW).load().pending_join_decrypt_fences == {"!code-fence:x"}
    assert not (storage / "sync_continuity" / f"{OLD}.json").exists()
    assert not (storage / "sync_continuity" / f"{OLD}.json.lock").exists()
    assert (storage / "calls" / NEW / "2026-10-20_call.md").read_text() == "# Call with code\n"
    avatar = resolve_avatar_path("agents", NEW, runtime_paths)
    assert avatar == workspace_avatar_path("agents", NEW, runtime_paths)
    assert avatar.read_bytes() == b"png of code"
    assert not workspace_avatar_path("agents", OLD, runtime_paths).exists()
    assert any(f"{storage / 'calls' / NEW}" in report for report in reports)


def test_dynamic_workflows_follow_the_agent(tmp_path: Path) -> None:
    """The agent's workflows, revisions, runs and reports load for the new name; other workflows name it as participant."""
    install, reports = _renamed(tmp_path)
    store = DynamicWorkflowStore(install.runtime_paths.storage_root)

    assert store.list_workflows(scope="agent", owner_id=OLD) == []
    [summary] = store.list_workflows(scope="agent", owner_id=NEW)
    assert (summary.workflow_id, summary.owner_id, summary.created_by) == ("code-flow", NEW, NEW)
    revisions = store.list_workflow_revisions(workflow_id="code-flow", scope="agent", owner_id=NEW)
    assert revisions == ["000001", "000002"]
    assert (
        install.runtime_paths.storage_root / "dynamic_workflows/agent/coder/code-flow/revisions/000002.yaml"
    ).read_text().count(
        f"updated_by: {NEW}",
    ) == 1
    run_path = next(
        (install.runtime_paths.storage_root / "dynamic_workflows/agent/coder/code-flow/runs").glob("*.json"),
    )
    run = store.get_workflow_run(workflow_id="code-flow", scope="agent", owner_id=NEW, run_id=run_path.stem)
    assert run.owner_id == NEW
    assert run.report_url == f"https://mindroom.example/reports/private/agent/{NEW}/code-flow/{run.run_id}"
    assert store.run_report_html_artifact_path(run).is_file()
    assert store.private_report_html_path(
        scope="agent",
        owner_key=NEW,
        workflow_id="code-flow",
        run_id=run.run_id,
    ).is_file()

    helper_revision = store.load_workflow_revision(
        workflow_id="helper-flow",
        scope="agent",
        owner_id=HELPER,
        revision="000002",
    )
    assert helper_revision["participants"] == [{"id": "member", "kind": "room_agent", "agent": NEW}]
    prefix_revision = store.load_workflow_revision(
        workflow_id="code-team-flow",
        scope="agent",
        owner_id=PREFIX,
        revision="000002",
    )
    assert prefix_revision["participants"] == [{"id": "member", "kind": "room_agent", "agent": HELPER}]
    assert any("/reports/private/agent/code/" in report for report in reports)


def test_hashed_workflow_owner_directory_moves(tmp_path: Path) -> None:
    """A name that is not a valid workflow id keeps its workflows in a hashed directory, which moves to the new hash."""
    runtime_paths = test_runtime_paths(tmp_path)
    store = DynamicWorkflowStore(runtime_paths.storage_root)
    store.create_workflow(spec=_workflow_spec("flow", HELPER), scope="agent", owner_id="Code", created_by="Code")
    [old_dir] = (runtime_paths.storage_root / "dynamic_workflows" / "agent").iterdir()
    assert old_dir.name.startswith("hash_")

    steps, _ = _plan(runtime_paths, old="Code", new="Coder")
    _apply(runtime_paths, steps)

    [summary] = store.list_workflows(scope="agent", owner_id="Coder")
    assert (summary.owner_id, summary.created_by) == ("Coder", "Coder")
    assert not old_dir.exists()


def test_external_triggers_name_the_new_agent(tmp_path: Path) -> None:
    """Triggers created by or targeting the agent read back with the new name."""
    install, _ = _renamed(tmp_path)

    records = {record.trigger_id: record for record in ExternalTriggerStore(install.runtime_paths).list_records()}

    assert {key: (record.created_by_agent_name, record.target.agent) for key, record in records.items()} == {
        "from-code": (NEW, HELPER),
        "to-code": (HELPER, NEW),
        "from-prefix": (PREFIX, PREFIX),
    }


def test_sqlite_stores_name_the_new_agent(tmp_path: Path) -> None:
    """Desktop pairings, background script runs and MCP gateway selections belong to the new name."""
    install = _install(tmp_path)
    runtime_paths = install.runtime_paths
    gateway = runtime_paths.storage_root / "mcp_gateway" / "oauth.sqlite3"

    def accounting() -> tuple[int, int]:
        with closing(sqlite3.connect(gateway)) as connection:
            used = connection.execute("SELECT bytes_used FROM oauth_usage").fetchone()[0]
            selections = connection.execute("SELECT SUM(accounted_bytes) FROM gateway_selections").fetchone()[0]
        return used, selections

    used_before, selections_before = accounting()
    steps, reports = _plan(runtime_paths)
    _apply(runtime_paths, steps)

    claim = {"device_user_id": ALICE, "device_id": "DEVICE", "device_ed25519": "key", "now": PAIRED_AT + 1}
    assert (
        claim_desktop_pairing(runtime_paths, token=install.pairing_tokens[OLD], agent_name=NEW, **claim).agent_name
        == NEW
    )
    with pytest.raises(DesktopPairingError, match="another agent"):
        claim_desktop_pairing(runtime_paths, token=install.pairing_tokens[OLD], agent_name=OLD, **claim)
    assert claim_desktop_pairing(runtime_paths, token=install.pairing_tokens[PREFIX], agent_name=PREFIX, **claim)

    script_runs = ScriptRunStore(runtime_paths)
    assert {run.run_id for run in script_runs.list_runs(agent_name=NEW)} == {"run-code", "run-code-done"}
    assert script_runs.list_runs(agent_name=OLD) == []
    assert [run.run_id for run in script_runs.list_runs(agent_name=PREFIX)] == ["run-prefix"]
    [script_report] = [report for report in reports if "background script" in report]
    assert "run-code" in script_report
    assert "run-code-done" not in script_report
    assert "run-prefix" not in script_report

    provider = GatewayOAuthProvider(runtime_paths, public_url="https://example.org")
    selections = asyncio.run(GatewaySelections(provider.store).get(OWNER, ()))
    assert selections == {NEW: None, PREFIX: ("calculator",), HELPER: None}
    used_after, selections_after = accounting()
    # The renamed selection is one byte longer, and the gateway's storage accounting follows it.
    assert selections_after - selections_before == used_after - used_before == len(NEW) - len(OLD)


def test_side_records_for_the_old_name_are_dropped(tmp_path: Path) -> None:
    """Subagent handles, idempotent send intents and pending OAuth states of the old name go into the backup."""
    install, _ = _renamed(tmp_path)
    runtime_paths = install.runtime_paths
    storage = runtime_paths.storage_root
    config = _config(NEW)

    sessions = storage / "subagent_sessions"
    assert not (sessions / f"{install.subagents[OLD, HELPER]}.json").exists()
    assert not (sessions / f"{install.subagents[HELPER, OLD]}.json").exists()
    prefix_id = install.subagents[PREFIX, HELPER]
    child = asyncio.run(
        load_subagent(prefix_id, owner=_identity(PREFIX), config=config, runtime_paths=runtime_paths, depth=0),
    )
    assert child.caller_agent_name == PREFIX
    with pytest.raises(SubagentSessionError):
        asyncio.run(
            load_subagent(
                install.subagents[OLD, HELPER],
                owner=_identity(NEW),
                config=config,
                runtime_paths=runtime_paths,
                depth=0,
            ),
        )

    assert runtime_paths.control_state_root is not None
    sends = runtime_paths.control_state_root / "matrix_message_sends"
    assert not (sends / f"{_digest((ALICE, OLD, ROOM))}.json").exists()
    prefix_scope = (ALICE, PREFIX, ROOM)
    assert _read(sends / f"{_digest(prefix_scope)}.json", prefix_scope).scope == prefix_scope

    for token in (install.oauth_tokens[OLD], install.oauth_tokens["by-key"]):
        with pytest.raises(OAuthProviderError):
            read_opaque_oauth_state(runtime_paths, kind="pending_oauth", token=token)
    prefix_state = read_opaque_oauth_state(runtime_paths, kind="pending_oauth", token=install.oauth_tokens[PREFIX])
    assert prefix_state["agent_name"] == PREFIX
    assert list((storage / BACKUPS_DIRECTORY_NAME).rglob(f"{install.subagents[OLD, HELPER]}.json"))


def test_published_workflow_reports_follow_the_agent(tmp_path: Path) -> None:
    """Public links to the agent's workflow reports keep serving them from the moved workflow directory."""
    install, _ = _renamed(tmp_path)
    storage = install.runtime_paths.storage_root
    publishing = ReportPublishingStore(storage)

    report = publishing.get_public_report(install.published[OLD])
    assert report.artifact_path.startswith(f"dynamic_workflows/agent/{NEW}/code-flow/artifacts/")
    assert publishing.report_asset_path(report).is_file()
    for owner in (HELPER, PREFIX):
        other = publishing.get_public_report(install.published[owner])
        assert other.artifact_path.startswith(f"dynamic_workflows/agent/{owner}/")
        assert publishing.report_asset_path(other).is_file()


def _paused_delegation(run_id: str, session_id: str, *, caller: str, child: str, status: RunStatus) -> RunOutput:
    """Return a parent run whose saved delegation state, as the delegation driver persists it, waits on a child."""
    config = _config(OLD)
    child_record = DelegationChild(
        delegation_id=f"delegation-{run_id}",
        parent_tool_call_id="call",
        caller_agent_name=caller,
        child_agent_name=child,
        task="Investigate",
        session_id=f"child-{session_id}",
        run_id=f"child-{run_id}",
        model_name="default",
        depth=1,
        execution_identity=serialize_tool_execution_identity(_identity(child)),
        storage_bindings=freeze_delegation_storage(config, [caller, child]),
        status="paused",
    )
    state = DelegationState(children=[child_record], storage_bindings=freeze_delegation_storage(config, [caller]))
    return RunOutput(
        run_id=run_id,
        agent_id=caller,
        session_id=session_id,
        user_id=ALICE,
        status=status,
        metadata={DELEGATION_STATE_KEY: state.to_dict()},
    )


def test_unfinished_delegations_naming_the_agent_are_reported(tmp_path: Path) -> None:
    """Saved delegations that wait on the agent or that it waits on cannot resume, so they are reported."""
    runtime_paths = test_runtime_paths(tmp_path)
    runs = {
        OLD: [_paused_delegation("r1", "s-caller", caller=OLD, child=HELPER, status=RunStatus.paused)],
        HELPER: [
            _paused_delegation("r2", "s-child", caller=HELPER, child=OLD, status=RunStatus.paused),
            _paused_delegation("r3", "s-child", caller=HELPER, child=OLD, status=RunStatus.paused),
            _paused_delegation("r4", "s-done", caller=HELPER, child=OLD, status=RunStatus.completed),
        ],
        PREFIX: [_paused_delegation("r5", "s-prefix", caller=PREFIX, child=HELPER, status=RunStatus.paused)],
    }
    for agent, agent_runs in runs.items():
        storage = create_state_storage(
            agent,
            agent_state_root_path(runtime_paths.storage_root, agent),
            subdir="sessions",
            session_table=f"{agent}_sessions",
        )
        for session_id in sorted({run.session_id for run in agent_runs}):
            session_runs = [run for run in agent_runs if run.session_id == session_id]
            seed_session(storage, AgentSession(session_id=session_id, agent_id=agent, user_id=ALICE, runs=session_runs))

    broken = agent_state_root_path(runtime_paths.storage_root, "broken") / "sessions" / "broken.db"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"not a database")

    _, reports = entity_file_steps(runtime_paths, config=_config(OLD), old_name=OLD, new_name=NEW)

    [report, unreadable] = [report for report in reports if "delegation" in report]
    assert f"could not be read to look for unfinished delegations: {broken}" in unreadable
    assert "3 unfinished delegations" in report
    assert "s-caller, s-child" in report
    assert "s-done" not in report
    assert "s-prefix" not in report


def test_config_paths_into_moved_shared_directories_are_reported(tmp_path: Path) -> None:
    """Config values pointing inside the agent's shared directories are reported, as the config edit keeps them."""
    runtime_paths = test_runtime_paths(tmp_path)
    create_shared_directory(runtime_paths, owner=OLD, name="notes")
    create_shared_directory(runtime_paths, owner=PREFIX, name="drafts")
    config = _config(OLD).model_copy(
        update={
            "knowledge_bases": {
                "notes": KnowledgeBaseConfig(path="${MINDROOM_STORAGE_PATH}/shared_directories/code/notes"),
                "drafts": KnowledgeBaseConfig(path="${MINDROOM_STORAGE_PATH}/shared_directories/code_team/drafts"),
            },
        },
    )

    _, reports = entity_file_steps(runtime_paths, config=config, old_name=OLD, new_name=NEW)

    [report] = [report for report in reports if "shared_directories" in report]
    assert "knowledge_bases.notes.path" in report
    assert "knowledge_bases.drafts.path" not in report


def test_private_delegation_exports_naming_the_agent_are_reported(tmp_path: Path) -> None:
    """Exports in a private child's workspace name the agent as caller; they stay and are reported."""
    runtime_paths = test_runtime_paths(tmp_path)
    config = _config(OLD).model_copy(
        update={
            "agents": {
                **_config(OLD).agents,
                "secret": AgentConfig(
                    display_name="Secret",
                    private=AgentPrivateConfig(per="user", root="secret_data"),
                ),
            },
        },
    )
    metadata = DelegationMetadata(
        caller_agent_name=OLD,
        child_agent_name="secret",
        requester_id=ALICE,
        parent_run_id="parent-run",
        parent_tool_call_id="parent-call",
        source_room_id=ROOM,
        source_thread_id=THREAD,
        model_name="test-model",
        task="code asks secret",
    )
    owner = DelegationRecordOwner(config, runtime_paths)
    handle = asyncio.run(
        owner.start(metadata, caller_execution_identity=_identity(OLD), child_execution_identity=_identity("secret")),
    )
    export = handle.child_workspace / handle.scoped_path / "run.json"
    assert export.is_file()

    steps, reports = entity_file_steps(runtime_paths, config=config, old_name=OLD, new_name=NEW)

    assert not [step for step in steps if isinstance(step, FileRewrite) and step.path == export]
    [report] = [report for report in reports if "private" in report]
    assert str(export) in report


def test_prefix_agent_untouched(tmp_path: Path) -> None:
    """Renaming code leaves every file of code_team byte for byte and plans nothing on its paths."""
    install = _install(tmp_path)
    runtime_paths = install.runtime_paths
    storage = runtime_paths.storage_root
    prefix_paths = [
        _records_dir(runtime_paths, PREFIX),
        automations_tracking_root(runtime_paths) / PREFIX,
        storage / "sync_continuity" / f"{PREFIX}.json",
        storage / "calls" / PREFIX,
        storage / "dynamic_workflows" / "agent" / PREFIX,
        storage / "shared_directories" / PREFIX,
        workspace_avatar_path("agents", PREFIX, runtime_paths),
        install.delegations[PREFIX, HELPER].state_dir,
        storage / "subagent_sessions" / f"{install.subagents[PREFIX, HELPER]}.json",
    ]
    before = {path: tree_snapshot(path) if path.is_dir() else path.read_bytes() for path in prefix_paths}

    steps, _ = _plan(runtime_paths)

    assert not [
        path
        for step in steps
        for path in _step_paths(step)
        if any(path.is_relative_to(prefix) for prefix in prefix_paths)
    ]
    _apply(runtime_paths, steps)
    assert {path: tree_snapshot(path) if path.is_dir() else path.read_bytes() for path in prefix_paths} == before
    assert resolve_agent_mode(runtime_paths, agent_state_root_path(storage, PREFIX), PREFIX, "s1") == "minimal"


@pytest.mark.parametrize(
    ("taken", "refusal"),
    [
        ("tracking", "tracking/agents/coder already exists"),
        ("thread_model", "thread \\$t1 already holds 'model:coder'"),
        ("shared_directory", "already holds 'coder/notes'"),
        ("selection", "MCP gateway selection .* already holds 'coder'"),
    ],
)
def test_new_name_already_taken_refuses(tmp_path: Path, taken: str, refusal: str) -> None:
    """A destination that exists, or a record that already names the new agent beside the old, refuses before writing."""
    install = _install(tmp_path)
    runtime_paths = install.runtime_paths
    if taken == "tracking":
        _records_dir(runtime_paths, NEW).mkdir(parents=True)
    elif taken == "thread_model":
        path = tracking_dir(runtime_paths) / "thread_models.json"
        records = _json(path)
        assert isinstance(records, dict)
        records["$t1"]["model:coder"] = "default"
        path.write_text(json.dumps(records))
    elif taken == "shared_directory":
        path = tracking_dir(runtime_paths) / "shared_directories.json"
        ledger = _json(path)
        assert isinstance(ledger, dict)
        ledger["coder/notes"] = {"owner": NEW, "name": "notes", "members": []}
        path.write_text(json.dumps(ledger))
    else:
        provider = GatewayOAuthProvider(runtime_paths, public_url="https://example.org")
        asyncio.run(GatewaySelections(provider.store).set(OWNER, {OLD: None, NEW: None}))
    before = tree_snapshot(tmp_path)

    with pytest.raises(RenameRefusedError, match=refusal):
        entity_file_steps(runtime_paths, config=_config(OLD), old_name=OLD, new_name=NEW)

    assert tree_snapshot(tmp_path) == before


def test_failure_after_every_step_restores_the_tree(tmp_path: Path) -> None:
    """When the step after the last of them fails, every move, rewrite, edit and drop is undone."""
    install = _install(tmp_path)
    runtime_paths = install.runtime_paths
    sentinel = runtime_paths.storage_root / "sentinel.txt"
    sentinel.write_text("sentinel")
    original = _tree(tmp_path)
    steps, _ = _plan(runtime_paths)
    assert {type(step) for step in steps} == {MovePath, FileRewrite, SqliteEdit, DropPath}

    def fail(content: bytes) -> bytes:  # noqa: ARG001
        msg = "boom"
        raise RuntimeError(msg)

    with pytest.raises(RenameApplyError, match="fail"):
        _apply(runtime_paths, [*steps, FileRewrite(sentinel, fail, "fail")])

    restored = _tree(tmp_path)
    assert {key: value for key, value in restored.items() if BACKUPS_DIRECTORY_NAME not in key} == original


def test_agent_without_side_stores_plans_nothing(tmp_path: Path) -> None:
    """An agent that never wrote a side store gets no steps and no reports."""
    runtime_paths = test_runtime_paths(tmp_path)

    assert entity_file_steps(runtime_paths, config=_config(OLD), old_name=OLD, new_name=NEW) == ([], [])
