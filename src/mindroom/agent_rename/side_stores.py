"""Plan moving the tracking records and side stores keyed by an agent's name when its config key is renamed.

Tracking records, side stores and their JSON and SQLite contents spell the agent's name as a path component,
a mapping key or a field value. After the rename MindRoom looks for all of them under the new name, so each moves
there and its contents follow; records that cannot follow, because they are bound to the old name's storage, are
dropped into the backup, and state that stays behind is reported. Names match exactly, never as substrings,
so renaming ``code`` leaves ``code_team`` alone.
Building the steps only reads; every refusal happens here, before anything is written.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

import yaml

from mindroom import yaml_io
from mindroom.agent_rename.entity_files import rewrite_worker_key_agent
from mindroom.agent_rename.sqlite_read import read_only_connection
from mindroom.agent_rename.steps import DropPath, FileRewrite, RenameRefusedError, SqliteEdit
from mindroom.agent_rename.workers import move_steps
from mindroom.automations.threads import automations_tracking_root
from mindroom.constants import primary_records_dir, tracking_dir, workspace_avatar_path
from mindroom.custom_tools.todo_state import state_root as todo_state_root
from mindroom.dynamic_workflows.store import owner_dir_name
from mindroom.script_runs.models import ScriptRunState
from mindroom.tool_system.worker_routing import (
    SHARED_DIRECTORIES_DIRNAME,
    agent_state_root_path,
    agent_workspace_root_path,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from mindroom.agent_rename.steps import Step
    from mindroom.config.main import Config
    from mindroom.constants import RuntimePaths


# The spellings below copy private names of the modules that own each store; the comments say where they come from.
# thread_models.py's _ENTITY_MODEL_PREFIX: each entity's model choice is one field of a thread's record.
_ENTITY_MODEL_PREFIX = "model:"
# delegation/records.py's _DELEGATION_DIRECTORY and _RECEIPT_DIRECTORY, below each workspace.
_DELEGATION_EXPORTS = (".mindroom/delegations/*/*/run.json", ".mindroom/delegation_receipts/*/*.json")
# dynamic_workflows/store.py keeps workflows in dynamic_workflows/<scope>/<owner dir>/<workflow id>/.
_WORKFLOWS_DIRNAME = "dynamic_workflows"
_AGENT_WORKFLOW_SCOPE = "agent"
# The run states of a background script that has not finished (script_runs/store.py's _TERMINAL_RUN_STATES).
_UNFINISHED_SCRIPT_STATES = (ScriptRunState.STARTING.value, ScriptRunState.RUNNING.value)


@dataclass(frozen=True)
class _NameRename:
    old: str
    new: str

    def value(self, value: object) -> object:
        """Return the new name for a value equal to the old name, and any other value unchanged."""
        return self.new if value == self.old else value


@dataclass(frozen=True)
class _Codec:
    """How MindRoom reads and writes one store's file, so a rewrite keeps its format."""

    load: Callable[[bytes], object]
    dump: Callable[[object], bytes]


def _json_codec(
    *,
    indent: int | None = None,
    sort_keys: bool = False,
    compact: bool = False,
    newline: bool = False,
) -> _Codec:
    separators = (",", ":") if compact else None

    def dump(value: object) -> bytes:
        text = json.dumps(value, indent=indent, sort_keys=sort_keys, separators=separators)
        return (text + "\n" if newline else text).encode()

    return _Codec(json.loads, dump)


# Written like durable_write.py's write_json_file_durable with its defaults.
_JSON = _json_codec()
# Written like write_json_file_durable(indent=2, sort_keys=True): override records, ledgers and trigger records.
_JSON_RECORDS = _json_codec(indent=2, sort_keys=True)
# Written like custom_tools/todo_state.py's locked_update_json and dynamic workflow runs.
_JSON_LINES = _json_codec(indent=2, sort_keys=True, newline=True)
# Written like delegation/records.py's _json_bytes.
_JSON_EXPORT = _json_codec(sort_keys=True, compact=True, newline=True)
# Written like oauth/state.py's _save_state_store.
_JSON_SORTED = _json_codec(sort_keys=True)
# Written like dynamic_workflows/store.py's _atomic_write_yaml.
_YAML = _Codec(yaml_io.safe_load, lambda value: yaml_io.safe_dump(value, sort_keys=False, allow_unicode=True).encode())


def _rewrite(content: bytes, *, codec: _Codec, transform: Callable[[object], object]) -> bytes:
    return codec.dump(transform(codec.load(content)))


def _rewrite_steps(
    source: Path,
    destination: Path,
    codec: _Codec,
    transform: Callable[[object], object],
    description: str,
) -> list[Step]:
    """Return the step that rewrites ``destination`` with ``transform``, when that changes what ``source`` holds now.

    ``destination`` is where ``source`` is once the moves before it ran. A missing or unreadable file gets no step,
    as MindRoom skips such files too.
    """
    try:
        content = codec.load(source.read_bytes())
    except (OSError, ValueError, yaml.YAMLError):
        return []
    if transform(content) == content:
        return []
    return [FileRewrite(destination, partial(_rewrite, codec=codec, transform=transform), description)]


def _object(value: object) -> dict[str, object] | None:
    """Return ``value`` as a JSON object, or ``None`` for anything else."""
    return cast("dict[str, object]", value) if isinstance(value, dict) else None


def _field(value: object, name: str) -> object:
    """Return the ``name`` member of a JSON object, or ``None`` for anything else."""
    record = _object(value)
    return None if record is None else record.get(name)


def _with_names(value: object, rename: _NameRename, fields: tuple[str, ...]) -> object:
    """Return a JSON object with each of ``fields`` that holds the old name holding the new one; anything else as is."""
    record = _object(value)
    if record is None:
        return value
    return {key: rename.value(item) if key in fields else item for key, item in record.items()}


def _rename_keys(mapping: dict[str, object], renamed_key: Callable[[str], str | None], where: str) -> dict[str, object]:
    """Return ``mapping`` with each key that ``renamed_key`` changes renamed, keeping the entries' order.

    Refuses when the new key is already taken, since the two entries cannot both keep it.
    """
    renamed: dict[str, object] = {}
    for key, value in mapping.items():
        new_key = renamed_key(key)
        if new_key is not None and new_key in mapping:
            msg = f"{where} already holds {new_key!r}, so {key!r} cannot be renamed to it."
            raise RenameRefusedError(msg)
        renamed[key if new_key is None else new_key] = value
    return renamed


def _renamed_list_key(key: str, rename: _NameRename) -> str | None:
    """Rename a key spelling a JSON list that starts with the agent, like agent modes and todo pokes use."""
    try:
        parts = json.loads(key)
    except ValueError:
        return None
    if not isinstance(parts, list) or not parts or parts[0] != rename.old:
        return None
    # Spelled like agent_modes.py's _key and custom_tools/todo_poke.py's _scope_key.
    return json.dumps([rename.new, *parts[1:]], separators=(",", ":"))


def _agent_modes(records: object, *, rename: _NameRename, where: str) -> object:
    """Rename the agent in agent_modes.py's records: keys ``["<agent>","<session>"]`` and field ``agent``."""
    mapping = _object(records)
    if mapping is None:
        return records
    renamed = _rename_keys(mapping, partial(_renamed_list_key, rename=rename), where)
    return {key: _with_names(record, rename, ("agent",)) for key, record in renamed.items()}


def _thread_models(records: object, *, rename: _NameRename, where: str) -> object:
    """Rename each thread's ``model:<agent>`` field."""
    mapping = _object(records)
    if mapping is None:
        return records
    old_field, new_field = f"{_ENTITY_MODEL_PREFIX}{rename.old}", f"{_ENTITY_MODEL_PREFIX}{rename.new}"
    renamed: dict[str, object] = {}
    for thread, value in mapping.items():
        record = _object(value)
        renamed[thread] = (
            value
            if record is None
            else _rename_keys(
                record,
                lambda field: new_field if field == old_field else None,
                f"{where} thread {thread}",
            )
        )
    return renamed


def _auto_flush(state: object, *, rename: _NameRename, where: str, old_scope: str, new_scope: str) -> object:
    """Rename the agent's ``agents`` entry and its state root's ``scopes`` entry in memory auto-flush progress."""
    progress = _object(state)
    if progress is None:
        return state
    renamed = dict(progress)
    agents, scopes = _object(progress.get("agents")), _object(progress.get("scopes"))
    if agents is not None:
        renamed["agents"] = _rename_keys(
            agents,
            lambda key: rename.new if key == rename.old else None,
            f"{where} agents",
        )
    if scopes is not None:
        renamed["scopes"] = _rename_keys(scopes, lambda key: new_scope if key == old_scope else None, f"{where} scopes")
    return renamed


def _shared_directories(ledger: object, *, rename: _NameRename, where: str) -> object:
    """Rename the agent as owner (keys ``<owner>/<name>`` and field ``owner``) and as member of shared directories."""
    mapping = _object(ledger)
    if mapping is None:
        return ledger

    def renamed_key(key: str) -> str | None:
        owner, separator, name = key.partition("/")
        return f"{rename.new}/{name}" if separator and owner == rename.old else None

    renamed = _rename_keys(mapping, renamed_key, where)
    return {key: _shared_directory(record, rename=rename, where=f"{where} {key}") for key, record in renamed.items()}


def _shared_directory(value: object, *, rename: _NameRename, where: str) -> object:
    record = _object(_with_names(value, rename, ("owner",)))
    members = _field(value, "members")
    if record is None or not isinstance(members, list) or rename.old not in members:
        return value if record is None else record
    if rename.new in members:
        msg = f"{where} already has member {rename.new!r} beside {rename.old!r}."
        raise RenameRefusedError(msg)
    return {**record, "members": [rename.value(member) for member in members]}


def _todos(thread: object, *, rename: _NameRename) -> object:
    """Rename the agent as ``assigned_agent`` of a thread's todos."""
    record, items = _object(thread), _field(thread, "items")
    if record is None or not isinstance(items, list):
        return thread
    return {**record, "items": [_with_names(item, rename, ("assigned_agent",)) for item in items]}


def _poke_state(state: object, *, rename: _NameRename, where: str) -> object:
    """Rename the agent in todo poke scope keys ``["<agent>","<room>","<thread>"(,"<requester>")]``."""
    record, scopes = _object(state), _object(_field(state, "scopes"))
    if record is None or scopes is None:
        return state
    return {**record, "scopes": _rename_keys(scopes, partial(_renamed_list_key, rename=rename), where)}


def _delegation_names(value: object, *, rename: _NameRename) -> object:
    """Rename the agent as caller or child of a delegation run or receipt, and in its ``<child>:<path>`` reference."""
    record = _object(_with_names(value, rename, ("caller_agent_name", "child_agent_name")))
    reference = _field(value, "record_reference")
    if record is None or not isinstance(reference, str):
        return value if record is None else record
    child, separator, path = reference.partition(":")
    if not separator or child != rename.old:
        return record
    return {**record, "record_reference": f"{rename.new}:{path}"}


def _delegation_state(state: object, *, rename: _NameRename) -> object:
    record = _object(state)
    if record is None or "run" not in record:
        return state
    return {**record, "run": _delegation_names(record["run"], rename=rename)}


def _workflow_summary(summary: object, *, rename: _NameRename, owned: bool) -> object:
    """Rename the agent as creator of a workflow, and as owner of one it owns."""
    return _with_names(summary, rename, ("owner_id", "created_by") if owned else ("created_by",))


def _workflow_revision(revision: object, *, rename: _NameRename) -> object:
    """Rename the agent as the author of a revision and as a room-agent participant."""
    record = _object(_with_names(revision, rename, ("updated_by",)))
    participants = _field(revision, "participants")
    if record is None or not isinstance(participants, list):
        return revision if record is None else record
    return {**record, "participants": [_with_names(participant, rename, ("agent",)) for participant in participants]}


def _workflow_run(run: object, *, rename: _NameRename, old_dir: str, new_dir: str) -> object:
    """Rename the agent as owner of a workflow run, and the owner directory in its artifact paths and report URL."""
    record = _object(_with_names(run, rename, ("owner_id",)))
    if record is None:
        return run
    renamed = dict(record)
    # Artifact paths are relative to the storage root (dynamic_workflows/store.py's _relative_artifact_path).
    old_prefix, new_prefix = (
        f"{_WORKFLOWS_DIRNAME}/{_AGENT_WORKFLOW_SCOPE}/{directory}/" for directory in (old_dir, new_dir)
    )
    artifacts = _object(record.get("artifacts"))
    if artifacts is not None:
        renamed["artifacts"] = {
            name: new_prefix + path.removeprefix(old_prefix)
            if isinstance(path, str) and path.startswith(old_prefix)
            else path
            for name, path in artifacts.items()
        }
    # Built like dynamic_workflows/store.py's _private_report_url.
    report_url = record.get("report_url")
    old_suffix, new_suffix = (
        f"/reports/private/{_AGENT_WORKFLOW_SCOPE}/{directory}/{record.get('workflow_id')}/{record.get('run_id')}"
        for directory in (old_dir, new_dir)
    )
    if isinstance(report_url, str) and report_url.endswith(old_suffix):
        renamed["report_url"] = report_url.removesuffix(old_suffix) + new_suffix
    return renamed


def _external_trigger(value: object, *, rename: _NameRename) -> object:
    record = _object(_with_names(value, rename, ("created_by_agent_name",)))
    if record is None or "target" not in record:
        return value if record is None else record
    return {**record, "target": _with_names(record["target"], rename, ("agent",))}


def _external_triggers(records: object, *, rename: _NameRename) -> object:
    """Rename the agent as creator (``created_by_agent_name``) and target (``target.agent``) of external triggers."""
    record, triggers = _object(records), _object(_field(records, "triggers"))
    if record is None or triggers is None:
        return records
    renamed = {trigger_id: _external_trigger(trigger, rename=rename) for trigger_id, trigger in triggers.items()}
    return {**record, "triggers": renamed}


def _names_agent(oauth_data: object, rename: _NameRename) -> bool:
    """Whether a pending OAuth state is bound to the agent, by name or by a worker key of the agent's."""
    worker_key = _field(oauth_data, "worker_key")
    return _field(oauth_data, "agent_name") == rename.old or (
        isinstance(worker_key, str)
        and rewrite_worker_key_agent(worker_key, old_name=rename.old, new_name=rename.new) is not None
    )


def _oauth_states(store: object, *, rename: _NameRename) -> object:
    """Leave out the pending OAuth states bound to the agent; their flow starts over under the new name."""
    record, states = _object(store), _object(_field(store, "states"))
    if record is None or states is None:
        return store
    kept = {token: state for token, state in states.items() if not _names_agent(_field(state, "data"), rename)}
    return {**record, "states": kept}


def _subagent_names_agent(handle: object, rename: _NameRename) -> bool:
    """Whether a subagent handle is the agent's, as caller or child, or binds the agent's storage."""
    child = _field(handle, "child")
    bindings = _field(child, "storage_bindings")
    return rename.old in (
        _field(_field(handle, "owner"), "agent_name"),
        _field(child, "caller_agent_name"),
        _field(child, "child_agent_name"),
    ) or (isinstance(bindings, dict) and rename.old in bindings)


def _read_json(path: Path) -> object:
    """Return the parsed file, or ``None`` when it is missing or not JSON; MindRoom skips such files too."""
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None


def _without_links(root: Path, path: Path) -> bool:
    """Whether ``path`` is a regular file reached from ``root`` without following a symlink.

    Worker code can write workspaces, so a link there could point a rewrite at a file outside them.
    """
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            return False
    return path.is_file()


@dataclass(frozen=True)
class _Moves:
    """Where paths are once the moves of the rename ran, so steps after them address the moved files."""

    pairs: tuple[tuple[Path, Path], ...]

    def after(self, path: Path) -> Path:
        for source, destination in self.pairs:
            if path.is_relative_to(source):
                return destination / path.relative_to(source)
        return path


def _records_dir(runtime_paths: RuntimePaths, name: str) -> Path:
    """Return the agent's primary-only records directory, ``tracking/agents/<name>``."""
    return primary_records_dir(agent_state_root_path(runtime_paths.storage_root, name), runtime_paths)


def _workflow_owner_dir(storage_root: Path, name: str) -> Path:
    return storage_root / _WORKFLOWS_DIRNAME / _AGENT_WORKFLOW_SCOPE / owner_dir_name(_AGENT_WORKFLOW_SCOPE, name)


def _sync_continuity_files(storage_root: Path, name: str) -> tuple[Path, Path]:
    """Return the agent's join-fence record and its lock (matrix/sync_continuity.py's SyncContinuityStore)."""
    record = storage_root / "sync_continuity" / f"{name}.json"
    return record, record.with_suffix(".json.lock")


def _named_paths(runtime_paths: RuntimePaths, name: str) -> list[tuple[Path, str]]:
    """Return every file or directory named after the agent outside its state root, with a label for each."""
    storage = runtime_paths.storage_root
    fences, fences_lock = _sync_continuity_files(storage, name)
    return [
        (_records_dir(runtime_paths, name), "the tracking records of"),
        (automations_tracking_root(runtime_paths) / name, "the automation state of"),
        (fences, "the pending room joins of"),
        (fences_lock, "the pending room joins lock of"),
        # matrix_rtc/transcript.py keeps the call transcripts of an agent without file memory there.
        (storage / "calls" / name, "the call transcripts of"),
        (_workflow_owner_dir(storage, name), "the dynamic workflows of"),
        (workspace_avatar_path("agents", name, runtime_paths), "the avatar of"),
        (storage / SHARED_DIRECTORIES_DIRNAME / name, "the shared directories owned by"),
    ]


def entity_file_steps(
    runtime_paths: RuntimePaths,
    *,
    config: Config,
    old_name: str,
    new_name: str,
) -> tuple[list[Step], list[str]]:
    """Return the steps that move the tracking records and side stores keyed by the agent's name, and report lines.

    Moves come first: ``tracking/agents/<OLD>``, ``tracking/automations/<OLD>``, ``sync_continuity/<OLD>.json``
    and its lock, ``calls/<OLD>``, the agent's dynamic workflow owner directory, its avatar, and
    ``shared_directories/<OLD>``. Then the JSON and YAML records naming the agent are rewritten at their final place,
    field by field; files inside the agent's state root are addressed below ``agents/<NEW>/``, which moves first.
    Then the SQLite stores are edited, and the records bound to the old name's storage are dropped.
    ``config`` names the agents whose workspaces may hold delegation exports naming the agent.
    Refuses when a destination already exists or a record already names the new agent where the old would go.
    """
    rename = _NameRename(old_name, new_name)
    storage = runtime_paths.storage_root
    old_paths, new_paths = _named_paths(runtime_paths, old_name), _named_paths(runtime_paths, new_name)
    steps: list[Step] = []
    pairs = [(agent_state_root_path(storage, old_name), agent_state_root_path(storage, new_name))]
    for (source, label), (destination, _) in zip(old_paths, new_paths, strict=True):
        if os.path.lexists(source):
            steps += move_steps(source, destination, f"{label} agent {old_name}", discard_new_state=False)
            pairs.append((source, destination))
    moves = _Moves(tuple(pairs))
    workflow_steps, workflow_reports = _workflow_steps(storage, rename, moves)
    script_steps, script_reports = _script_run_steps(runtime_paths, rename)
    steps += [
        *_tracking_steps(runtime_paths, rename, moves),
        *_delegation_export_steps(runtime_paths, config, rename, moves),
        *_todo_steps(runtime_paths, rename),
        *workflow_steps,
        *_external_trigger_steps(runtime_paths, rename),
        # desktop/pairing.py's _pairing_db_path.
        *_agent_name_edit(
            tracking_dir(runtime_paths) / "desktop_pairing.sqlite",
            "desktop_pairings",
            rename,
            "desktop pairings",
        ),
        *script_steps,
        *_gateway_selection_steps(storage, rename),
        *_drop_steps(runtime_paths, rename),
    ]
    return steps, [*workflow_reports, *script_reports, *_call_reports(storage, rename)]


def _call_reports(storage_root: Path, rename: _NameRename) -> list[str]:
    calls = storage_root / "calls" / rename.old
    if not os.path.lexists(calls):
        return []
    return [
        f"Memories of {rename.old} that cite call transcripts in {calls} keep citing that path; "
        f"the transcripts move to {storage_root / 'calls' / rename.new}.",
    ]


def _tracking_steps(runtime_paths: RuntimePaths, rename: _NameRename, moves: _Moves) -> list[Step]:
    """Rewrite the records below ``tracking/`` that name the agent."""
    tracking = tracking_dir(runtime_paths)
    label = f"Rename agent {rename.old} to {rename.new} in"
    modes = _records_dir(runtime_paths, rename.old) / "agent_modes.json"
    steps = _rewrite_steps(
        modes,
        moves.after(modes),
        _JSON_RECORDS,
        partial(_agent_modes, rename=rename, where=str(modes)),
        f"{label} the conversation modes {moves.after(modes)}",
    )
    # A record's state sits in the child's records directory, delegations/<date>/<id>/ (delegation/records.py).
    for state in sorted(tracking.glob("**/delegations/*/*/state.json")):
        steps += _rewrite_steps(
            state,
            moves.after(state),
            _JSON,
            partial(_delegation_state, rename=rename),
            f"{label} the delegation record {moves.after(state)}",
        )
    # memory/auto_flush.py keys each scope by its state root relative to the storage root.
    old_scope, new_scope = (
        _records_dir(runtime_paths, name).relative_to(tracking).as_posix() for name in (rename.old, rename.new)
    )
    auto_flush = partial(_auto_flush, old_scope=old_scope, new_scope=new_scope)
    for filename, codec, transform, what in (
        ("thread_models.json", _JSON_RECORDS, _thread_models, "the thread model choices"),
        ("memory_auto_flush.json", _JSON, auto_flush, "the memory auto-flush progress"),
        ("shared_directories.json", _JSON_RECORDS, _shared_directories, "the shared directory records"),
    ):
        path = tracking / filename
        bound = partial(transform, rename=rename, where=str(path))
        steps += _rewrite_steps(path, path, codec, bound, f"{label} {what} {path}")
    return steps


def _delegation_export_steps(
    runtime_paths: RuntimePaths,
    config: Config,
    rename: _NameRename,
    moves: _Moves,
) -> list[Step]:
    """Rewrite the delegation run exports and receipts in configured agents' workspaces that name the agent.

    MindRoom writes them from the primary's record and never reads them back, so a settled record's exports
    keep the old name unless rewritten here.
    """
    steps: list[Step] = []
    for name in config.agents:
        workspace = agent_workspace_root_path(runtime_paths.storage_root, name)
        exports = sorted(path for pattern in _DELEGATION_EXPORTS for path in workspace.glob(pattern))
        for export in exports:
            if not _without_links(workspace, export):
                continue
            steps += _rewrite_steps(
                export,
                moves.after(export),
                _JSON_EXPORT,
                partial(_delegation_names, rename=rename),
                f"Rename agent {rename.old} to {rename.new} in the delegation export {moves.after(export)}",
            )
    return steps


def _todo_steps(runtime_paths: RuntimePaths, rename: _NameRename) -> list[Step]:
    """Rewrite the todos assigned to the agent and its todo poke history."""
    todo_root = todo_state_root(runtime_paths)
    label = f"Rename agent {rename.old} to {rename.new} in"
    steps: list[Step] = []
    # custom_tools/todo_state.py's todos_path: threads/<key>/todos.json.
    for todos in sorted(todo_root.glob("threads/*/todos.json")):
        steps += _rewrite_steps(todos, todos, _JSON_LINES, partial(_todos, rename=rename), f"{label} the todos {todos}")
    # custom_tools/todo_poke.py's _POKE_STATE_FILENAME.
    pokes = todo_root / "poke_state.json"
    steps += _rewrite_steps(
        pokes,
        pokes,
        _JSON_LINES,
        partial(_poke_state, rename=rename, where=str(pokes)),
        f"{label} the todo poke history {pokes}",
    )
    return steps


def _workflow_steps(storage_root: Path, rename: _NameRename, moves: _Moves) -> tuple[list[Step], list[str]]:
    """Rewrite the dynamic workflows the agent owns, created, revised or takes part in, and report sent report links."""
    root = storage_root / _WORKFLOWS_DIRNAME
    old_owner, new_owner = _workflow_owner_dir(storage_root, rename.old), _workflow_owner_dir(storage_root, rename.new)
    label = f"Rename agent {rename.old} to {rename.new} in the dynamic workflow"
    steps: list[Step] = []
    for summary in sorted(root.glob("*/*/*/workflow.yaml")):
        steps += _rewrite_steps(
            summary,
            moves.after(summary),
            _YAML,
            partial(_workflow_summary, rename=rename, owned=summary.is_relative_to(old_owner)),
            f"{label} {moves.after(summary)}",
        )
    for revision in sorted(root.glob("*/*/*/revisions/*.yaml")):
        steps += _rewrite_steps(
            revision,
            moves.after(revision),
            _YAML,
            partial(_workflow_revision, rename=rename),
            f"{label} revision {moves.after(revision)}",
        )
    reports: list[str] = []
    for run in sorted(old_owner.glob("*/runs/*.json")):
        steps += _rewrite_steps(
            run,
            moves.after(run),
            _JSON_LINES,
            partial(_workflow_run, rename=rename, old_dir=old_owner.name, new_dir=new_owner.name),
            f"{label} run {moves.after(run)}",
        )
        report_url = _field(_read_json(run), "report_url")
        if isinstance(report_url, str) and not reports:
            reports.append(
                f"Report links of dynamic workflows of {rename.old} already sent, like {report_url}, stop working; "
                f"the reports are now served below /reports/private/{_AGENT_WORKFLOW_SCOPE}/{new_owner.name}/.",
            )
    return steps, reports


def _external_trigger_steps(runtime_paths: RuntimePaths, rename: _NameRename) -> list[Step]:
    """Rewrite the external triggers created by or targeting the agent."""
    if runtime_paths.control_state_root is None:
        return []
    # external_triggers/store.py's ExternalTriggerStore.store_path.
    triggers = runtime_paths.control_state_root / "external_triggers" / "triggers.json"
    return _rewrite_steps(
        triggers,
        triggers,
        _JSON_RECORDS,
        partial(_external_triggers, rename=rename),
        f"Rename agent {rename.old} to {rename.new} in the external triggers {triggers}",
    )


def _rows(database: Path, query: str, parameters: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    """Return the rows ``query`` reads from ``database``, or none when the database or its table is missing."""
    if not database.exists():
        return []
    try:
        with read_only_connection(database) as connection:
            return connection.execute(query, parameters).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            return []
        msg = f"Cannot read {database}: {exc}"
        raise RenameRefusedError(msg) from exc
    except sqlite3.Error as exc:
        msg = f"Cannot read {database}: {exc}"
        raise RenameRefusedError(msg) from exc


def _rename_column_value(connection: sqlite3.Connection, *, table: str, column: str, rename: _NameRename) -> None:
    # Only the fixed table and column names of this module are interpolated, never stored data.
    connection.execute(f"UPDATE {table} SET {column} = ? WHERE {column} = ?", (rename.new, rename.old))  # noqa: S608


def _agent_name_edit(database: Path, table: str, rename: _NameRename, label: str) -> list[Step]:
    """Return the edit that renames the agent in ``table.agent_name`` when a row there names it."""
    if not _rows(database, f"SELECT 1 FROM {table} WHERE agent_name = ? LIMIT 1", (rename.old,)):  # noqa: S608
        return []
    return [
        SqliteEdit(
            database,
            partial(_rename_column_value, table=table, column="agent_name", rename=rename),
            f"Rename agent {rename.old} to {rename.new} in the {label} {database}",
        ),
    ]


def _script_run_steps(runtime_paths: RuntimePaths, rename: _NameRename) -> tuple[list[Step], list[str]]:
    """Rename the agent in background script runs, and report its runs that had not finished."""
    if runtime_paths.control_state_root is None:
        return [], []
    # script_runs/store.py's ScriptRunStore.database_path.
    database = runtime_paths.control_state_root / "script_runs" / "script_runs.sqlite3"
    steps = _agent_name_edit(database, "script_runs", rename, "background script runs")
    unfinished = _rows(
        database,
        "SELECT run_id FROM script_runs WHERE agent_name = ? AND state IN (?, ?) ORDER BY run_id",
        (rename.old, *_UNFINISHED_SCRIPT_STATES),
    )
    if not unfinished:
        return steps, []
    run_ids = ", ".join(str(run_id) for (run_id,) in unfinished)
    report = (
        f"These background script runs of {rename.old} had not finished: {run_ids}. "
        f"They run with the identity of {rename.old}, so MindRoom refuses their further tool calls; start them again."
    )
    return steps, [report]


def _renamed_selection(agents: str, rename: _NameRename, where: str) -> str | None:
    """Return a gateway selection with the agent's entry renamed, or ``None`` when it does not select the agent."""
    try:
        selection = json.loads(agents)
    except ValueError:
        return None
    if not isinstance(selection, dict) or rename.old not in selection:
        return None
    renamed = _rename_keys(selection, lambda name: rename.new if name == rename.old else None, where)
    # Spelled like mcp_gateway/selection.py's _payload.
    return json.dumps(renamed)


def _rename_selected_agent(connection: sqlite3.Connection, *, rename: _NameRename, where: str) -> None:
    # The update trigger of gateway_selections moves the byte accounting with the new spelling.
    for owner_key, agents in connection.execute("SELECT owner_key, agents FROM gateway_selections").fetchall():
        renamed = _renamed_selection(agents, rename, where)
        if renamed is not None:
            connection.execute("UPDATE gateway_selections SET agents = ? WHERE owner_key = ?", (renamed, owner_key))


def _gateway_selection_steps(storage_root: Path, rename: _NameRename) -> list[Step]:
    """Rename the agent in the MCP gateway's agent selections.

    Pending consents no longer name an agent (mcp_gateway/legacy_schema.py's _migrate_user_authority).
    """
    # mcp_gateway/store.py's GatewayOAuthStore.path.
    database = storage_root / "mcp_gateway" / "oauth.sqlite3"
    where = f"An MCP gateway selection in {database}"
    rows = _rows(database, "SELECT agents FROM gateway_selections")
    if not any(_renamed_selection(str(agents), rename, where) is not None for (agents,) in rows):
        return []
    return [
        SqliteEdit(
            database,
            partial(_rename_selected_agent, rename=rename, where=where),
            f"Rename agent {rename.old} to {rename.new} in the MCP gateway agent selections {database}",
        ),
    ]


def _drop_steps(runtime_paths: RuntimePaths, rename: _NameRename) -> list[Step]:
    """Drop the subagent handles, idempotent send intents and pending OAuth states bound to the old name."""
    storage = runtime_paths.storage_root
    # delegation/sessions.py's _path: a handle binds the storage of its agents, which the rename moves.
    steps: list[Step] = [
        DropPath(path, f"Drop the subagent handle {path} of agent {rename.old}; a new subagent must be started")
        for path in sorted((storage / "subagent_sessions").glob("*.json"))
        if _subagent_names_agent(_read_json(path), rename)
    ]
    if runtime_paths.control_state_root is not None:
        # custom_tools/matrix_message_idempotency.py's _claim_matrix_message_send: one file per requester, agent and room.
        sends = runtime_paths.control_state_root / "matrix_message_sends"
        for path in sorted(sends.glob("*.json")):
            scope = _field(_read_json(path), "scope")
            if isinstance(scope, list) and len(scope) == 3 and scope[1] == rename.old:
                steps.append(DropPath(path, f"Drop the Matrix send intents of agent {rename.old} in {path}"))
    # oauth/state.py's _state_file.
    states = storage / "oauth_state" / "oauth_state.json"
    steps += _rewrite_steps(
        states,
        states,
        _JSON_SORTED,
        partial(_oauth_states, rename=rename),
        f"Drop the pending OAuth states of agent {rename.old} from {states}",
    )
    return steps
