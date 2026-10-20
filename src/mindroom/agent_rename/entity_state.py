"""Plan moving an agent's state root and renaming the session and learning databases inside it.

An agent's state lives in ``agents/<name>/`` (``agent_state_root_path``), its conversation history in
``sessions/<name>.db`` with tables ``<name>_sessions`` and ``<name>_sessions_*``, and its learnings in
``learning/<name>.db``. History only loads runs whose agent id is the agent's name
(``history/run_visibility.py``'s ``run_in_scope``) and replays the summary rows of the scope ``agent:<name>``,
so after the move the tables, the agent ids and the scope keys inside the databases take the new name too.
The Matrix account keeps its username, device and token; only its key in ``matrix_state.yaml`` changes.
Names match exactly, never as substrings, so renaming ``code`` leaves ``code_team`` alone.
Building the steps only reads; every refusal happens here, before anything is written.
"""

from __future__ import annotations

import json
import os
import re
from functools import partial
from typing import TYPE_CHECKING, cast

from mindroom import yaml_io
from mindroom.agent_rename.sqlite_read import quote_identifier
from mindroom.agent_rename.steps import FileRewrite, MovePath, RenameRefusedError, SqliteEdit
from mindroom.agent_rename.workers import DestinationExistsError, move_steps
from mindroom.constants import matrix_state_file, resolve_session_state_root
from mindroom.matrix.identity import managed_account_key
from mindroom.tool_system.worker_routing import agent_state_root_path

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from mindroom.agent_rename.steps import Step
    from mindroom.constants import RuntimePaths

# A crashed writer can leave commits or a rollback journal beside a database; SQLite finds them by name.
_SQLITE_SIDE_FILES = ("-journal", "-wal", "-shm")
# Agno's default schema-version table (agno/db/base.py), which MindRoom does not rename.
_SCHEMA_VERSIONS_TABLE = "agno_schema_versions"
# The learning store's own table (agno/db/base.py's learnings_table default).
_LEARNINGS_TABLE = "agno_learnings"
# The stored head of a CREATE INDEX statement and the index name it names.
_INDEX_HEAD = re.compile(
    r'CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?("(?:[^"]|"")*"|\[[^\]]*\]|`(?:[^`]|``)*`|[^\s(]+)',
    re.IGNORECASE,
)


def entity_state_steps(runtime_paths: RuntimePaths, *, old_name: str, new_name: str) -> list[Step]:
    """Return the steps that move the agent's state root to the new name and rename its databases there.

    The state root ``agents/<OLD>`` moves to ``agents/<NEW>`` first, so the database steps address paths below
    the new root. With ``MINDROOM_SESSION_STORAGE_PATH`` set, the agent's root in that session root moves too.
    ``sessions/<OLD>.db`` and ``learning/<OLD>.db`` become ``<NEW>.db``, with their tables, indexes,
    schema-version rows, agent ids and history scope keys renamed in one transaction each.
    An agent without a state root never ran and gets no steps. Refuses when a destination already exists.
    """
    storage_root = runtime_paths.storage_root
    old_root, new_root = agent_state_root_path(storage_root, old_name), agent_state_root_path(storage_root, new_name)
    old_session_root = resolve_session_state_root(old_root, runtime_paths)
    new_session_root = resolve_session_state_root(new_root, runtime_paths)
    steps: list[Step] = []
    if os.path.lexists(old_root):
        steps += move_steps(old_root, new_root, f"the state of agent {old_name}", discard_new_state=False)
    if old_session_root != old_root and os.path.lexists(old_session_root):
        steps += move_steps(
            old_session_root,
            new_session_root,
            f"the sessions of agent {old_name}",
            discard_new_state=False,
        )
    steps += _database_steps(
        old_session_root / "sessions",
        new_session_root / "sessions",
        old_name=old_name,
        new_name=new_name,
        label="session database",
        edit=partial(
            _rename_agent_in_database,
            old_table=f"{old_name}_sessions",
            new_table=f"{new_name}_sessions",
            old_name=old_name,
            new_name=new_name,
            other_tables=(),
        ),
    )
    steps += _database_steps(
        old_root / "learning",
        new_root / "learning",
        old_name=old_name,
        new_name=new_name,
        label="learning database",
        edit=partial(
            _rename_agent_in_database,
            old_table=f"{old_name}_learning_sessions",
            new_table=f"{new_name}_learning_sessions",
            old_name=old_name,
            new_name=new_name,
            other_tables=(_LEARNINGS_TABLE,),
        ),
    )
    return steps


def _database_steps(
    old_directory: Path,
    new_directory: Path,
    *,
    old_name: str,
    new_name: str,
    label: str,
    edit: Callable[[sqlite3.Connection], None],
) -> list[Step]:
    """Rename ``<OLD>.db`` and its side files in the moved directory to ``<NEW>.db``, then edit it there."""
    old_file, new_file = f"{old_name}.db", f"{new_name}.db"
    if not os.path.lexists(old_directory / old_file):
        return []
    steps: list[Step] = []
    for suffix in ("", *_SQLITE_SIDE_FILES):
        if suffix and not os.path.lexists(old_directory / f"{old_file}{suffix}"):
            continue
        if os.path.lexists(old_directory / f"{new_file}{suffix}"):
            msg = f"Cannot rename {old_directory / old_file}: {old_directory / f'{new_file}{suffix}'} already exists."
            raise DestinationExistsError(msg)
        source, destination = new_directory / f"{old_file}{suffix}", new_directory / f"{new_file}{suffix}"
        steps.append(MovePath(source, destination, f"Move the {label} of agent {old_name} {source} to {destination}"))
    database = new_directory / new_file
    steps.append(
        SqliteEdit(
            database,
            edit,
            f"Rename agent {old_name} to {new_name} in the tables, agent ids and history scopes of {database}",
        ),
    )
    return steps


def _rename_agent_in_database(
    connection: sqlite3.Connection,
    *,
    old_table: str,
    new_table: str,
    old_name: str,
    new_name: str,
    other_tables: tuple[str, ...],
) -> None:
    """Rename the tables named ``old_table`` or ``old_table_*`` with their indexes, then the agent inside them.

    ``other_tables`` are tables whose names stay but whose rows name the agent too.
    """
    # Renaming a table then also renames it in other tables' foreign keys, indexes and triggers.
    connection.execute("PRAGMA legacy_alter_table = OFF")
    tables = [
        name
        for (name,) in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        if name == old_table or name.startswith(f"{old_table}_")
    ]
    renamed = {name: new_table + name.removeprefix(old_table) for name in tables}
    for old, new in renamed.items():
        connection.execute(f"ALTER TABLE {quote_identifier(old)} RENAME TO {quote_identifier(new)}")
    _rename_indexes(connection, renamed.values(), old_table=old_table, new_table=new_table)
    if _table_exists(connection, _SCHEMA_VERSIONS_TABLE):
        for old, new in renamed.items():
            connection.execute(
                f"UPDATE {_SCHEMA_VERSIONS_TABLE} SET table_name = ? WHERE table_name = ?",  # noqa: S608
                (new, old),
            )
    for table in (*renamed.values(), *(name for name in other_tables if _table_exists(connection, name))):
        _rename_agent_in_table(connection, table, old_name=old_name, new_name=new_name)


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    query = "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?"
    return connection.execute(query, (name,)).fetchone() is not None


def _rename_indexes(connection: sqlite3.Connection, tables: Iterable[str], *, old_table: str, new_table: str) -> None:
    """Recreate the indexes of ``tables`` whose names start with the old table name under the new one.

    Agno names indexes ``idx_<table>_<columns>`` and ``<table>_<name>``, MindRoom ``<table>_<name>``.
    SQLite's own indexes have no SQL and take the new name with their table.
    """
    rows = connection.execute(
        "SELECT name, sql FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL "
        "AND tbl_name IN (SELECT value FROM json_each(?))",
        (json.dumps(list(tables)),),
    ).fetchall()
    for name, sql in rows:
        new_name = next(
            (
                head + new_table + name.removeprefix(head + old_table)
                for head in ("idx_", "")
                if name.startswith(head + old_table)
            ),
            None,
        )
        if new_name is None:
            continue
        connection.execute(f"DROP INDEX {quote_identifier(name)}")
        connection.execute(_with_index_name(sql, name, new_name))


def _with_index_name(sql: str, name: str, new_name: str) -> str:
    """Return the stored ``CREATE INDEX`` statement of index ``name`` creating ``new_name`` instead."""
    match = _INDEX_HEAD.match(sql)
    token = match.group(1) if match is not None else ""
    if token[:1] in {'"', "`"}:
        token = token[1:-1].replace(token[0] * 2, token[0])
    elif token[:1] == "[":
        token = token[1:-1]
    if match is None or token != name:
        msg = f"Cannot read the definition of index {name}: {sql}"
        raise RuntimeError(msg)
    return sql[: match.start(1)] + quote_identifier(new_name) + sql[match.end(1) :]


def _rename_agent_in_table(connection: sqlite3.Connection, table: str, *, old_name: str, new_name: str) -> None:
    """Set the agent id columns, history scope keys and the agent fields of JSON columns from OLD to NEW."""
    quoted = quote_identifier(table)
    columns = {name for (name,) in connection.execute("SELECT name FROM pragma_table_info(?)", (table,))}
    old_scope, new_scope = f"agent:{old_name}", f"agent:{new_name}"
    if "agent_id" in columns:
        connection.execute(f"UPDATE {quoted} SET agent_id = ? WHERE agent_id = ?", (new_name, old_name))  # noqa: S608
    if "scope_key" in columns:
        connection.execute(f"UPDATE {quoted} SET scope_key = ? WHERE scope_key = ?", (new_scope, old_scope))  # noqa: S608
    renames: dict[str, Callable[[object], object]] = {
        "agent_data": partial(_renamed_field, field="agent_id", old=old_name, new=new_name),
        "run_data": partial(_renamed_run, old=old_name, new=new_name),
        # The Agno 2 run list that the open-time migration turns into run rows (legacy_session_storage.py).
        "runs": partial(_renamed_runs, old=old_name, new=new_name),
        # MindRoom keys per-scope history state by scope key (history/storage.py).
        "metadata": partial(_renamed_scope_keys, old=old_scope, new=new_scope),
        "session_data": partial(_renamed_scope_keys, old=old_scope, new=new_scope),
    }
    for column, rename in renames.items():
        if column not in columns:
            continue
        quoted_column = quote_identifier(column)
        # Agent names hold no character JSON escapes, so a value naming the agent contains the name verbatim.
        rows = connection.execute(
            f"SELECT rowid, {quoted_column} FROM {quoted} WHERE instr({quoted_column}, ?) > 0",  # noqa: S608
            (old_name,),
        ).fetchall()
        for rowid, raw in rows:
            text = _renamed_json(raw, rename)
            if text is not None:
                connection.execute(f"UPDATE {quoted} SET {quoted_column} = ? WHERE rowid = ?", (text, rowid))  # noqa: S608


def _renamed_json(raw: object, rename: Callable[[object], object]) -> str | None:
    """Return the JSON text with ``rename`` applied, encoded as often as before, or ``None`` when nothing changes.

    Older releases stored some session values as the JSON text of their JSON text.
    Values that are not JSON are left alone, as MindRoom cannot read them either.
    """
    if not isinstance(raw, str):
        return None
    try:
        value = json.loads(raw)
        nested = isinstance(value, str)
        if nested:
            value = json.loads(value)
    except (RecursionError, ValueError):
        return None
    renamed = rename(value)
    if renamed == value:
        return None
    # Agno writes its JSON columns through SQLAlchemy's default serializer, json.dumps.
    text = json.dumps(renamed)
    return json.dumps(text) if nested else text


def _renamed_field(value: object, *, field: str, old: str, new: str) -> object:
    if isinstance(value, dict) and cast("dict[str, object]", value).get(field) == old:
        return {**cast("dict[str, object]", value), field: new}
    return value


def _renamed_run(value: object, *, old: str, new: str) -> object:
    """Rename the agent id fields Agno 3.0.9 writes into a run: its own, its events', its requirements' member."""
    run = _renamed_field(value, field="agent_id", old=old, new=new)
    if not isinstance(run, dict):
        return run
    renamed = dict(cast("dict[str, object]", run))
    for key, field in (("events", "agent_id"), ("requirements", "member_agent_id")):
        items = renamed.get(key)
        if isinstance(items, list):
            renamed[key] = [_renamed_field(item, field=field, old=old, new=new) for item in cast("list[object]", items)]
    return renamed


def _renamed_runs(value: object, *, old: str, new: str) -> object:
    if not isinstance(value, list):
        return value
    return [_renamed_run(run, old=old, new=new) for run in cast("list[object]", value)]


def _renamed_scope_keys(value: object, *, old: str, new: str) -> object:
    """Replace every key and string equal to the scope key ``old``, however deep."""
    if value == old:
        return new
    if isinstance(value, list):
        return [_renamed_scope_keys(item, old=old, new=new) for item in cast("list[object]", value)]
    if isinstance(value, dict):
        return {
            new if key == old else key: _renamed_scope_keys(item, old=old, new=new)
            for key, item in cast("dict[str, object]", value).items()
        }
    return value


def matrix_state_key_step(runtime_paths: RuntimePaths, *, old_name: str, new_name: str) -> FileRewrite:
    """Return the step that moves the agent's stored Matrix account to the new name's key in ``matrix_state.yaml``.

    MindRoom finds an agent's account under ``agent_<name>``; the username, password, device and token stay,
    so the agent keeps its Matrix ID and device.
    Refuses when the file holds no account for the old name or already holds one for the new name.
    """
    path = matrix_state_file(runtime_paths)
    old_key, new_key = managed_account_key(old_name), managed_account_key(new_name)
    state = yaml_io.safe_load(path.read_bytes()) if path.exists() else None
    accounts = cast("dict[str, object]", state).get("accounts") if isinstance(state, dict) else None
    if not isinstance(accounts, dict) or old_key not in accounts:
        msg = f"{path} holds no account {old_key}, so agent {old_name} has no Matrix account to keep."
        raise RenameRefusedError(msg)
    if new_key in accounts:
        msg = f"{path} already holds an account {new_key}."
        raise RenameRefusedError(msg)
    return FileRewrite(
        path,
        partial(_rename_account_key, old_key=old_key, new_key=new_key),
        f"Rename the account {old_key} in {path} to {new_key}; its username, device and access token stay",
    )


def _rename_account_key(content: bytes, *, old_key: str, new_key: str) -> bytes:
    state = yaml_io.safe_load(content)
    accounts = state["accounts"]
    if old_key not in accounts or new_key in accounts:
        msg = f"the accounts changed since planning: expected {old_key} and no {new_key}"
        raise RuntimeError(msg)
    # The account keeps its place among the others.
    state["accounts"] = {new_key if key == old_key else key: account for key, account in accounts.items()}
    # Dumped like matrix/state.py's _write_matrix_state_file.
    return yaml_io.safe_dump(state, default_flow_style=False, sort_keys=False).encode()
