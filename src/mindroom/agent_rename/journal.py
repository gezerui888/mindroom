"""Rewrite the event journal when an agent's Matrix account or config name is renamed.

The journal keys each bot's state by its principal, ``<agent>@<Matrix ID>``, and mirrors Matrix senders and payloads.
After a server renamed the account in place, every event the server re-delivers carries the new ID.
A stored sender that differs from the re-delivered one fails ingestion with ``IngestionBatchIntegrityError``,
so the journal must hold the new ID everywhere the server now shows it (``journal_account_steps``).

Renaming the agent's config name keeps its Matrix account and device, so nothing Matrix shows changes; the
principal, the turn records and the values naming the agent take the new name (``journal_entity_steps``).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

from mindroom.agent_rename.matching import contains_user_id, replace_user_id
from mindroom.agent_rename.sqlite_read import quote_identifier, read_only_connection, text_columns
from mindroom.agent_rename.steps import RenameRefusedError, SqliteEdit
from mindroom.agent_rename.teams import renamed_ad_hoc_scope_id

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

_PRINCIPAL_COLUMNS = frozenset({"principal_id", "responder_principal_id"})
# Rows for replies MindRoom already sent to Matrix; discarding them would lose track of messages in the rooms.
_SENT_REPLY_TABLES = ("reply_messages", "matrix_delivery_outbox")
# Grants hash the old principal into their scope key, so a renamed grant could never match again.
_GRANT_TABLES = ("approval_grants", "approval_grant_cards")
# A principal's sync consumer and the producer positions its stream reached; they belong to one device's sync stream.
_SYNC_STREAM_TABLES = ("matrix_sync_consumers", "matrix_ingestion_membership")

type _SqliteValue = str | bytes | int | float | None


def journal_account_steps(
    database: Path,
    *,
    agent: str,
    old_user_id: str,
    new_user_id: str,
    discard_new_principal: bool,
    new_device_kept: bool,
) -> tuple[list[SqliteEdit], list[str]]:
    """Return the one edit that moves ``agent``'s journal state from ``old_user_id`` to ``new_user_id``, and reports.

    When the old principal has rows, the edit moves them: it renames the principal, replaces the old ID in every
    text cell, and drops the principal's approval grants. Rows the new principal has as well mean
    MindRoom ran under the new ID before the journal was moved; that refuses unless ``discard_new_principal``
    is set, and always when they include replies already sent.

    The sync stream state, a principal's sync consumer and producer membership positions, must belong to the stream
    the device's store continues. The old principal's never does, so it is dropped. The new principal's stays only
    when ``new_device_kept`` says the stored account keeps the device it uses under the new ID; otherwise the next
    start logs in on a fresh device that binds a fresh stream, and it is dropped as well.

    When the old principal has no rows there is nothing to move, and the new principal's rows are the agent's own
    history: moved by an earlier run that was killed or is repeated, or written since. They are never deleted,
    whatever ``discard_new_principal`` says. Text cells that still hold the old ID, such as the agent's messages
    as other principals saw them, are still rewritten; when there are none, the journal is left alone.

    The edit checks the old principal again in its transaction and refuses if a rewritten row would collide.
    A journal that was never created needs no edit.
    """
    if not database.exists():
        return [], []
    old_principal = f"{agent}@{old_user_id}"
    new_principal = f"{agent}@{new_user_id}"
    try:
        with read_only_connection(database) as connection:
            columns = text_columns(connection)
            old_rows = _principal_row_counts(connection, columns, old_principal)
            new_rows = _principal_row_counts(connection, columns, new_principal)
            old_cells = [] if old_rows else _cells_holding(connection, columns, old_user_id)
    except (sqlite3.Error, OSError) as exc:
        msg = f"Cannot read the event journal {database}: {exc}"
        raise RenameRefusedError(msg) from exc
    if old_rows:
        _refuse_new_principal_rows(
            old_rows,
            new_rows,
            old_principal=old_principal,
            new_principal=new_principal,
            discard=discard_new_principal,
        )
        kept = {table: count for table, count in new_rows.items() if new_device_kept and table in _SYNC_STREAM_TABLES}
        discarded = {table: count for table, count in new_rows.items() if table not in kept}
        deletion = f"delete the rows already under {new_principal} ({_format_counts(discarded)}), " if discarded else ""
        stream = (
            f"drop the sync stream state of {old_principal} and keep that of {new_principal} "
            f"({_format_counts(kept)}), whose device continues that stream"
            if kept
            else "drop the principal's sync stream state, so the next start binds a fresh stream"
        )
        description = (
            f"Rewrite the event journal {database}: {deletion}rename principal {old_principal} to {new_principal}, "
            f"replace {old_user_id} with {new_user_id} in every text cell, "
            f"{stream}, and drop the principal's approval grants"
        )
    elif old_cells:
        description = (
            f"Rewrite the event journal {database}: replace {old_user_id} with {new_user_id} in {', '.join(old_cells)}; "
            f"{old_principal} has no rows, so the rows of {new_principal} stay"
        )
    else:
        kept = f"; its rows for {new_principal} ({_format_counts(new_rows)}) are kept" if new_rows else ""
        return [], [f"The event journal {database} has no rows for {old_principal}, so it stays as it is{kept}."]
    edit = partial(
        _rewrite,
        old_user_id=old_user_id,
        new_user_id=new_user_id,
        old_principal=old_principal,
        new_principal=new_principal,
        move_principal=bool(old_rows),
        discard_new_principal=discard_new_principal,
        new_device_kept=new_device_kept,
    )
    return [SqliteEdit(database=database, edit=edit, description=description)], []


def other_principal_user_ids(database: Path, *, agent: str, user_id: str) -> dict[str, str]:
    """Return the IDs other than ``user_id`` that ``agent``'s journal principals name, with their row counts.

    The journal keys ``agent``'s rows by ``<agent>@<Matrix ID>``, so rows under another ID mean the journal still
    holds state from before the stored account took ``user_id``.
    A journal that was never created holds none.
    """
    if not database.exists():
        return {}
    prefix = f"{agent}@"
    try:
        with read_only_connection(database) as connection:
            columns = text_columns(connection)
            principals: set[str] = set()
            for table, names in columns.items():
                for column in names:
                    if column not in _PRINCIPAL_COLUMNS:
                        continue
                    quoted = quote_identifier(column)
                    # substr instead of LIKE: `_` in an agent name is a LIKE wildcard.
                    query = f"SELECT DISTINCT {quoted} FROM {quote_identifier(table)} WHERE substr({quoted}, 1, ?) = ?"  # noqa: S608 - quoted schema identifiers
                    principals.update(principal for (principal,) in connection.execute(query, (len(prefix), prefix)))
            return {
                principal.removeprefix(prefix): _format_counts(_principal_row_counts(connection, columns, principal))
                for principal in sorted(principals)
                if principal != f"{prefix}{user_id}"
            }
    except (sqlite3.Error, OSError) as exc:
        msg = f"Cannot read the event journal {database}: {exc}"
        raise RenameRefusedError(msg) from exc


def _sql_contains_user_id(value: _SqliteValue, user_id: str) -> bool:
    return isinstance(value, str) and contains_user_id(value, user_id)


def _sql_replace_user_id(value: _SqliteValue, old_user_id: str, new_user_id: str) -> _SqliteValue:
    return replace_user_id(value, old_user_id, new_user_id) if isinstance(value, str) else value


def _holds_user_id(quoted_column: str) -> str:
    """Return a condition true when the column holds ``:old`` as a whole user ID.

    ``instr`` runs first so the Python matcher only sees cells that contain the ID's text at all.
    """
    return f"instr({quoted_column}, :old) > 0 AND contains_user_id({quoted_column}, :old)"


def _principal_match(columns: tuple[str, ...]) -> str:
    """Return a condition matching rows whose principal columns name ``:principal``, or ``""`` if there are none."""
    return " OR ".join(f"{quote_identifier(column)} = :principal" for column in columns if column in _PRINCIPAL_COLUMNS)


def _principal_row_counts(
    connection: sqlite3.Connection,
    columns: dict[str, tuple[str, ...]],
    principal: str,
) -> dict[str, int]:
    """Count, per table, the rows that name ``principal`` in a principal column."""
    counts: dict[str, int] = {}
    for table, names in columns.items():
        match = _principal_match(names)
        if not match:
            continue
        query = f"SELECT count(*) FROM {quote_identifier(table)} WHERE {match}"  # noqa: S608 - quoted schema identifiers
        (count,) = connection.execute(query, {"principal": principal}).fetchone()
        if count:
            counts[table] = count
    return counts


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{table}: {count}" for table, count in sorted(counts.items()))


def _refuse_new_principal_rows(
    old_counts: dict[str, int],
    new_counts: dict[str, int],
    *,
    old_principal: str,
    new_principal: str,
    discard: bool,
) -> None:
    """Refuse when the new principal has rows besides the old one's, unless they may be discarded.

    Only called while the old principal has rows: without them the new principal's rows are the agent's history.
    """
    if not new_counts:
        return
    both = (
        f"The event journal has rows for both {old_principal} ({_format_counts(old_counts)}) "
        f"and {new_principal} ({_format_counts(new_counts)}), "
        "because MindRoom ran under the new ID before the journal was moved"
    )
    if any(table in new_counts for table in _SENT_REPLY_TABLES):
        msg = (
            f"{both}. The {new_principal} rows include replies MindRoom already sent under the new ID, "
            "which --discard-new-principal does not delete."
        )
        raise RenameRefusedError(msg)
    if not discard:
        msg = (
            f"{both}. Rerun with --discard-new-principal to delete the {new_principal} rows "
            f"and move the {old_principal} rows in their place."
        )
        raise RenameRefusedError(msg)


def _register_matchers(connection: sqlite3.Connection) -> None:
    connection.create_function("contains_user_id", 2, _sql_contains_user_id, deterministic=True)
    connection.create_function("replace_user_id", 3, _sql_replace_user_id, deterministic=True)


def _rewrite(
    connection: sqlite3.Connection,
    *,
    old_user_id: str,
    new_user_id: str,
    old_principal: str,
    new_principal: str,
    move_principal: bool,
    discard_new_principal: bool,
    new_device_kept: bool,
) -> None:
    """Move the journal to the new ID inside the caller's transaction; raising rolls everything back.

    Without ``move_principal`` the old principal had no rows when planned, so only text cells change
    and nothing of the new principal is deleted, its sync stream state included.
    """
    _register_matchers(connection)
    columns = text_columns(connection)
    old_rows = _principal_row_counts(connection, columns, old_principal)
    if bool(old_rows) != move_principal:
        # Never decide about deleting the new principal's rows on a journal the plan did not see.
        change = "no longer has" if move_principal else "now has"
        msg = (
            f"The event journal {change} rows for {old_principal}, unlike when the plan was made. "
            "The event journal was left unchanged; run the command again to plan anew."
        )
        raise RenameRefusedError(msg)
    if move_principal:
        new_rows = _principal_row_counts(connection, columns, new_principal)
        _refuse_new_principal_rows(
            old_rows,
            new_rows,
            old_principal=old_principal,
            new_principal=new_principal,
            discard=discard_new_principal,
        )
        # The new device's store continues the new principal's stream, so that stream's state must stay.
        kept = _SYNC_STREAM_TABLES if new_device_kept else ()
        discarded = tuple(table for table in new_rows if table not in kept)
        _delete_principal_rows(connection, columns, tables=discarded, principal=new_principal)
        # No device continues the old login's stream; a fresh stream starts every room at membership epoch 0.
        _delete_principal_rows(connection, columns, tables=_SYNC_STREAM_TABLES, principal=old_principal)
    _replace_identity(
        connection,
        columns,
        {"old_principal": old_principal, "new_principal": new_principal, "old": old_user_id, "new": new_user_id},
    )
    if "room_member_joins" in columns:
        # Any old-ID row left here collided with a row the bot already recorded for the new ID.
        connection.execute("DELETE FROM room_member_joins WHERE user_id = ?", (old_user_id,))
    if move_principal:
        _delete_principal_rows(connection, columns, tables=_GRANT_TABLES, principal=new_principal)
    _refuse_leftovers(connection, columns, old_user_id=old_user_id, new_user_id=new_user_id)


def _delete_principal_rows(
    connection: sqlite3.Connection,
    columns: dict[str, tuple[str, ...]],
    *,
    tables: tuple[str, ...],
    principal: str,
) -> None:
    """Delete the rows of ``tables`` that name ``principal`` in a principal column."""
    for table in tables:
        match = _principal_match(columns.get(table, ()))
        if match:
            delete = f"DELETE FROM {quote_identifier(table)} WHERE {match}"  # noqa: S608 - quoted schema identifiers
            connection.execute(delete, {"principal": principal})


def _replace_identity(
    connection: sqlite3.Connection,
    columns: dict[str, tuple[str, ...]],
    names: dict[str, str],
) -> None:
    """Rename the principal in every principal column, then replace the old ID in every other text column."""
    renames: list[str] = []
    replacements: list[str] = []
    for table, table_columns in columns.items():
        for column in table_columns:
            quoted_table, quoted = quote_identifier(table), quote_identifier(column)
            if column in _PRINCIPAL_COLUMNS:
                renames.append(f"UPDATE {quoted_table} SET {quoted} = :new_principal WHERE {quoted} = :old_principal")  # noqa: S608 - quoted schema identifiers
            else:
                # A row whose rewrite would collide keeps the old ID, and the leftover check refuses it.
                replacements.append(
                    f"UPDATE OR IGNORE {quoted_table} SET {quoted} = replace_user_id({quoted}, :old, :new) "  # noqa: S608 - quoted schema identifiers
                    f"WHERE {_holds_user_id(quoted)}",
                )
    for statement in (*renames, *replacements):
        connection.execute(statement, names)


def _cells_holding(connection: sqlite3.Connection, columns: dict[str, tuple[str, ...]], user_id: str) -> list[str]:
    """Return ``table.column (count)`` for every text column with cells that hold ``user_id`` as a whole user ID."""
    _register_matchers(connection)
    cells: list[str] = []
    for table, table_columns in columns.items():
        if not table_columns:
            continue
        sums = ", ".join(f"sum({_holds_user_id(quote_identifier(column))})" for column in table_columns)
        counts = connection.execute(f"SELECT {sums} FROM {quote_identifier(table)}", {"old": user_id}).fetchone()  # noqa: S608 - quoted schema identifiers
        cells.extend(
            f"{table}.{column} ({count})" for column, count in zip(table_columns, counts, strict=True) if count
        )
    return cells


def _refuse_leftovers(
    connection: sqlite3.Connection,
    columns: dict[str, tuple[str, ...]],
    *,
    old_user_id: str,
    new_user_id: str,
) -> None:
    """Refuse when any text cell still holds the old ID as a whole user ID, naming each table and column."""
    leftovers = _cells_holding(connection, columns, old_user_id)
    if leftovers:
        msg = (
            f"Rewriting the event journal would leave {old_user_id} in {', '.join(leftovers)}, "
            f"most likely in rows that would collide with rows already holding {new_user_id}. "
            "The event journal was left unchanged."
        )
        raise RenameRefusedError(msg)


# Columns holding an agent's config name verbatim (event_journal/schema.py): the agent a turn record is scoped to,
# the entity owning a reply or a scoped approval card, and the agent a scheduled or paused call runs as.
_NAME_COLUMNS = (
    ("turn_records", "agent_name"),
    ("reply_messages", "entity_name"),
    ("scheduled_call_approvals", "agent_name"),
    ("approval_continuation_calls", "invoking_agent"),
    ("approval_grant_cards", "entity_name"),
    ("approval_grant_cards", "invoking_agent"),
)
# A grant's scope key hashes its invoking agent and responder principal (event_journal/approval_grants.py), so a grant
# naming the agent in either could never match a call again; its entity is the agent too.
_GRANT_MATCH = (
    "principal_id = :principal OR responder_principal_id = :principal OR entity_name = :name OR invoking_agent = :name"
)
# ToolExecutionIdentity's fields naming agents (tool_system/worker_routing.py).
_IDENTITY_FIELDS = ("agent_name", "transport_agent_name")


@dataclass(frozen=True)
class _EntityRename:
    old: str
    new: str
    user_id: str
    database: Path

    @property
    def old_principal(self) -> str:
        return f"{self.old}@{self.user_id}"

    @property
    def new_principal(self) -> str:
        return f"{self.new}@{self.user_id}"

    def name(self, value: object) -> object:
        """Return the new name for a value that is the old name, and any other value unchanged."""
        return self.new if value == self.old else value


def journal_entity_steps(database: Path, *, old_name: str, new_name: str, user_id: str) -> list[SqliteEdit]:
    """Return the one edit that renames agent ``old_name`` to ``new_name`` in the event journal, if it names the agent.

    The agent keeps its Matrix account ``user_id`` and its device, so its principal ``<old_name>@<user_id>`` becomes
    ``<new_name>@<user_id>`` in every principal column with all of its rows. That includes its sync consumer and
    producer membership positions: the device's store continues their stream. Its turn records move to the new name,
    and so do the columns and JSON fields naming the agent; an ad-hoc team scope listing it takes the id of the
    renamed member set. Approval grants bound to the agent are dropped, as their scope keys hash its name.

    Refuses when the journal already has rows for ``new_name``, a principal ``<new_name>@<any ID>`` or turn records,
    or an ad-hoc team scope listing both names; the edit checks again in its transaction.
    A journal that was never created, or that names the agent nowhere, needs no edit.
    """
    if not database.exists():
        return []
    rename = _EntityRename(old_name, new_name, user_id, database)
    try:
        with read_only_connection(database) as connection:
            changes = _entity_changes(connection, rename)
    except (sqlite3.Error, OSError) as exc:
        msg = f"Cannot read the event journal {database}: {exc}"
        raise RenameRefusedError(msg) from exc
    parts: list[str] = []
    if changes.principal_rows:
        parts.append(
            f"rename principal {rename.old_principal} to {rename.new_principal} "
            f"({_format_counts(changes.principal_rows)}), keeping the sync stream state its device continues",
        )
    if changes.name_rows:
        renamed = ", ".join(f"{table}.{column} ({count})" for (table, column), count in changes.name_rows.items())
        parts.append(f"rename agent {old_name} to {new_name} in {renamed}")
    if changes.json_rows:
        rewritten = ", ".join(f"{table}.{column} ({len(rows)})" for (table, column), rows in changes.json_rows.items())
        parts.append(f"rewrite the fields naming agent {old_name} and the ad-hoc team scopes listing it in {rewritten}")
    if changes.grants:
        parts.append(f"drop the approval grants bound to agent {old_name} ({changes.grants})")
    if not parts:
        return []
    description = f"Rewrite the event journal {database}: {'; '.join(parts)}"
    return [SqliteEdit(database=database, edit=partial(_rename_entity, rename=rename), description=description)]


@dataclass(frozen=True)
class _EntityChanges:
    """What renaming the agent changes in a journal, counted per table or column."""

    columns: dict[str, tuple[str, ...]]
    principal_rows: dict[str, int]
    name_rows: dict[tuple[str, str], int]
    # The rowid and rewritten text of each row of a JSON column whose value the rename changes.
    json_rows: dict[tuple[str, str], list[tuple[int, str]]]
    grants: int


def _entity_changes(connection: sqlite3.Connection, rename: _EntityRename) -> _EntityChanges:
    """Read what renaming the agent changes, refusing a journal that already has rows for the new name."""
    columns = text_columns(connection)
    _refuse_new_name_rows(connection, columns, rename)
    name_rows: dict[tuple[str, str], int] = {}
    for table, column in _NAME_COLUMNS:
        if column in columns.get(table, ()):
            quoted = quote_identifier(column)
            query = f"SELECT count(*) FROM {quote_identifier(table)} WHERE {quoted} = ?"  # noqa: S608 - quoted schema identifiers
            (count,) = connection.execute(query, (rename.old,)).fetchone()
            if count:
                name_rows[table, column] = count
    json_rows: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for spec in _JSON_COLUMNS:
        if rewrites := _json_rewrites(connection, columns, spec, rename):
            json_rows[spec.table, spec.column] = rewrites
    grants = 0
    if "approval_grants" in columns:
        query = f"SELECT count(*) FROM approval_grants WHERE {_GRANT_MATCH}"  # noqa: S608 - a fixed condition
        (grants,) = connection.execute(query, {"principal": rename.old_principal, "name": rename.old}).fetchone()
    principal_rows = _principal_row_counts(connection, columns, rename.old_principal)
    return _EntityChanges(
        columns=columns,
        # Every grant naming the principal is dropped rather than renamed.
        principal_rows={table: count for table, count in principal_rows.items() if table != "approval_grants"},
        name_rows=name_rows,
        json_rows=json_rows,
        grants=grants,
    )


def _refuse_new_name_rows(
    connection: sqlite3.Connection,
    columns: dict[str, tuple[str, ...]],
    rename: _EntityRename,
) -> None:
    """Refuse when the journal has rows under a principal of the new name, for any Matrix ID, or turn records for it."""
    prefix = f"{rename.new}@"
    counts: dict[str, int] = {}
    for table, names in columns.items():
        # substr instead of LIKE: `_` in an agent name is a LIKE wildcard.
        match = " OR ".join(
            f"substr({quote_identifier(column)}, 1, :length) = :prefix"
            for column in names
            if column in _PRINCIPAL_COLUMNS
        )
        if match:
            query = f"SELECT count(*) FROM {quote_identifier(table)} WHERE {match}"  # noqa: S608 - quoted schema identifiers
            (count,) = connection.execute(query, {"length": len(prefix), "prefix": prefix}).fetchone()
            if count:
                counts[table] = count
    if "agent_name" in columns.get("turn_records", ()):
        (count,) = connection.execute(
            "SELECT count(*) FROM turn_records WHERE agent_name = ?",
            (rename.new,),
        ).fetchone()
        if count:
            counts["turn_records"] = count
    if counts:
        msg = (
            f"The event journal {rename.database} already has rows for agent {rename.new} ({_format_counts(counts)}), "
            f"left by an agent of that name, which renaming {rename.old} would mix with its own. "
            "Choose another new name."
        )
        raise RenameRefusedError(msg)


@dataclass(frozen=True)
class _JsonColumn:
    """A JSON column whose values name agents, and how MindRoom writes it."""

    table: str
    column: str
    # The column naming the entity that wrote the row, as its name or its principal; NULL when there is none.
    owner: str
    rewrite: Callable[[object, _EntityRename, str], object]
    dump: Callable[[object], str]


def _json_rewrites(
    connection: sqlite3.Connection,
    columns: dict[str, tuple[str, ...]],
    spec: _JsonColumn,
    rename: _EntityRename,
) -> list[tuple[int, str]]:
    """Return the rowid and rewritten text of each row whose JSON value the rename changes."""
    if spec.column not in columns.get(spec.table, ()):
        return []
    quoted = quote_identifier(spec.column)
    # Every value the rename changes holds the old name verbatim, as agent names hold no character JSON escapes.
    query = f"SELECT rowid, {spec.owner}, {quoted} FROM {quote_identifier(spec.table)} WHERE instr({quoted}, ?) > 0"  # noqa: S608 - quoted schema identifiers
    rewrites: list[tuple[int, str]] = []
    for rowid, owner, text in connection.execute(query, (rename.old,)).fetchall():
        try:
            value = json.loads(text)
        except (RecursionError, TypeError, ValueError):
            # MindRoom cannot read such a value either.
            continue
        # A principal is `<entity>@<Matrix ID>`, and entity names hold no `@`.
        renamed = spec.rewrite(value, rename, str(owner or "").partition("@")[0])
        if renamed != value:
            rewrites.append((rowid, spec.dump(renamed)))
    return rewrites


def _rename_entity(connection: sqlite3.Connection, *, rename: _EntityRename) -> None:
    """Rename the agent inside the caller's transaction, refusing again first; raising rolls everything back."""
    changes = _entity_changes(connection, rename)
    if changes.grants:
        connection.execute(
            f"DELETE FROM approval_grants WHERE {_GRANT_MATCH}",  # noqa: S608 - a fixed condition
            {"principal": rename.old_principal, "name": rename.old},
        )
    for (table, column), rewrites in changes.json_rows.items():
        update = f"UPDATE {quote_identifier(table)} SET {quote_identifier(column)} = ? WHERE rowid = ?"  # noqa: S608 - quoted schema identifiers
        connection.executemany(update, [(text, rowid) for rowid, text in rewrites])
    renames = [(table, column, rename.old, rename.new) for table, column in changes.name_rows]
    renames += [
        (table, column, rename.old_principal, rename.new_principal)
        for table, names in changes.columns.items()
        for column in names
        if column in _PRINCIPAL_COLUMNS
    ]
    for table, column, old, new in renames:
        quoted = quote_identifier(column)
        update = f"UPDATE {quote_identifier(table)} SET {quoted} = ? WHERE {quoted} = ?"  # noqa: S608 - quoted schema identifiers
        connection.execute(update, (new, old))


def _with_fields(value: object, transforms: Mapping[str, Callable[[object], object]]) -> object:
    """Return the mapping ``value`` with each transform applied to the field it names; anything else as it is."""
    if not isinstance(value, dict):
        return value
    record = cast("dict[str, object]", value)
    return {**record, **{key: transform(record[key]) for key, transform in transforms.items() if key in record}}


def _names(rename: _EntityRename, *fields: str) -> dict[str, Callable[[object], object]]:
    """Return transforms renaming the agent in each of ``fields``."""
    return dict.fromkeys(fields, rename.name)


def _mapped(value: object, transform: Callable[[object], object]) -> object:
    return [transform(item) for item in cast("list[object]", value)] if isinstance(value, list) else value


def _renamed_pair(pair: object, rename: _EntityRename) -> object:
    """Rename the agent as the member of a ``[member, model]`` pair."""
    if not isinstance(pair, list) or not pair:
        return pair
    items = cast("list[object]", pair)
    return [rename.name(items[0]), *items[1:]]


def _renamed_bindings(value: object, rename: _EntityRename) -> object:
    """Move the agent's delegation storage binding to the new name; its display name is its name (delegation/storage.py)."""
    if not isinstance(value, dict):
        return value
    bindings = cast("dict[str, object]", value)
    display_name = partial(_with_fields, transforms=_names(rename, "display_name"))
    return {
        rename.new if key == rename.old else key: display_name(binding) if key == rename.old else binding
        for key, binding in bindings.items()
    }


def _renamed_scope(value: object, rename: _EntityRename, owner: str) -> object:
    """Rename the agent's history scope, or the ad-hoc team scope listing it, of a response by ``owner``.

    An agent's scope id is its name. A configured team's scope id is the team's name, and only that team's bot
    answers in it (conversation_state_writer.py), so any other team scope is an ad-hoc one listing its members.
    """
    if not isinstance(value, dict):
        return value
    scope = cast("dict[str, object]", value)
    scope_id = scope.get("scope_id")
    if not isinstance(scope_id, str):
        return value
    if scope.get("kind") == "agent":
        return {**scope, "scope_id": rename.name(scope_id)}
    if scope.get("kind") != "team" or scope_id == owner:
        return value
    where = f"the event journal {rename.database}"
    new_scope_id = renamed_ad_hoc_scope_id(scope_id, old_name=rename.old, new_name=rename.new, where=where)
    return value if new_scope_id is None else {**scope, "scope_id": new_scope_id}


def _renamed_turn_record(value: object, rename: _EntityRename, owner: str) -> object:
    """Rename the agent as a turn's response owner and in its history scope (handled_turns.py's TurnRecordCodec)."""
    return _with_fields(
        value,
        {"response_owner": rename.name, "history_scope": partial(_renamed_scope, rename=rename, owner=owner)},
    )


def _renamed_continuation(value: object, rename: _EntityRename, owner: str) -> object:
    """Rename the agent in the context a paused response stores (approval_continuations.py's ``_context``)."""
    names = partial(_mapped, transform=rename.name)
    # A CLI approval call's requirements name the team member that raised them (agent_cli/approval.py).
    requirement = partial(_with_fields, transforms=_names(rename, "member_agent_id"))
    cli_call = {"requirements": partial(_mapped, transform=requirement), "external_requirement": requirement}
    return _with_fields(
        value,
        {
            "execution_identity": partial(_with_fields, transforms=_names(rename, *_IDENTITY_FIELDS)),
            "origin": partial(_with_fields, transforms=_names(rename, "sender_entity_name", "requester_entity_name")),
            "team_member_names": names,
            "team_member_model_names": partial(_mapped, transform=partial(_renamed_pair, rename=rename)),
            "mentioned_agents": names,
            "delegation_storage_bindings": partial(_renamed_bindings, rename=rename),
            "history_scope": partial(_renamed_scope, rename=rename, owner=owner),
            "cli_call": partial(_with_fields, transforms=cli_call),
        },
    )


def _renamed_job(value: object, rename: _EntityRename, _owner: str) -> object:
    """Rename the agent as owner of a background tool job, and as caller or child of a delegation job.

    A job's owner is an execution identity (tool_jobs/runtime.py's BackgroundJob). A delegation job is known by the
    agent it runs and snapshots its child (delegation/background.py, delegation/state.py's DelegationChild).
    """
    identity = partial(_with_fields, transforms=_names(rename, *_IDENTITY_FIELDS))
    job = _with_fields(value, {"owner": identity})
    if not isinstance(job, dict) or cast("dict[str, object]", job).get("kind") != "delegation":
        return job
    child = {
        **_names(rename, "caller_agent_name", "child_agent_name"),
        "execution_identity": identity,
        "storage_bindings": partial(_renamed_bindings, rename=rename),
    }
    adapter = partial(_with_fields, transforms={"child": partial(_with_fields, transforms=child)})
    return _with_fields(job, {**_names(rename, "tool_name"), "adapter": adapter})


def _compact_json(value: object) -> str:
    """Encode like approval_continuations.py's ``_json``."""
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


# Turn records and tool jobs are written with json.dumps' defaults (handled_turns.py, tool_jobs/runtime.py).
_JSON_COLUMNS = (
    _JsonColumn("turn_records", "record_json", "agent_name", _renamed_turn_record, json.dumps),
    _JsonColumn("approval_continuations", "context_json", "principal_id", _renamed_continuation, _compact_json),
    _JsonColumn("tool_jobs", "job_json", "NULL", _renamed_job, json.dumps),
)
