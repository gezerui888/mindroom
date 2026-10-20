"""Rewrite the event journal after a Matrix server renamed an agent's account in place.

The journal keys each bot's state by its principal, ``<agent>@<Matrix ID>``, and mirrors Matrix senders and payloads.
After the server rename, every event the server re-delivers carries the new ID.
A stored sender that differs from the re-delivered one fails ingestion with ``IngestionBatchIntegrityError``,
so the journal must hold the new ID everywhere the server now shows it.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from functools import partial
from pathlib import Path

from mindroom.agent_rename.steps import RenameRefusedError, SqliteEdit

_PRINCIPAL_COLUMNS = frozenset({"principal_id", "responder_principal_id"})
# Rows for replies MindRoom already sent to Matrix; discarding them would lose track of messages in the rooms.
_SENT_REPLY_TABLES = ("reply_messages", "matrix_delivery_outbox")
# Grants hash the old principal into their scope key, so a renamed grant could never match again.
_GRANT_TABLES = ("approval_grants", "approval_grant_cards")


def journal_account_steps(
    database: Path,
    *,
    agent: str,
    old_user_id: str,
    new_user_id: str,
    discard_new_principal: bool,
) -> list[SqliteEdit]:
    """Return the one edit that moves ``agent``'s journal state from ``old_user_id`` to ``new_user_id``.

    Refuses when rows under the new principal exist (MindRoom ran after the server rename)
    unless ``discard_new_principal`` is set, and always when they include replies already sent.
    The edit checks this again in its transaction, renames the principal, replaces the old ID in every text cell,
    drops the principal's sync consumer and approval grants, and refuses if a rewritten row would collide.
    A journal that was never created needs no edit.
    """
    if not database.exists():
        return []
    old_principal = f"{agent}@{old_user_id}"
    new_principal = f"{agent}@{new_user_id}"
    try:
        with closing(_connect_read_only(database)) as connection:
            new_rows = _principal_row_counts(connection, _text_columns(connection), new_principal)
    except sqlite3.Error as exc:
        msg = f"Cannot read the event journal {database}: {exc}"
        raise RenameRefusedError(msg) from exc
    _refuse_new_principal_rows(new_rows, new_principal=new_principal, discard=discard_new_principal)

    discarded = f"delete the rows already under {new_principal} ({_format_counts(new_rows)}), " if new_rows else ""
    description = (
        f"Rewrite the event journal {database}: {discarded}rename principal {old_principal} to {new_principal}, "
        f"replace {old_user_id} with {new_user_id} in every text cell, "
        "and drop the principal's sync consumer and approval grants"
    )
    edit = partial(
        _rewrite,
        old_user_id=old_user_id,
        new_user_id=new_user_id,
        old_principal=old_principal,
        new_principal=new_principal,
        discard_new_principal=discard_new_principal,
    )
    return [SqliteEdit(database=database, edit=edit, description=description)]


def _connect_read_only(database: Path) -> sqlite3.Connection:
    """Open the journal for reading without creating anything next to it.

    Even a read-only connection to a WAL database creates its ``-wal`` and ``-shm`` files.
    MindRoom removes both when it closes cleanly, leaving every commit in the main file, which ``immutable`` reads
    exactly. Only when a ``-wal`` file is present can commits live outside the main file, so read through it then.
    """
    uri = database.absolute().as_uri()
    if Path(f"{database}-wal").exists():
        return sqlite3.connect(f"{uri}?mode=ro", uri=True)
    return sqlite3.connect(f"{uri}?immutable=1", uri=True)


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _holds_text(declared_type: str) -> bool:
    """Whether SQLite keeps text stored in a column of this type as text: TEXT affinity, or no type at all."""
    declared = declared_type.upper()
    return not declared or ("INT" not in declared and any(word in declared for word in ("CHAR", "CLOB", "TEXT")))


def _text_columns(connection: sqlite3.Connection) -> dict[str, tuple[str, ...]]:
    """Map every ordinary table to its text columns.

    SQLite's own tables are skipped, and so are virtual tables and their shadow tables,
    which only the virtual table's module may write.
    """
    tables = connection.execute("SELECT name, sql FROM sqlite_master WHERE type = 'table'").fetchall()
    virtual = [name for name, sql in tables if (sql or "").lstrip().upper().startswith("CREATE VIRTUAL TABLE")]
    columns: dict[str, tuple[str, ...]] = {}
    for name, _ in tables:
        if name.startswith("sqlite_") or any(name == table or name.startswith(f"{table}_") for table in virtual):
            continue
        rows = connection.execute("SELECT name, type FROM pragma_table_info(?)", (name,)).fetchall()
        columns[name] = tuple(column for column, declared_type in rows if _holds_text(declared_type))
    return columns


def _principal_match(columns: tuple[str, ...]) -> str:
    """Return a condition matching rows whose principal columns name ``:principal``, or ``""`` if there are none."""
    return " OR ".join(f"{_quote(column)} = :principal" for column in columns if column in _PRINCIPAL_COLUMNS)


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
        query = f"SELECT count(*) FROM {_quote(table)} WHERE {match}"  # noqa: S608 - quoted schema identifiers
        (count,) = connection.execute(query, {"principal": principal}).fetchone()
        if count:
            counts[table] = count
    return counts


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{table}: {count}" for table, count in sorted(counts.items()))


def _refuse_new_principal_rows(counts: dict[str, int], *, new_principal: str, discard: bool) -> None:
    if not counts:
        return
    if any(table in counts for table in _SENT_REPLY_TABLES):
        msg = (
            f"The event journal already has rows for {new_principal} ({_format_counts(counts)}), "
            "including replies MindRoom already sent under the new ID, "
            "which --discard-new-principal does not delete."
        )
        raise RenameRefusedError(msg)
    if not discard:
        msg = (
            f"The event journal already has rows for {new_principal} because MindRoom ran after the server rename "
            f"({_format_counts(counts)}). Rerun with --discard-new-principal to delete them."
        )
        raise RenameRefusedError(msg)


def _rewrite(
    connection: sqlite3.Connection,
    *,
    old_user_id: str,
    new_user_id: str,
    old_principal: str,
    new_principal: str,
    discard_new_principal: bool,
) -> None:
    """Move the journal to the new ID inside the caller's transaction; raising rolls everything back."""
    columns = _text_columns(connection)
    new_rows = _principal_row_counts(connection, columns, new_principal)
    _refuse_new_principal_rows(new_rows, new_principal=new_principal, discard=discard_new_principal)
    _delete_principal_rows(connection, columns, tables=tuple(new_rows), principal=new_principal)
    _replace_identity(
        connection,
        columns,
        {"old_principal": old_principal, "new_principal": new_principal, "old": old_user_id, "new": new_user_id},
    )
    if "room_member_joins" in columns:
        # Any old-ID row left here collided with a row the bot already recorded for the new ID.
        connection.execute("DELETE FROM room_member_joins WHERE user_id = ?", (old_user_id,))
    if "matrix_sync_consumers" in columns:
        # The consumer binds a sync stream to the old login's device; the new device must bind a fresh one.
        connection.execute("DELETE FROM matrix_sync_consumers WHERE principal_id = ?", (new_principal,))
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
            delete = f"DELETE FROM {_quote(table)} WHERE {match}"  # noqa: S608 - quoted schema identifiers
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
            quoted_table, quoted = _quote(table), _quote(column)
            if column in _PRINCIPAL_COLUMNS:
                renames.append(f"UPDATE {quoted_table} SET {quoted} = :new_principal WHERE {quoted} = :old_principal")  # noqa: S608 - quoted schema identifiers
            else:
                # A row whose rewrite would collide keeps the old ID, and the leftover check refuses it.
                replacements.append(
                    f"UPDATE OR IGNORE {quoted_table} SET {quoted} = replace({quoted}, :old, :new) "  # noqa: S608 - quoted schema identifiers
                    f"WHERE instr({quoted}, :old) > 0",
                )
    for statement in (*renames, *replacements):
        connection.execute(statement, names)


def _refuse_leftovers(
    connection: sqlite3.Connection,
    columns: dict[str, tuple[str, ...]],
    *,
    old_user_id: str,
    new_user_id: str,
) -> None:
    """Refuse when any text cell still holds the old ID, naming each table and column."""
    leftovers: list[str] = []
    for table, table_columns in columns.items():
        if not table_columns:
            continue
        sums = ", ".join(f"sum(instr({_quote(column)}, :old) > 0)" for column in table_columns)
        counts = connection.execute(f"SELECT {sums} FROM {_quote(table)}", {"old": old_user_id}).fetchone()  # noqa: S608 - quoted schema identifiers
        leftovers.extend(
            f"{table}.{column} ({count})" for column, count in zip(table_columns, counts, strict=True) if count
        )
    if leftovers:
        msg = (
            f"Rewriting the event journal would leave {old_user_id} in {', '.join(leftovers)}, "
            f"most likely in rows that would collide with rows already holding {new_user_id}. "
            "The event journal was left unchanged."
        )
        raise RenameRefusedError(msg)
