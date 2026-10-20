"""Tests for the scan that lists what still names the old identity after an agent account rename."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
import yaml

from mindroom.agent_rename.scan import _LeftoverReport, scan_leftovers
from mindroom.credentials.manager import scoped_credentials_dir_part
from mindroom.event_journal import EventJournalStore, EventKind, encode_thread_id
from mindroom.event_journal.opening import event_journal_sqlite_path
from tests.agent_rename_helpers import tree_snapshot

OLD = "@old:x"
NEW = "@new:x"
CREDENTIALS_PART = scoped_credentials_dir_part(OLD)
NEEDLES = (OLD, CREDENTIALS_PART)
ROOM = "!room:x"


def _write(root: Path, relative: str, content: str | bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def _scan(root: Path, *expected: Path) -> _LeftoverReport:
    return scan_leftovers(root, needles=NEEDLES, expected=frozenset(expected))


def _event(
    connection: sqlite3.Connection,
    event_id: str,
    *,
    sender: str,
    source: dict[str, str],
    principal: str = f"harry@{NEW}",
) -> None:
    connection.execute(
        "INSERT INTO journal_events (principal_id, event_id, room_id, thread_id, kind, sender, origin_server_ts, "
        "source_json, membership_epoch, state) VALUES (?, ?, ?, ?, ?, ?, 10, ?, 0, 'settled')",
        (principal, event_id, ROOM, encode_thread_id(None), EventKind.MESSAGE.value, sender, json.dumps(source)),
    )


async def _journal(root: Path) -> Path:
    """Create a journal with the real schema and return its path."""
    database = event_journal_sqlite_path(root)
    await EventJournalStore.open_sqlite(database).close()
    return database


@pytest.mark.asyncio
async def test_finds_paths_files_and_journal_cells(tmp_path: Path) -> None:
    """Path names, JSON and YAML contents and journal text cells that name the old identity are all listed."""
    (tmp_path / CREDENTIALS_PART).mkdir()
    _write(tmp_path, f"credentials/{CREDENTIALS_PART}.json", "{}")
    _write(tmp_path, "records/owner.json", json.dumps({"owner": OLD}))
    _write(tmp_path, "records/nested/deep/schedule.yaml", f'owner: "{OLD}"\n')
    _write(tmp_path, "config.yml", f"- {OLD}\n")
    _write(tmp_path, "records/clean.json", json.dumps({"owner": NEW}))
    _write(tmp_path, "records/other.yaml", "owner: '@older:x'\n")
    _write(tmp_path, "notes.txt", f"Not a data file, so not read: {OLD}")
    database = await _journal(tmp_path)
    with closing(sqlite3.connect(database)) as connection, connection:
        _event(connection, "$one", sender=OLD, source={"sender": OLD})
        _event(connection, "$two", sender=OLD, source={"sender": NEW})
        _event(connection, "$three", sender=NEW, source={"sender": NEW})

    report = _scan(tmp_path)

    assert report == _LeftoverReport(
        unexpected=(
            "config.yml",
            f"credentials/{CREDENTIALS_PART}.json",
            "journal: journal_events.sender (2 cells)",
            "journal: journal_events.source_json (1 cell)",
            CREDENTIALS_PART,
            "records/nested/deep/schedule.yaml",
            "records/owner.json",
        ),
        expected=(),
    )


def test_expected_paths_are_listed_separately(tmp_path: Path) -> None:
    """A path the rename reports as staying, or anything inside it, is expected; absolute and relative both work."""
    desktop = _write(tmp_path, "desktop_bridge/native_config.json", json.dumps({"controller": {"user_id": OLD}}))
    _write(tmp_path, "kept/a/history.json", json.dumps({"owner": OLD}))
    _write(tmp_path, "kept/b.yaml", f"owner: {OLD}\n")
    _write(tmp_path, "keptover/other.json", json.dumps({"owner": OLD}))
    _write(tmp_path, "stray.json", json.dumps({"owner": OLD}))

    report = _scan(tmp_path, desktop, Path("kept"), tmp_path / "elsewhere" / "stray.json")

    assert report == _LeftoverReport(
        unexpected=("keptover/other.json", "stray.json"),
        expected=("desktop_bridge/native_config.json", "kept/a/history.json", "kept/b.yaml"),
    )


def test_skips_backups_and_encryption_keys(tmp_path: Path) -> None:
    """Backups and the crypto store are not reported, nor are worker virtualenvs and caches."""
    hit = json.dumps({"owner": OLD})
    _write(tmp_path, f"agent-rename-backups/20261019T120000Z-rename-account/000/{CREDENTIALS_PART}/a.json", hit)
    _write(tmp_path, f"encryption_keys/{CREDENTIALS_PART}/store.json", hit)
    _write(tmp_path, "workers/w1/venv/lib/site-packages/pkg/data.json", hit)
    _write(tmp_path, "workers/w1/cache/uv/data.yaml", f"owner: {OLD}\n")
    _write(tmp_path, "workers/w1/workspace/cache/data.json", hit)
    # The same names elsewhere are ordinary directories and are scanned.
    _write(tmp_path, "other/venv/data.json", hit)
    _write(tmp_path, "other/cache/data.json", hit)
    _write(tmp_path, "other/encryption_keys/data.json", hit)
    _write(tmp_path, "other/agent-rename-backups/data.json", hit)
    _write(tmp_path, "workers/w1/workspace/data.json", hit)

    report = _scan(tmp_path)

    assert report.expected == ()
    assert report.unexpected == (
        "other/agent-rename-backups/data.json",
        "other/cache/data.json",
        "other/encryption_keys/data.json",
        "other/venv/data.json",
        "workers/w1/workspace/data.json",
    )


def test_matches_user_ids_only_where_the_journal_rewrite_would(tmp_path: Path) -> None:
    """Other users' IDs that merely start with the old one are not leftovers; punctuation and device suffixes end it."""
    _write(tmp_path, "other-users.json", json.dumps(["@old:x.example", "@old:x:8448", "@old:x-bot:x", "@older:x"]))
    _write(tmp_path, "punctuation.json", json.dumps({"body": f"ask {OLD}.", "device": f"{OLD}:DEVICEID"}))
    (tmp_path / "records" / OLD).mkdir(parents=True)
    (tmp_path / "records" / "@old:x.example").mkdir()

    report = _scan(tmp_path)

    assert report.unexpected == ("punctuation.json", f"records/{OLD}")


@pytest.mark.asyncio
async def test_journal_cells_follow_the_same_rule(tmp_path: Path) -> None:
    """A journal cell naming a longer user ID is not counted, and non-text columns are ignored."""
    database = await _journal(tmp_path)
    with closing(sqlite3.connect(database)) as connection, connection:
        _event(connection, "$one", sender="@old:x.example", source={"sender": "@older:x"})
        _event(connection, "$two", sender=NEW, source={"note": f"via {CREDENTIALS_PART}"})

    report = _scan(tmp_path)

    assert report.unexpected == ("journal: journal_events.source_json (1 cell)",)


def test_large_data_files_are_reported_not_read(tmp_path: Path) -> None:
    """A data file over 5 MB is named instead of read, so the operator knows it was not checked."""
    big = tmp_path / "history" / "big.json"
    big.parent.mkdir()
    with big.open("wb") as handle:
        handle.truncate(5 * 1024 * 1024 + 1)

    assert _scan(tmp_path).unexpected == ("history/big.json: not scanned, larger than 5 MB",)


def test_symlinks_are_never_followed(tmp_path: Path) -> None:
    """A link to something outside the storage root is neither walked nor read."""
    outside = tmp_path / "outside"
    _write(outside, "dir/data.json", json.dumps({"owner": OLD}))
    _write(outside, "file.json", json.dumps({"owner": OLD}))
    root = tmp_path / "storage"
    root.mkdir()
    (root / "linked-dir").symlink_to(outside / "dir", target_is_directory=True)
    (root / "linked-file.json").symlink_to(outside / "file.json")

    assert _scan(root) == _LeftoverReport(unexpected=(), expected=())


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads files whatever their mode")
def test_unreadable_files_and_directories_are_named(tmp_path: Path) -> None:
    """What the scan cannot open is listed, so a clean report never hides a place it could not look."""
    locked_file = _write(tmp_path, "locked.json", "{}")
    locked_directory = tmp_path / "locked"
    _write(locked_directory, "inside.json", "{}")
    locked_file.chmod(0)
    locked_directory.chmod(0)
    try:
        report = _scan(tmp_path)
    finally:
        locked_file.chmod(0o600)
        locked_directory.chmod(0o700)

    assert report.expected == ()
    assert report.unexpected == (
        "locked.json: cannot read (Permission denied)",
        "locked: cannot read (Permission denied)",
    )


@pytest.mark.asyncio
async def test_journal_is_read_without_writing_next_to_it(tmp_path: Path) -> None:
    """Rows only a leftover WAL holds are seen, and the live files stay byte-identical."""
    database = await _journal(tmp_path)
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA wal_autocheckpoint = 0")
    with connection:
        _event(connection, "$only-in-wal", sender=OLD, source={"sender": NEW})
    saved = {suffix: Path(f"{database}{suffix}").read_bytes() for suffix in ("", "-wal")}
    connection.close()
    for suffix, content in saved.items():
        Path(f"{database}{suffix}").write_bytes(content)
    before = tree_snapshot(tmp_path)

    assert _scan(tmp_path).unexpected == ("journal: journal_events.sender (1 cell)",)
    assert tree_snapshot(tmp_path) == before


def test_unreadable_journal_is_reported(tmp_path: Path) -> None:
    """A journal SQLite cannot read is named instead of silently counted as clean."""
    database = event_journal_sqlite_path(tmp_path)
    database.parent.mkdir(parents=True)
    database.write_bytes(b"not a database")

    (line,) = _scan(tmp_path).unexpected

    assert line.startswith("journal: cannot read tracking/event_journal.db: ")


# Renaming the agent `code` while the agent `code_team` stays: a spelling of `code_team` never counts as `code`.
NAME = "code"
OTHER = "code_team"
DIGEST_16 = "0123456789abcdef"
DIGEST_12 = "0123456789ab"
DIGEST_8 = "01234567"

# Where MindRoom keeps something under an agent's name: (what, a file to create there, what the scan reports for it).
PATH_SPELLINGS = {
    "state root": ("agents/code/notes.txt", "agents/code"),
    "private-instance root": (
        f"private_instances/scope-{DIGEST_16}/code/notes.txt",
        f"private_instances/scope-{DIGEST_16}/code",
    ),
    "session database": ("agents/x/sessions/code.db", "agents/x/sessions/code.db"),
    "learning database side file": ("learning/code.db-wal", "learning/code.db-wal"),
    "plugin state": ("plugins/ping/code.json", "plugins/ping/code.json"),
    "avatar": ("avatars/code.png", "avatars/code.png"),
    "handled-turn ledger": ("tracking/code_responded.json", "tracking/code_responded.json"),
    "imported ledger": ("tracking/code_responded.json.imported", "tracking/code_responded.json.imported"),
    "dispatch obligations": (
        f"tracking/dispatch_obligations-code-{DIGEST_12}.sqlite3",
        f"tracking/dispatch_obligations-code-{DIGEST_12}.sqlite3",
    ),
    "file-memory search index": (
        f"knowledge_db/file_memory_agent_code_{DIGEST_16}_{DIGEST_8}",
        f"knowledge_db/file_memory_agent_code_{DIGEST_16}_{DIGEST_8}",
    ),
    "legacy file-memory search index": (
        f"memory_search_db/agent_code_{DIGEST_16}",
        f"memory_search_db/agent_code_{DIGEST_16}",
    ),
    "file-memory scope": ("agents/x/memory_files/agent_code/notes.txt", "agents/x/memory_files/agent_code"),
    "team file-memory scope": (
        "agents/x/memory_files/team_helper+code/notes.txt",
        "agents/x/memory_files/team_helper+code",
    ),
    "shared worker root": (
        f"workers/v1_default_shared_code-{DIGEST_16}/notes.txt",
        f"workers/v1_default_shared_code-{DIGEST_16}",
    ),
    "requester worker root": (
        f"workers/v1_default_user_agent_@alice_example.org_code-{DIGEST_16}/notes.txt",
        f"workers/v1_default_user_agent_@alice_example.org_code-{DIGEST_16}",
    ),
}


def _scan_names(root: Path, *expected: Path, names: tuple[str, ...] = (NAME,)) -> _LeftoverReport:
    return scan_leftovers(root, needles=(), expected=frozenset(expected), names=names)


@pytest.mark.parametrize("spelling", PATH_SPELLINGS)
def test_paths_naming_the_agent_are_found(tmp_path: Path, spelling: str) -> None:
    """Each store path built from the old name is listed once, at the component that carries the name."""
    created, reported = PATH_SPELLINGS[spelling]
    _write(tmp_path, created, "")

    assert _scan_names(tmp_path) == _LeftoverReport(unexpected=(reported,), expected=())


@pytest.mark.parametrize("spelling", PATH_SPELLINGS)
def test_paths_naming_a_longer_agent_are_not_found(tmp_path: Path, spelling: str) -> None:
    """The same store paths of ``code_team`` stay unreported when ``code`` is renamed."""
    created, _ = PATH_SPELLINGS[spelling]
    _write(tmp_path, created.replace(NAME, OTHER), "")

    assert _scan_names(tmp_path) == _LeftoverReport(unexpected=(), expected=())


def test_other_names_near_the_agent_are_not_found(tmp_path: Path) -> None:
    """Names that contain the agent's name, differ in case, or only start like a store path are not the agent."""
    for relative in (
        "agents/codex/notes.txt",
        "agents/decode/notes.txt",
        "agents/Code/notes.txt",
        "agents/code-team/notes.txt",
        "agents/x/memory_files/team_helper+codex/notes.txt",
        "agents/x/memory_files/team_code_team+helper/notes.txt",
        "agents/x/memory_files/team_helper+mycode/notes.txt",
        "agents/x/memory_files/team_mycode+helper/notes.txt",
        "plugins/ping/code.json.bak",
        "plugins/ping/mycode.json",
        "plugins/ping/code.json5",
        f"tracking/code-{DIGEST_12}.sqlite3",
        f"workers/v1_default_shared_mycode-{DIGEST_16}/notes.txt",
        f"workers/v1_default_shared_code-{DIGEST_12}/notes.txt",
        f"workers/v1_default_shared_code-{DIGEST_16}0/notes.txt",
        f"knowledge_db/file_memory_agent_code_{DIGEST_12}",
    ):
        _write(tmp_path, relative, "")

    assert _scan_names(tmp_path).unexpected == ()


def test_oauth_directories_of_the_agent_are_found(tmp_path: Path) -> None:
    """The primary-runtime OAuth directories hashed from the name are found, and those of ``code_team`` are not."""
    agent_part, other_part = scoped_credentials_dir_part(NAME), scoped_credentials_dir_part(OTHER)
    _write(tmp_path, f"private_oauth/_agents/{agent_part}/google_credentials.sqlite3", "")
    _write(tmp_path, f"private_oauth/{scoped_credentials_dir_part(OLD)}/{agent_part}/google_credentials.sqlite3", "")
    _write(tmp_path, f"private_oauth/_agents/{other_part}/google_credentials.sqlite3", "")
    _write(tmp_path, f"private_oauth/_agents/{NAME}-{DIGEST_16}/google_credentials.sqlite3", "")

    assert _scan_names(tmp_path).unexpected == (
        f"private_oauth/_agents/{agent_part}",
        f"private_oauth/{scoped_credentials_dir_part(OLD)}/{agent_part}",
    )


def test_several_names_are_all_looked_for(tmp_path: Path) -> None:
    """Every name passed is scanned, and a path naming none of them stays out."""
    for name in ("code", "helper", "other"):
        _write(tmp_path, f"agents/{name}/notes.txt", "")

    assert _scan_names(tmp_path, names=("code", "helper")).unexpected == ("agents/code", "agents/helper")


def test_names_are_not_looked_for_unless_passed(tmp_path: Path) -> None:
    """Without ``names`` nothing about an agent's name is reported, whatever the paths and values hold."""
    _write(tmp_path, "agents/code/notes.txt", "")
    _write(tmp_path, "records.json", json.dumps({"agent": NAME}))

    assert scan_leftovers(tmp_path, needles=NEEDLES, expected=frozenset()) == _LeftoverReport(
        unexpected=(),
        expected=(),
    )


def test_names_and_user_ids_are_looked_for_together(tmp_path: Path) -> None:
    """A file that names both the old ID and the old name is listed once, beside the paths naming either."""
    _write(tmp_path, "records/both.json", json.dumps({"owner": OLD, "agent": NAME}))
    _write(tmp_path, "records/id.json", json.dumps({"owner": OLD}))
    _write(tmp_path, "records/name.json", json.dumps({"agent": NAME}))
    _write(tmp_path, "agents/code/notes.txt", "")

    report = scan_leftovers(tmp_path, needles=NEEDLES, expected=frozenset(), names=(NAME,))

    assert report.unexpected == ("agents/code", "records/both.json", "records/id.json", "records/name.json")


def test_names_in_expected_paths_are_listed_as_expected(tmp_path: Path) -> None:
    """A path the rename reports as staying is expected for names too, with everything below it."""
    _write(tmp_path, "plugins/ping/code.json", "{}")
    _write(tmp_path, "agents/code/kept.json", json.dumps({"agent": NAME}))
    _write(tmp_path, "records.json", json.dumps({"agent": NAME}))

    report = _scan_names(tmp_path, tmp_path / "plugins" / "ping" / "code.json", Path("agents/code"))

    assert report == _LeftoverReport(
        unexpected=("records.json",),
        expected=("agents/code", "agents/code/kept.json", "plugins/ping/code.json"),
    )


def test_names_are_not_looked_for_where_the_scan_never_enters(tmp_path: Path) -> None:
    """Backups, the crypto store and worker virtualenvs and caches stay unscanned for names too."""
    hit = json.dumps({"agent": NAME})
    _write(tmp_path, "agent-rename-backups/20261019T120000Z-rename/000/agents/code/a.json", hit)
    _write(tmp_path, "encryption_keys/code/store.json", hit)
    _write(tmp_path, "workers/w1/venv/lib/code/data.json", hit)
    _write(tmp_path, "workers/w1/cache/code/data.json", hit)

    assert _scan_names(tmp_path) == _LeftoverReport(unexpected=(), expected=())


# Strings MindRoom stores with an agent's name in them; each is the whole value of a field.
VALUE_SPELLINGS = {
    "name": "code",
    "memory scope": "agent_code",
    "history scope": "agent:code",
    "model override": "model:code",
    "team scope first": "team_code+helper",
    "team scope last": "team_helper+code",
    "team scope middle": "team_a+code+z",
    "team scope of one": "team_code",
    "team scope with requester": f"team_helper+code_requester_{DIGEST_12}",
    "principal": "code@@mindroom_code:example.org",
    "shared worker key": "v1:default:shared:code",
    "unscoped worker key": "v1:default:unscoped:code",
    "requester worker key": "v1:default:user_agent:@alice:example.org:code",
}
# What each of those looks like for the agent `code_team`.
OTHER_VALUE_SPELLINGS = {
    "name": "code_team",
    "memory scope": "agent_code_team",
    "history scope": "agent:code_team",
    "model override": "model:code_team",
    "team scope first": "team_code_team+helper",
    "team scope last": "team_helper+code_team",
    "team scope middle": "team_a+code_team+z",
    "team scope of one": "team_code_team",
    "team scope with requester": f"team_helper+code_team_requester_{DIGEST_12}",
    "principal": "code_team@@mindroom_code_team:example.org",
    "shared worker key": "v1:default:shared:code_team",
    "unscoped worker key": "v1:default:unscoped:code_team",
    "requester worker key": "v1:default:user_agent:@alice:example.org:code_team",
}


def _data_file(value: str, suffix: str) -> tuple[str, str]:
    quoted = json.dumps(value)
    if suffix == ".json":
        return f"records/data{suffix}", f'{{"outer": {{"items": [1, {{"field": {quoted}}}]}}}}'
    return f"records/data{suffix}", f"outer:\n  items:\n    - 1\n    - field: {quoted}\n"


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
@pytest.mark.parametrize("spelling", VALUE_SPELLINGS)
def test_values_naming_the_agent_are_found(tmp_path: Path, spelling: str, suffix: str) -> None:
    """A string value that is one of the agent's spellings is found at any depth of a JSON or YAML file."""
    relative, content = _data_file(VALUE_SPELLINGS[spelling], suffix)
    _write(tmp_path, relative, content)

    assert _scan_names(tmp_path) == _LeftoverReport(unexpected=(relative,), expected=())


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
@pytest.mark.parametrize("spelling", OTHER_VALUE_SPELLINGS)
def test_values_naming_a_longer_agent_are_not_found(tmp_path: Path, spelling: str, suffix: str) -> None:
    """The same spellings of ``code_team`` are not ``code``, so such a file stays unreported."""
    relative, content = _data_file(OTHER_VALUE_SPELLINGS[spelling], suffix)
    _write(tmp_path, relative, content)

    assert _scan_names(tmp_path) == _LeftoverReport(unexpected=(), expected=())


def test_values_that_only_mention_the_name_are_not_found(tmp_path: Path) -> None:
    """Prose around the name, another case, non-strings and names used as keys are not the agent's spellings."""
    document = {
        "prose": "use the code tool",
        "sentence": "agent:code said hello",
        "case": "Code",
        "longer": "codex",
        "team": "team_code_team+helper",
        "number": 1,
        "flag": True,
        "nothing": None,
        "code": "a key is not a value",
        "list": ["decode", ["agent_codex"]],
        "member": ["team_helper+mycode", "team_mycode+helper", "agent_mycode", "my_agent_code"],
        "tenant": "v1:code:shared:helper",
        "requester": "v1:default:user:@code:example.org",
        "other requester": "v1:default:user_agent:@code:example.org:helper",
    }
    _write(tmp_path, "records/mentions.json", json.dumps(document))
    _write(tmp_path, "records/mentions.yaml", yaml.safe_dump(document))

    assert _scan_names(tmp_path) == _LeftoverReport(unexpected=(), expected=())


def test_files_that_cannot_be_read_as_data_are_named(tmp_path: Path) -> None:
    """A JSON or YAML file that does not parse cannot be checked for names, so it is listed instead of passed."""
    _write(tmp_path, "records/broken.json", '{"agent": "code"')
    _write(tmp_path, "records/broken.yaml", "agent: [code\n")
    _write(tmp_path, "records/empty.json", "")
    _write(tmp_path, "records/blank.yaml", "\n")
    _write(tmp_path, "records/fine.json", "{}")

    assert _scan_names(tmp_path).unexpected == (
        "records/broken.json: not scanned for names, it is not valid JSON or YAML",
        "records/broken.yaml: not scanned for names, it is not valid JSON or YAML",
    )
    # Nothing is said about such files when no name is looked for.
    assert scan_leftovers(tmp_path, needles=NEEDLES, expected=frozenset()).unexpected == ()


def test_deeply_nested_json_is_named_not_crashed_on(tmp_path: Path) -> None:
    """A document nested deeper than the JSON parser allows is listed like any other file that cannot be checked."""
    _write(tmp_path, "records/deep.json", "[" * 100_000 + "]" * 100_000)

    assert _scan_names(tmp_path).unexpected == (
        "records/deep.json: not scanned for names, it is not valid JSON or YAML",
    )


def _principal_events(database: Path, *principals: str) -> None:
    with closing(sqlite3.connect(database)) as connection, connection:
        for index, principal in enumerate(principals):
            _event(connection, f"$event{index}", sender=NEW, source={"sender": NEW}, principal=principal)
            connection.execute("INSERT INTO approval_grant_locks (principal_id) VALUES (?)", (principal,))


@pytest.mark.asyncio
async def test_journal_principals_of_the_agent_are_found(tmp_path: Path) -> None:
    """A journal cell that starts with ``<name>@`` is the agent's principal; ``code_team``'s and ``harry``'s are not."""
    database = await _journal(tmp_path)
    _principal_events(database, f"code@{NEW}", f"code@{OLD}", f"code_team@{NEW}", f"harry@{NEW}", "codex@@x:y")

    report = _scan_names(tmp_path)

    assert report.unexpected == (
        "journal: approval_grant_locks.principal_id (2 cells)",
        "journal: journal_events.principal_id (2 cells)",
    )


@pytest.mark.asyncio
async def test_journal_cells_holding_the_name_are_found(tmp_path: Path) -> None:
    """A journal cell that is the agent's name, or one of its spellings, is found; a longer agent's is not."""
    database = await _journal(tmp_path)
    with closing(sqlite3.connect(database)) as connection, connection:
        for continuation, entity, invoking in (("$one", "code", "code_team"), ("$two", "agent:code", "codex")):
            connection.execute(
                "INSERT INTO approval_grant_cards (principal_id, continuation_id, continuation_generation, "
                "tool_call_id, delivery_id, scope_key, room_id, thread_id, requester_id, entity_name, invoking_agent, "
                "operation, responder_principal_id, responder_epoch, membership_epoch) "
                "VALUES (?, ?, 0, 'call', ?, 'scope', ?, ?, ?, ?, ?, 'run', ?, 0, 0)",
                (
                    f"harry@{NEW}",
                    continuation,
                    continuation,
                    ROOM,
                    encode_thread_id(None),
                    NEW,
                    entity,
                    invoking,
                    f"harry@{NEW}",
                ),
            )

    assert _scan_names(tmp_path).unexpected == ("journal: approval_grant_cards.entity_name (2 cells)",)


@pytest.mark.asyncio
async def test_journal_is_scanned_for_names_and_ids_together(tmp_path: Path) -> None:
    """A journal that names the old ID in one column and the old name in another lists both columns."""
    database = await _journal(tmp_path)
    _principal_events(database, f"code@{NEW}")
    with closing(sqlite3.connect(database)) as connection, connection:
        _event(connection, "$id", sender=OLD, source={"sender": NEW}, principal=f"harry@{NEW}")

    report = scan_leftovers(tmp_path, needles=NEEDLES, expected=frozenset(), names=(NAME,))

    assert report.unexpected == (
        "journal: approval_grant_locks.principal_id (1 cell)",
        "journal: journal_events.principal_id (1 cell)",
        "journal: journal_events.sender (1 cell)",
    )
