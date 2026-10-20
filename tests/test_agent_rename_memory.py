"""Tests for carrying an agent's long-term memory, in mem0's Chroma stores and in memory files, to its new name."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from chromadb.api.client import Client
from chromadb.config import Settings
from mem0.vector_stores.chroma import ChromaDB

from mindroom.agent_rename.entity_state import entity_state_steps
from mindroom.agent_rename.memory import memory_steps
from mindroom.agent_rename.steps import (
    BACKUPS_DIRECTORY_NAME,
    DirectoryEdit,
    FileRewrite,
    MovePath,
    RenameApplyError,
    RenamePlan,
    RenameRefusedError,
    apply_plan,
)
from mindroom.config.agent import AgentConfig, TeamConfig
from mindroom.config.main import Config
from mindroom.memory._file_backend import _append_scope_memory_entry, _search_scope_memory_entries
from mindroom.memory._policy import (
    allowed_scope_storage_paths,
    resolve_file_memory_resolution,
    storage_paths_for_scope_user_id,
)
from mindroom.memory.config import _get_memory_config, _install_chroma_similarity_scores
from mindroom.memory_scope_ids import agent_name_from_scope_user_id, agent_scope_user_id, build_team_user_id
from mindroom.tool_system.worker_routing import agent_state_root_path
from tests.agent_rename_helpers import tree_snapshot
from tests.conftest import test_runtime_paths

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from mindroom.agent_rename.steps import Step
    from mindroom.constants import RuntimePaths
    from mindroom.memory._shared import FileMemoryResolution

OLD = "code"
NEW = "coder"
# Its name extends the renamed agent's, so a substring match would wrongly touch it.
PREFIX = "code_team"
HELPER = "helper"
NOW = datetime(2026, 10, 20, 9, 30, 0, tzinfo=UTC)
# A team formed on the fly, never configured; MindRoom stores its memory under every member's state root.
AD_HOC = [OLD, PREFIX, HELPER]

# (scope user id, memory text, the metadata MindRoom adds), written to every store MindRoom writes the scope to.
_MEMORIES = (
    (agent_scope_user_id(OLD), "code prefers tabs", {"type": "conversation", "agent": OLD}),
    (build_team_user_id([OLD, HELPER]), "crew ships on fridays", {"is_team": True, "team_members": [OLD, HELPER]}),
    (build_team_user_id(AD_HOC), "pairing notes", {"is_team": True, "team_members": AD_HOC}),
    (agent_scope_user_id(PREFIX), "code_team reviews", {"type": "conversation", "agent": PREFIX}),
    (agent_scope_user_id(HELPER), "helper drafts", {"type": "conversation", "agent": HELPER}),
    (build_team_user_id([PREFIX, HELPER]), "duo standup", {"is_team": True, "team_members": [PREFIX, HELPER]}),
)
_TEXTS = [text for _, text, _ in _MEMORIES]


def _config(agent: str) -> Config:
    """Return the config naming the renamed agent ``agent``, a member of team ``crew`` with ``helper``."""
    return Config(
        agents={name: AgentConfig(display_name=name.title()) for name in (agent, PREFIX, HELPER)},
        teams={"crew": TeamConfig(display_name="Crew", role="Ships releases", agents=[agent, HELPER])},
    )


def _vector(text: str) -> list[float]:
    return [1.0 if candidate == text else 0.0 for candidate in _TEXTS]


def _client(path: str) -> Client:
    """Return a client for the Chroma store at ``path`` with mem0's settings, which releases the store when closed."""
    return Client(settings=Settings(anonymized_telemetry=False, is_persistent=True, persist_directory=path))


@contextmanager
def _store(root: Path, config: Config, runtime_paths: RuntimePaths) -> Iterator[ChromaDB]:
    """Open the mem0 Chroma store MindRoom keeps at a state root, and close it again like a stopped MindRoom."""
    vector_store = _get_memory_config(root, config, runtime_paths)["vector_store"]["config"]
    with _client(vector_store["path"]) as client:
        store = ChromaDB(collection_name=vector_store["collection_name"], client=client)
        _install_chroma_similarity_scores(store)
        yield store


def _file_resolution(scope: str, root: Path, config: Config, runtime_paths: RuntimePaths) -> FileMemoryResolution:
    """Resolve a scope's file memory at ``root`` the way the file backend does for reads and writes."""
    return resolve_file_memory_resolution(
        root,
        config,
        runtime_paths,
        agent_name=agent_name_from_scope_user_id(scope),
        original_storage_path=runtime_paths.storage_root,
    )


def _install(tmp_path: Path, memories: tuple[tuple[str, str, dict[str, Any]], ...] = _MEMORIES) -> RuntimePaths:
    """Write each memory to mem0 and to file memory at every state root MindRoom stores its scope under."""
    runtime_paths = test_runtime_paths(tmp_path)
    config = _config(OLD)
    for index, (scope, text, metadata) in enumerate(memories):
        for root in storage_paths_for_scope_user_id(scope, runtime_paths.storage_root, config, runtime_paths):
            with _store(root, config, runtime_paths) as store:
                payload = {"data": text, "user_id": scope, **metadata}
                store.insert(vectors=[_vector(text)], payloads=[payload], ids=[f"memory-{index}"])
            resolution = _file_resolution(scope, root, config, runtime_paths)
            _append_scope_memory_entry(scope, text, resolution, config)
    return runtime_paths


def _plan(runtime_paths: RuntimePaths) -> list[Step]:
    """Plan like `mindroom agents rename`: the state root moves first, then the memory inside and beside it."""
    return entity_state_steps(runtime_paths, old_name=OLD, new_name=NEW) + memory_steps(
        runtime_paths,
        config=_config(OLD),
        old_name=OLD,
        new_name=NEW,
    )


def _apply(runtime_paths: RuntimePaths, steps: list[Step]) -> None:
    apply_plan(
        RenamePlan(storage_root=runtime_paths.storage_root, steps=tuple(steps), reports=()),
        kind="rename",
        now=NOW,
    )


def _visible(caller: str | list[str], config: Config, runtime_paths: RuntimePaths) -> Counter[tuple[str, str]]:
    """Count the (scope, text) memories ``caller`` reads from mem0, over every store MindRoom's scope policy reads."""
    seen: Counter[tuple[str, str]] = Counter()
    for scope, root in allowed_scope_storage_paths(caller, runtime_paths.storage_root, config, runtime_paths):
        with _store(root, config, runtime_paths) as store:
            seen.update((scope, hit.payload["data"]) for hit in store.list(filters={"user_id": scope}, top_k=100)[0])
    return seen


def _visible_files(caller: str | list[str], config: Config, runtime_paths: RuntimePaths) -> Counter[tuple[str, str]]:
    """Count the (scope, text) memories ``caller`` finds in file memory, searching each memory's own words."""
    seen: Counter[tuple[str, str]] = Counter()
    for scope, root in allowed_scope_storage_paths(caller, runtime_paths.storage_root, config, runtime_paths):
        resolution = _file_resolution(scope, root, config, runtime_paths)
        for text in _TEXTS:
            hits = _search_scope_memory_entries(scope, text, resolution, config, limit=10)
            seen.update((scope, hit["memory"]) for hit in hits if hit["memory"] == text)
    return seen


def _scope_ids_in_store(root: Path) -> set[str]:
    with _client(str(root / "chroma")) as client:
        return {
            str(metadata["user_id"])
            for collection in client.list_collections()
            for metadata in collection.get(include=["metadatas"])["metadatas"] or []
        }


def _root(runtime_paths: RuntimePaths, agent: str) -> Path:
    return agent_state_root_path(runtime_paths.storage_root, agent)


def _step_paths(step: Step) -> list[Path]:
    if isinstance(step, MovePath):
        return [step.source, step.destination]
    if isinstance(step, DirectoryEdit):
        return [step.path]
    msg = f"Unexpected memory step {step}"
    raise AssertionError(msg)


def _without_backups(snapshot: dict[str, bytes | str]) -> dict[str, bytes | str]:
    return {key: value for key, value in snapshot.items() if not key.startswith(BACKUPS_DIRECTORY_NAME)}


def test_memories_searchable_under_new_scope_ids(tmp_path: Path) -> None:
    """After the rename, the agent reads exactly the memories it read before, under its new scope ids."""
    runtime_paths = _install(tmp_path)
    before = _visible(OLD, _config(OLD), runtime_paths)
    files_before = _visible_files(OLD, _config(OLD), runtime_paths)
    untouched = tree_snapshot(runtime_paths.storage_root)

    steps = _plan(runtime_paths)

    assert tree_snapshot(runtime_paths.storage_root) == untouched
    _apply(runtime_paths, steps)

    expected = Counter(
        {(agent_scope_user_id(NEW), "code prefers tabs"): 1, ("team_coder+helper", "crew ships on fridays"): 2},
    )
    assert before == Counter({("agent_code", "code prefers tabs"): 1, ("team_code+helper", "crew ships on fridays"): 2})
    assert _visible(NEW, _config(NEW), runtime_paths) == expected
    assert files_before == before
    assert _visible_files(NEW, _config(NEW), runtime_paths) == expected

    with _store(_root(runtime_paths, NEW), _config(NEW), runtime_paths) as store:
        hits = store.search(
            query="tabs",
            vectors=_vector("code prefers tabs"),
            top_k=3,
            filters={"user_id": agent_scope_user_id(NEW)},
        )
        assert [(hit.payload["data"], hit.payload["agent"]) for hit in hits] == [("code prefers tabs", NEW)]
        assert hits[0].score == pytest.approx(1.0)
        assert store.search(query="tabs", vectors=_vector("code prefers tabs"), filters={"user_id": "agent_code"}) == []
    assert not (_root(runtime_paths, NEW) / "memory_files" / agent_scope_user_id(OLD)).exists()


def test_team_scopes_recomputed(tmp_path: Path) -> None:
    """Configured and ad-hoc team scopes naming the agent get their sorted member list with the new name everywhere."""
    runtime_paths = _install(tmp_path)
    ad_hoc_before = _visible(AD_HOC, _config(OLD), runtime_paths)

    _apply(runtime_paths, _plan(runtime_paths))

    ad_hoc = [NEW, PREFIX, HELPER]
    # Sorted again: the new name sorts after code_team, the old one before it.
    assert build_team_user_id(ad_hoc) == "team_code_team+coder+helper"
    assert ad_hoc_before == Counter({("team_code+code_team+helper", "pairing notes"): 3})
    expected = Counter({("team_code_team+coder+helper", "pairing notes"): 3})
    assert _visible(ad_hoc, _config(NEW), runtime_paths) == expected
    assert _visible_files(ad_hoc, _config(NEW), runtime_paths) == expected

    old_scopes = {"agent_code", "team_code+helper", "team_code+code_team+helper"}
    expected_scopes = {
        NEW: {"agent_coder", "team_coder+helper", "team_code_team+coder+helper"},
        PREFIX: {"agent_code_team", "team_code_team+coder+helper", "team_code_team+helper"},
        HELPER: {"agent_helper", "team_coder+helper", "team_code_team+coder+helper", "team_code_team+helper"},
    }
    for agent, scopes in expected_scopes.items():
        root = _root(runtime_paths, agent)
        assert _scope_ids_in_store(root) == scopes
        assert {path.name for path in (root / "memory_files").iterdir()} == scopes
        assert not scopes & old_scopes
    with _store(_root(runtime_paths, HELPER), _config(NEW), runtime_paths) as store:
        team_members = {
            hit.payload["user_id"]: hit.payload["team_members"]
            for hit in store.list(top_k=100)[0]
            if "team_members" in hit.payload
        }
    assert team_members == {
        "team_coder+helper": [NEW, HELPER],
        "team_code_team+coder+helper": [NEW, PREFIX, HELPER],
        "team_code_team+helper": [PREFIX, HELPER],
    }


def test_prefix_agent_untouched(tmp_path: Path) -> None:
    """Renaming code leaves code_team's stores alone when no team scope there names code."""
    memories = tuple(memory for memory in _MEMORIES if memory[0] != build_team_user_id(AD_HOC))
    runtime_paths = _install(tmp_path, memories)
    prefix_root = _root(runtime_paths, PREFIX)
    before = tree_snapshot(prefix_root)

    steps = _plan(runtime_paths)

    assert any(isinstance(step, DirectoryEdit) for step in steps)
    assert not [path for step in steps for path in _step_paths(step) if path.is_relative_to(prefix_root)]
    _apply(runtime_paths, steps)
    assert tree_snapshot(prefix_root) == before
    assert _visible(PREFIX, _config(NEW), runtime_paths) == Counter({("agent_code_team", "code_team reviews"): 1})


def test_failure_after_any_step_restores_the_tree(tmp_path: Path) -> None:
    """A failure after any number of completed steps puts every chroma store and memory file back byte for byte."""
    probe = _plan(_install(tmp_path / "probe"))
    assert {type(step) for step in probe} == {MovePath, DirectoryEdit}

    def fail(content: bytes) -> bytes:  # noqa: ARG001
        msg = "boom"
        raise RuntimeError(msg)

    for completed in range(len(probe) + 1):
        runtime_paths = _install(tmp_path / f"run-{completed}")
        sentinel = runtime_paths.storage_root / "sentinel.txt"
        sentinel.write_text("sentinel")
        original = tree_snapshot(runtime_paths.storage_root)
        steps = [*_plan(runtime_paths)[:completed], FileRewrite(sentinel, fail, "fail")]

        with pytest.raises(RenameApplyError, match="fail"):
            _apply(runtime_paths, steps)

        assert _without_backups(tree_snapshot(runtime_paths.storage_root)) == original


@pytest.mark.parametrize("taken", ["chroma", "memory_files"])
def test_existing_new_scope_refuses(tmp_path: Path, taken: str) -> None:
    """A store that already holds the new scope id refuses before anything is written."""
    runtime_paths = _install(tmp_path)
    old_root = _root(runtime_paths, OLD)
    if taken == "chroma":
        with _store(old_root, _config(OLD), runtime_paths) as store:
            payload = {"data": "a former coder's memory", "user_id": agent_scope_user_id(NEW)}
            store.insert(vectors=[_vector("code prefers tabs")], payloads=[payload], ids=["stale"])
    else:
        (old_root / "memory_files" / agent_scope_user_id(NEW)).mkdir()
    before = tree_snapshot(runtime_paths.storage_root)

    with pytest.raises(RenameRefusedError, match="agent_coder"):
        memory_steps(runtime_paths, config=_config(OLD), old_name=OLD, new_name=NEW)

    assert tree_snapshot(runtime_paths.storage_root) == before


def test_agent_without_memory_plans_nothing(tmp_path: Path) -> None:
    """No store names the agent, so there is nothing to plan."""
    runtime_paths = test_runtime_paths(tmp_path)
    _root(runtime_paths, OLD).mkdir(parents=True)

    assert memory_steps(runtime_paths, config=_config(OLD), old_name=OLD, new_name=NEW) == []
