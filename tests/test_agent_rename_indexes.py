"""Tests for dropping the derived indexes named after an agent and reporting the state its rename leaves as it is."""

from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from mem0.memory.main import _build_session_scope
from mem0.memory.storage import SQLiteManager

from mindroom.agent_rename.indexes import index_steps
from mindroom.agent_rename.steps import BACKUPS_DIRECTORY_NAME, DropPath, RenamePlan, apply_plan
from mindroom.config.agent import AgentConfig, AgentPrivateConfig
from mindroom.config.knowledge import KnowledgeBaseConfig, KnowledgeGitConfig
from mindroom.config.main import Config, load_config
from mindroom.constants import resolve_runtime_paths, tracking_dir
from mindroom.knowledge.file_memory_knowledge import resolve_agent_file_memory_knowledge, resolve_file_memory_knowledge
from mindroom.knowledge.indexing_config import knowledge_git_dir
from mindroom.knowledge.registry import published_index_storage_path, resolve_published_index_key
from mindroom.knowledge_graph.ingestion import KnowledgeGraphIngestionWorker
from mindroom.knowledge_graph.scope import graph_scope
from mindroom.legacy_handled_turns import legacy_responses_file_path
from mindroom.memory_scope_ids import agent_scope_user_id
from mindroom.runtime_resolution import shared_knowledge_path
from mindroom.tool_system.runtime_context import get_plugin_state_root
from mindroom.tool_system.worker_routing import agent_state_root_path, build_requester_execution_identity
from tests.agent_rename_helpers import tree_snapshot

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from mindroom.agent_rename.steps import Step
    from mindroom.constants import RuntimePaths

OLD = "code"
NEW = "coder"
# Its name extends the renamed agent's, so a substring match would wrongly touch it.
PREFIX = "code_team"
HELPER = "helper"
REQUESTER = "@alice:localhost"
NOW = datetime(2026, 10, 20, 9, 30, 0, tzinfo=UTC)
_SCHEDULED = "Scheduled tasks"
_GRAPH = "knowledge graph"
_GIT = "Git directory"
_MEM0 = "mem0's history database"
_PLUGINS = "plugin state paths"
_CONFIG_LINES = "config lines"
_LEGACY = "older MindRoom versions"


def _runtime_paths(tmp_path: Path, *, mem0_dir: bool = True) -> RuntimePaths:
    """Return a runtime context whose mem0 directory is inside ``tmp_path``, so no test reads the real one."""
    config_path = tmp_path / "config.yaml"
    if not config_path.exists():
        config_path.write_text("router:\n  model: default\n", encoding="utf-8")
    process_env = {"MATRIX_HOMESERVER": "http://localhost:8008", "MINDROOM_NAMESPACE": ""}
    if mem0_dir:
        process_env["MEM0_DIR"] = str(tmp_path / "mem0")
    return resolve_runtime_paths(
        config_path=config_path,
        storage_path=tmp_path / "mindroom_data",
        process_env=process_env,
    )


def _config(**agent_fields: object) -> Config:
    """Return a config with the renamed agent, the agent whose name extends it, and a helper."""
    return Config(
        agents={name: AgentConfig(display_name=name.title(), **agent_fields) for name in (OLD, PREFIX, HELPER)},
        models={"default": {"provider": "openai", "id": "gpt-5"}},
        knowledge_graph={"model": "default"},
    )


def _plan(runtime_paths: RuntimePaths, config: Config) -> tuple[list[Step], list[str]]:
    return index_steps(runtime_paths, config=config, old_name=OLD, new_name=NEW)


def _apply(runtime_paths: RuntimePaths, steps: list[Step]) -> None:
    plan = RenamePlan(storage_root=runtime_paths.storage_root, steps=tuple(steps), reports=())
    apply_plan(plan, kind="rename", now=NOW)


def _report(reports: list[str], marker: str) -> str | None:
    """Return the one report containing ``marker``, or ``None`` when none does."""
    found = [report for report in reports if marker in report]
    assert len(found) <= 1, found
    return found[0] if found else None


def _listed(report: str | None) -> set[str]:
    """Return the paths a report lists after its last colon."""
    assert report is not None
    return set(report.rsplit(": ", 1)[1].removesuffix(".").split(", "))


def _mkdir(path: Path) -> Path:
    """Create ``path`` with one file in it, like a store that holds data."""
    path.mkdir(parents=True)
    (path / "data").write_bytes(b"data")
    return path


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"data")
    return path


def _file_memory_index(agent: str, config: Config, runtime_paths: RuntimePaths, *, root: Path | None = None) -> Path:
    """Create the store of an agent's file-memory search index where MindRoom keeps it, for its root or ``root``."""
    if root is None:
        resolution = resolve_agent_file_memory_knowledge(agent, config, runtime_paths, None)
        assert resolution is not None
    else:
        resolution = resolve_file_memory_knowledge(
            scope_user_id=agent_scope_user_id(agent),
            root=root,
            config=config,
            search_config=config.resolve_entity(agent).memory_search,
        )
    key = resolve_published_index_key(resolution.base_id, config=resolution.config, runtime_paths=runtime_paths)
    return _mkdir(published_index_storage_path(key))


def _private_knowledge_index(agent: str, config: Config, runtime_paths: RuntimePaths) -> Path:
    """Create the store of a private agent's knowledge index where MindRoom keeps it for one requester."""
    base_id = config.resolve_entity(agent).private_knowledge_base_id
    assert base_id is not None
    identity = build_requester_execution_identity(agent_name=agent, runtime_paths=runtime_paths, requester_id=REQUESTER)
    key = resolve_published_index_key(
        base_id,
        config=config,
        runtime_paths=runtime_paths,
        execution_identity=identity,
        create=True,
    )
    return _mkdir(published_index_storage_path(key))


def _start_graph(agent: str, config: Config, runtime_paths: RuntimePaths) -> str:
    """Start an agent's knowledge graph the way ingestion does, which saves its progress file; return its name."""
    graph = graph_scope(agent, config, runtime_paths, None).graph_name
    worker = KnowledgeGraphIngestionWorker(
        runtime_paths=runtime_paths,
        config_provider=lambda: config,
        store=None,
        redacted_event_ids=None,
    )
    worker._load_state(graph, time.time())
    return graph


def _git_config(agents: tuple[str, ...]) -> Config:
    """Return a config with a Git-backed knowledge base in the workspace of each agent in ``agents``."""
    config = _config()
    bases = {
        f"{agent}_docs": KnowledgeBaseConfig(
            path=f"${{MINDROOM_STORAGE_PATH}}/agents/{agent}/workspace/docs",
            git=KnowledgeGitConfig(repo_url=f"https://example.org/{agent}.git"),
        )
        for agent in agents
    }
    return config.model_copy(update={"knowledge_bases": bases})


def _knowledge_git(base_id: str, config: Config, runtime_paths: RuntimePaths) -> Path:
    """Create the Git directory MindRoom keeps for a knowledge base's checkout."""
    checkout = shared_knowledge_path(config.knowledge_bases[base_id].path, runtime_paths)
    return _mkdir(knowledge_git_dir(runtime_paths.storage_root, checkout))


def _mem0_messages(database: Path, agent: str) -> None:
    """Save a message of an agent's memory scope in mem0's history database, as mem0's add does."""
    database.parent.mkdir(parents=True, exist_ok=True)
    manager = SQLiteManager(str(database))
    try:
        session_scope = _build_session_scope({"user_id": agent_scope_user_id(agent)})
        manager.save_messages([{"role": "user", "content": f"{agent} likes tabs"}], session_scope)
    finally:
        manager.close()


def _legacy_files(runtime_paths: RuntimePaths, agent: str) -> list[Path]:
    """Create the files older MindRoom versions kept for an agent, which no code reads any more."""
    storage = runtime_paths.storage_root
    principal = hashlib.sha256(f"@mindroom_{agent}:localhost".encode()).hexdigest()[:12]
    # The memory search index of file memory before #1165: memory_search_db/<scope id>_<16 hex digits>.
    scope = agent_scope_user_id(agent)
    search_digest = hashlib.sha256(f"{scope}:{agent_state_root_path(storage, agent)}".encode()).hexdigest()[:16]
    responded = legacy_responses_file_path(storage, agent)
    return [
        # The dispatch obligations before the event journal (#1800): dispatch_obligations-<entity>-<12 hex>.sqlite3.
        _touch(tracking_dir(runtime_paths) / f"dispatch_obligations-{agent}-{principal}.sqlite3"),
        _touch(tracking_dir(runtime_paths) / f"dispatch_obligations-{agent}-{principal}.sqlite3-wal"),
        _touch(storage / "sessions" / f"{agent}.db"),
        _touch(storage / f"{agent}.db"),
        _mkdir(storage / "memory_search_db" / f"{scope}_{search_digest}"),
        _touch(responded.with_suffix(f"{responded.suffix}.imported")),
    ]


def test_an_agent_without_indexes_only_gets_the_scheduled_task_report(tmp_path: Path) -> None:
    """With nothing derived from the agent on disk, only the scheduled task note remains, since it needs Matrix."""
    steps, reports = _plan(_runtime_paths(tmp_path), _config())

    assert steps == []
    assert len(reports) == 1
    assert _SCHEDULED in reports[0]
    assert f"@{OLD}" in reports[0]
    assert f"@{NEW}" in reports[0]


def test_file_memory_search_indexes_of_the_agent_are_dropped_into_the_backup(tmp_path: Path) -> None:
    """The agent's file-memory search indexes, for its current root and an earlier one, are dropped recoverably."""
    runtime_paths = _runtime_paths(tmp_path)
    config = _config(memory_backend="file", memory_search={"mode": "semantic"})
    own = _file_memory_index(OLD, config, runtime_paths)
    earlier = _file_memory_index(OLD, config, runtime_paths, root=tmp_path / "earlier_memory_root")
    prefix = _file_memory_index(PREFIX, config, runtime_paths)

    steps, _ = _plan(runtime_paths, config)

    assert all(isinstance(step, DropPath) for step in steps)
    assert sorted(step.path.resolve() for step in steps if isinstance(step, DropPath)) == sorted([own, earlier])
    _apply(runtime_paths, steps)
    assert not own.exists()
    assert not earlier.exists()
    assert (prefix / "data").read_bytes() == b"data"
    backups = runtime_paths.storage_root / BACKUPS_DIRECTORY_NAME
    assert {path.name for path in backups.glob("*/*/*") if path.is_dir()} == {own.name, earlier.name}


def test_the_private_knowledge_index_of_a_formerly_private_agent_is_dropped(tmp_path: Path) -> None:
    """An index MindRoom built for the agent's private knowledge while it was private is dropped."""
    runtime_paths = _runtime_paths(tmp_path)
    private = AgentPrivateConfig(per="user", knowledge={"path": "docs"})
    private_config = Config(
        agents={
            OLD: AgentConfig(display_name="Code", private=private),
            PREFIX: AgentConfig(display_name="Code Team", private=private),
            HELPER: AgentConfig(display_name="Helper"),
        },
    )
    own = _private_knowledge_index(OLD, private_config, runtime_paths)
    prefix = _private_knowledge_index(PREFIX, private_config, runtime_paths)

    # The rename supports only agents that are not private, so the agent is no longer private now.
    steps, _ = _plan(runtime_paths, _config())

    assert [step.path.resolve() for step in steps if isinstance(step, DropPath)] == [own]
    _apply(runtime_paths, steps)
    assert not own.exists()
    assert (prefix / "data").read_bytes() == b"data"


def test_the_knowledge_graph_of_the_old_state_root_is_reported(tmp_path: Path) -> None:
    """A started graph of the agent is named after its old state root, so it is reported as left behind."""
    runtime_paths = _runtime_paths(tmp_path)
    config = _config(knowledge_graph=True)
    graph = _start_graph(OLD, config, runtime_paths)

    _, reports = _plan(runtime_paths, config)

    report = _report(reports, _GRAPH)
    assert report is not None
    assert graph in report
    assert "knowledge_graph.generation" in report
    assert "knowledge_graph.ingest.backfill_days (7)" in report


def test_no_graph_report_without_a_started_graph_of_the_agent(tmp_path: Path) -> None:
    """A graph-enabled agent whose graph never started has nothing left behind; another agent's graph is not its own."""
    runtime_paths = _runtime_paths(tmp_path)
    config = _config(knowledge_graph=True)
    _start_graph(PREFIX, config, runtime_paths)

    _, reports = _plan(runtime_paths, config)

    assert _report(reports, _GRAPH) is None


def test_git_directories_of_checkouts_in_the_moved_state_root_are_reported(tmp_path: Path) -> None:
    """A Git-backed base checked out in the agent's state root keeps its Git directory under the old path's hash."""
    runtime_paths = _runtime_paths(tmp_path)
    config = _git_config((OLD, PREFIX))
    old_git = _knowledge_git(f"{OLD}_docs", config, runtime_paths)
    prefix_git = _knowledge_git(f"{PREFIX}_docs", config, runtime_paths)
    moved_checkout = agent_state_root_path(runtime_paths.storage_root, NEW) / "workspace" / "docs"
    new_git = knowledge_git_dir(runtime_paths.storage_root, moved_checkout)

    _, reports = _plan(runtime_paths, config)

    report = _report(reports, _GIT)
    assert report is not None
    assert str(old_git) in report
    assert str(new_git) in report
    assert f"{OLD}_docs" in report
    assert str(prefix_git) not in report


def test_no_git_report_when_the_checkout_has_no_git_directory_yet(tmp_path: Path) -> None:
    """A base that never synced has no Git directory to leave behind."""
    runtime_paths = _runtime_paths(tmp_path)

    _, reports = _plan(runtime_paths, _git_config((OLD,)))

    assert _report(reports, _GIT) is None


def test_mem0_history_messages_of_the_agent_scope_are_reported(tmp_path: Path) -> None:
    """mem0 keeps recent messages per scope outside the storage root; the agent's scope id does not change there."""
    runtime_paths = _runtime_paths(tmp_path)
    database = tmp_path / "mem0" / "history.db"
    _mem0_messages(database, OLD)

    _, reports = _plan(runtime_paths, _config())

    report = _report(reports, _MEM0)
    assert report is not None
    assert str(database) in report
    assert f"user_id={agent_scope_user_id(OLD)}" in report


def test_mem0_history_defaults_to_the_home_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without MEM0_DIR, mem0 keeps its history database in ~/.mem0, since MindRoom's memory config sets no path."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    runtime_paths = _runtime_paths(tmp_path, mem0_dir=False)
    database = tmp_path / "home" / ".mem0" / "history.db"
    _mem0_messages(database, OLD)

    _, reports = _plan(runtime_paths, _config())

    report = _report(reports, _MEM0)
    assert report is not None
    assert str(database) in report


def test_no_mem0_report_without_messages_of_the_agent_scope(tmp_path: Path) -> None:
    """Messages of the agent whose name extends the renamed one's are not the agent's."""
    runtime_paths = _runtime_paths(tmp_path)
    _mem0_messages(tmp_path / "mem0" / "history.db", PREFIX)

    _, reports = _plan(runtime_paths, _config())

    assert _report(reports, _MEM0) is None


def test_plugin_state_paths_naming_the_agent_are_reported(tmp_path: Path) -> None:
    """Paths below a plugin's state root named exactly after the agent, or its JSON file, are listed."""
    runtime_paths = _runtime_paths(tmp_path)
    notes = get_plugin_state_root("notes", runtime_paths=runtime_paths)
    expected = {
        _touch(notes / "agents" / OLD / "state.json").parent,
        _touch(notes / f"{OLD}.json"),
        _mkdir(notes / "rooms" / "lobby" / OLD),
    }
    _touch(notes / f"{PREFIX}.json")
    _touch(notes / PREFIX / f"{OLD}-notes.txt")
    # A plugin that happens to share the agent's name is not state of the agent.
    _touch(get_plugin_state_root(OLD, runtime_paths=runtime_paths) / "data.json")

    _, reports = _plan(runtime_paths, _config())

    assert _listed(_report(reports, _PLUGINS)) == {str(path) for path in expected}


def test_no_plugin_report_without_paths_naming_the_agent(tmp_path: Path) -> None:
    """A state file of the agent whose name extends the renamed one's is not the agent's."""
    runtime_paths = _runtime_paths(tmp_path)
    _touch(get_plugin_state_root("notes", runtime_paths=runtime_paths) / f"{PREFIX}.json")

    _, reports = _plan(runtime_paths, _config())

    assert _report(reports, _PLUGINS) is None


def test_config_lines_the_config_edit_leaves_are_reported(tmp_path: Path) -> None:
    """Other agents' instructions mentioning the agent stay; the lines the config edit renames are not listed."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "agents:\n"
        f"  {OLD}:\n"
        "    display_name: Code\n"
        f"  {PREFIX}:\n"
        "    display_name: Code Team\n"
        f"    role: Reviews what {PREFIX} ships.\n"
        f"  {HELPER}:\n"
        "    display_name: Helper\n"
        "    role: !include_text helper.md\n"
        "    instructions:\n"
        f"      - Ask @{OLD} for fixes.\n"
        f"    delegate_to: [{OLD}, {PREFIX}]\n"
        "teams:\n"
        "  crew:\n"
        "    display_name: Crew\n"
        "    role: Ships releases\n"
        f"    agents: [{OLD}, {HELPER}]\n"
        "router:\n"
        "  model: default\n",
        encoding="utf-8",
    )
    prompt = _touch(tmp_path / "helper.md")
    prompt.write_text(f"Pair with {PREFIX}.\nWhen stuck, hand off to {OLD}.\n", encoding="utf-8")
    runtime_paths = _runtime_paths(tmp_path)

    _, reports = _plan(runtime_paths, load_config(runtime_paths))

    report = _report(reports, _CONFIG_LINES)
    assert report is not None
    assert report.endswith(f"{config_path} (line 11); {prompt} (line 2).")


def test_no_config_line_report_when_the_config_edit_renames_every_mention(tmp_path: Path) -> None:
    """Keys and delegate lists the config edit renames leave no line to report."""
    (tmp_path / "config.yaml").write_text(
        "agents:\n"
        f"  {OLD}:\n"
        "    display_name: Code\n"
        f"  {PREFIX}:\n"
        "    display_name: Code Team\n"
        f"    delegate_to: [{OLD}]\n"
        "router:\n"
        "  model: default\n",
        encoding="utf-8",
    )
    runtime_paths = _runtime_paths(tmp_path)

    _, reports = _plan(runtime_paths, load_config(runtime_paths))

    assert _report(reports, _CONFIG_LINES) is None


def test_legacy_files_of_the_agent_are_reported(tmp_path: Path) -> None:
    """Files older versions kept for the agent are listed; those of the agent whose name extends it are not."""
    runtime_paths = _runtime_paths(tmp_path)
    expected = _legacy_files(runtime_paths, OLD)
    _legacy_files(runtime_paths, PREFIX)

    _, reports = _plan(runtime_paths, _config())

    assert _listed(_report(reports, _LEGACY)) == {str(path) for path in expected}


def test_planning_writes_nothing_and_leaves_the_prefix_agent_alone(tmp_path: Path) -> None:
    """With every store present for both agents, planning changes no file and nothing of `code_team` is planned."""
    runtime_paths = _runtime_paths(tmp_path)
    config = _git_config((OLD, PREFIX)).model_copy(
        update={
            "agents": {
                name: AgentConfig(
                    display_name=name.title(),
                    memory_backend="file",
                    memory_search={"mode": "semantic"},
                    knowledge_graph=True,
                )
                for name in (OLD, PREFIX, HELPER)
            },
        },
    )
    for agent in (OLD, PREFIX):
        _file_memory_index(agent, config, runtime_paths)
        _start_graph(agent, config, runtime_paths)
        _knowledge_git(f"{agent}_docs", config, runtime_paths)
        _mem0_messages(tmp_path / "mem0" / "history.db", agent)
        _touch(get_plugin_state_root("notes", runtime_paths=runtime_paths) / f"{agent}.json")
        _legacy_files(runtime_paths, agent)
    before = tree_snapshot(tmp_path)

    steps, reports = _plan(runtime_paths, config)

    assert tree_snapshot(tmp_path) == before
    assert [PREFIX not in step.path.name for step in steps if isinstance(step, DropPath)] == [True]
    assert {_report(reports, marker) is not None for marker in (_GRAPH, _GIT, _MEM0, _PLUGINS, _LEGACY)} == {True}
    assert [report for report in reports if PREFIX in report] == []
