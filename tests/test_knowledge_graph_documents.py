"""Files of flagged knowledge bases become dated, redacted document episodes, read by code rather than a model."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from mindroom.config.agent import AgentConfig
from mindroom.config.knowledge import KnowledgeBaseConfig
from mindroom.config.knowledge_graph import KnowledgeGraphIngestConfig
from mindroom.config.main import Config
from mindroom.config.models import ModelConfig
from mindroom.constants import resolve_runtime_paths
from mindroom.knowledge_graph.documents import DocumentFile, document_files, render_document
from mindroom.knowledge_graph.scope import graph_scope

SETTINGS = KnowledgeGraphIngestConfig(max_episode_chars=1000, max_tool_result_chars=400)
MODIFIED = datetime(2026, 3, 4, 12, 30, tzinfo=UTC)


def _file() -> DocumentFile:
    return DocumentFile(key="facts/people.md", path=Path("/kb/facts/people.md"), sha256="ab" * 32, modified_at=MODIFIED)


def test_only_flagged_bases_of_the_agent_feed_its_graph(tmp_path: Path) -> None:
    """A base feeds the graph when it is flagged and the agent uses it; the listing rules of the base still apply."""
    for base in ("facts", "notes", "other"):
        (tmp_path / base).mkdir()
        (tmp_path / base / f"{base}.md").write_text(f"About {base}.\n", encoding="utf-8")
    (tmp_path / "facts" / "topics").mkdir()
    (tmp_path / "facts" / "topics" / "atlas.yaml").write_text("owner: Alice\n", encoding="utf-8")
    (tmp_path / "facts" / "photo.png").write_bytes(b"\x89PNG")
    (tmp_path / "outside.md").write_text("Not part of any base.\n", encoding="utf-8")
    (tmp_path / "facts" / "linked.md").symlink_to(tmp_path / "outside.md")
    config = Config(
        agents={"mind": AgentConfig(display_name="Mind", knowledge_graph=True, knowledge_bases=["facts", "notes"])},
        models={"luna": ModelConfig(provider="openai", id="gpt-6-luna")},
        knowledge_graph={"model": "luna"},
        knowledge_bases={
            "facts": KnowledgeBaseConfig(path=str(tmp_path / "facts"), knowledge_graph=True),
            "notes": KnowledgeBaseConfig(path=str(tmp_path / "notes")),
            "other": KnowledgeBaseConfig(path=str(tmp_path / "other"), knowledge_graph=True),
        },
    )
    paths = resolve_runtime_paths(config_path=tmp_path / "config.yaml", storage_path=tmp_path / "storage")

    files = document_files(config, graph_scope("mind", config, paths, None), paths)

    assert [file.key for file in files] == ["facts/facts.md", "facts/topics/atlas.yaml"]
    atlas = files[1]
    assert atlas.sha256 == hashlib.sha256(b"owner: Alice\n").hexdigest()
    assert atlas.modified_at == datetime.fromtimestamp(atlas.path.stat().st_mtime, UTC)


def test_render_document_splits_at_lines_and_dates_by_mtime() -> None:
    """A long file becomes pieces cut only between lines, each dated by the file's modification time."""
    lines = [f"Line {index:02d} of the notes about Project Atlas ".ljust(99, "o") for index in range(25)]

    episodes = render_document(_file(), "\n".join(lines) + "\n", SETTINGS)

    assert len(episodes) == 3
    assert {episode.kind for episode in episodes} == {"document"}
    assert all(len(episode.body) <= SETTINGS.max_episode_chars for episode in episodes)
    assert [line for episode in episodes for line in episode.body.splitlines()] == lines
    assert [episode.name for episode in episodes] == [f"document facts/people.md {'ab' * 6} {i}" for i in range(3)]
    assert {episode.source_description for episode in episodes} == {"document: facts/people.md"}
    assert {episode.reference_time for episode in episodes} == {MODIFIED}


def test_secrets_are_redacted_from_documents() -> None:
    """Credential-like values in a file never reach extraction."""
    credential = "sk-proj-4f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c"

    episodes = render_document(_file(), f"Alice owns Atlas.\nOPENAI_API_KEY={credential}\n", SETTINGS)

    assert len(episodes) == 1
    assert "Alice owns Atlas." in episodes[0].body
    assert credential not in episodes[0].body
