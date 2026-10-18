"""Files of flagged knowledge bases become dated, redacted document episodes, read by code rather than a model."""

from __future__ import annotations

import hashlib
import os
import subprocess
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mindroom.config.agent import AgentConfig
from mindroom.config.knowledge import KnowledgeBaseConfig, KnowledgeGitConfig
from mindroom.config.knowledge_graph import KnowledgeGraphIngestConfig
from mindroom.config.main import Config
from mindroom.config.models import ModelConfig
from mindroom.constants import resolve_runtime_paths
from mindroom.knowledge.indexing_config import knowledge_git_dir
from mindroom.knowledge_graph import documents
from mindroom.knowledge_graph.documents import DocumentFile, document_files, render_document
from mindroom.knowledge_graph.scope import graph_scope

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mindroom.constants import RuntimePaths

SETTINGS = KnowledgeGraphIngestConfig(max_episode_chars=1000, max_tool_result_chars=400)
MODIFIED = datetime(2026, 3, 4, 12, 30, tzinfo=UTC)


def _file() -> DocumentFile:
    return DocumentFile(
        key="facts/people.md",
        path=Path("/kb/facts/people.md"),
        sha256="ab" * 32,
        modified_at=MODIFIED,
        stat=(1, 1, 1),
    )


def _facts_config(tmp_path: Path, **base: object) -> tuple[Config, RuntimePaths]:
    """Return a config whose agent uses one flagged base `facts` at `tmp_path / "facts"`, and its runtime paths."""
    config = Config(
        agents={"mind": AgentConfig(display_name="Mind", knowledge_graph=True, knowledge_bases=["facts"])},
        models={"luna": ModelConfig(provider="openai", id="gpt-6-luna")},
        knowledge_graph={"model": "luna"},
        knowledge_bases={"facts": KnowledgeBaseConfig(path=str(tmp_path / "facts"), knowledge_graph=True, **base)},
    )
    return config, resolve_runtime_paths(config_path=tmp_path / "config.yaml", storage_path=tmp_path / "storage")


def _list(config: Config, paths: RuntimePaths, known: dict | None = None) -> list[DocumentFile]:
    files, errors = document_files(config, graph_scope("mind", config, paths, None), paths, known or {})
    assert errors == {}
    return files


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

    files = _list(config, paths)

    assert [file.key for file in files] == ["facts/facts.md", "facts/topics/atlas.yaml"]
    atlas = files[1]
    assert atlas.sha256 == hashlib.sha256(b"owner: Alice\n").hexdigest()
    assert abs(atlas.modified_at.timestamp() - atlas.path.stat().st_mtime) < 1e-5


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


def test_a_credential_crossing_a_cut_is_still_redacted() -> None:
    """A line longer than a piece is redacted before it is cut, so a key split across two pieces never leaks."""
    credential = "sk-proj-4f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c"
    line = "word " * 196 + f"OPENAI_API_KEY={credential} and more words after it"

    episodes = render_document(_file(), line + "\n", SETTINGS)

    assert len(episodes) == 2
    assert all(len(episode.body) <= SETTINGS.max_episode_chars for episode in episodes)
    joined = "".join(episode.body for episode in episodes)
    assert "4f9a8b7c" not in joined
    assert "OPENAI_API_KEY=" in joined


def test_a_file_whose_size_time_and_inode_are_unchanged_is_not_read_again(tmp_path: Path) -> None:
    """A known status reuses the known hash; any change to it, such as a touch, hashes the file again."""
    config, paths = _facts_config(tmp_path)
    (tmp_path / "facts").mkdir()
    note = tmp_path / "facts" / "note.md"
    note.write_text("Quinn leads Project Larkspur.\n", encoding="utf-8")
    [first] = _list(config, paths)
    mtime_ns = note.stat().st_mtime_ns
    # Same size, time, and inode, different bytes: only a read would notice.
    note.write_text("Quinn quits Project Larkspur.\n", encoding="utf-8")
    os.utime(note, ns=(mtime_ns, mtime_ns))
    assert note.stat().st_size == first.stat[0]

    [cached] = _list(config, paths, {first.key: (first.stat, first.sha256)})
    os.utime(note, ns=(mtime_ns + 1_000_000_000, mtime_ns + 1_000_000_000))
    [rehashed] = _list(config, paths, {first.key: (first.stat, first.sha256)})

    assert cached.sha256 == first.sha256
    assert rehashed.sha256 == hashlib.sha256(note.read_bytes()).hexdigest() != first.sha256


def test_a_file_that_grows_past_the_read_cap_while_read_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reading stops one byte past the cap, as the confined reader does, even when the status said the file was small."""
    config, paths = _facts_config(tmp_path)
    (tmp_path / "facts").mkdir()
    (tmp_path / "facts" / "log.md").write_text("small\n", encoding="utf-8")

    @contextmanager
    def growing_file(path: Path) -> Iterator[int]:
        # A pipe reports a size of zero and then yields more bytes than the cap, like a file that grew after its status.
        del path
        read_end, write_end = os.pipe()
        os.write(write_end, b"x" * 20)
        os.close(write_end)
        try:
            yield read_end
        finally:
            os.close(read_end)

    monkeypatch.setattr(documents, "MAX_READ_BYTES", 10)
    monkeypatch.setattr(documents, "open_knowledge_file", growing_file)

    assert _list(config, paths) == []


def test_a_git_backed_base_feeds_only_its_tracked_files(tmp_path: Path) -> None:
    """A Git-backed base contributes the files Git tracks, listed through the Git directory MindRoom owns."""
    config, paths = _facts_config(tmp_path, git=KnowledgeGitConfig(repo_url="https://example.com/org/facts.git"))
    root = tmp_path / "facts"
    root.mkdir()
    (root / "tracked.md").write_text("Rowan founded Harbor Labs.\n", encoding="utf-8")
    git_dir = knowledge_git_dir(paths.storage_root, root.resolve())
    git_dir.parent.mkdir(parents=True)
    env = {**os.environ, "GIT_DIR": str(git_dir), "GIT_WORK_TREE": str(root)}
    for args in (
        ("init", "-b", "main"),
        ("add", "tracked.md"),
        (
            "-c",
            "user.email=tests@example.com",
            "-c",
            "user.name=Tests",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-m",
            "x",
        ),
    ):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, env=env)
    (root / "draft.md").write_text("Not committed.\n", encoding="utf-8")

    assert [file.key for file in _list(config, paths)] == ["facts/tracked.md"]


@pytest.mark.parametrize(
    "secret_lines",
    [
        pytest.param(["db:", "  password:", "    Zq8vT2mLx9Rk4Wp7"], id="secret-key-with-its-value-on-the-next-line"),
        pytest.param(["Authorization: Bearer", "Zq8vT2mLx9Rk4Wp7abc"], id="bearer-token-on-the-next-line"),
    ],
)
@pytest.mark.parametrize("across_pieces", [False, True], ids=["alone", "across-pieces"])
def test_a_credential_continued_on_the_next_line_is_still_redacted(
    secret_lines: list[str],
    *,
    across_pieces: bool,
) -> None:
    """Redaction sees the lines before a credential, also when the line that names it ends the previous piece."""
    *naming, value = secret_lines
    filler: list[str] = []
    if across_pieces:
        # One filler line sized so the naming lines end the first piece and the value starts the next one.
        filler = ["o" * (SETTINGS.max_episode_chars - 10 - sum(len(line) + 1 for line in naming) - 1)]
    text = "\n".join([*filler, *naming, value, "Closing words."]) + "\n"

    episodes = render_document(_file(), text, SETTINGS)

    assert len(episodes) == (2 if across_pieces else 1)
    assert "Zq8vT2mLx9Rk4Wp7" not in "".join(episode.body for episode in episodes)
