"""Files of knowledge bases flagged ``knowledge_graph: true``, rendered as ``document`` episodes.

Code reads each file, so a document's evidence is its own text, never a model's paraphrase of it. Files are listed with
the knowledge-base listing rules, which never follow a link a worker could plant, and named by their base and relative
path; a content hash, not the modification time a checkout rewrites, decides whether a file changed.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import islice
from typing import TYPE_CHECKING

from mindroom.knowledge.file_listing import list_git_tracked_knowledge_files, list_knowledge_files, open_knowledge_file
from mindroom.knowledge.indexing_config import knowledge_git_dir
from mindroom.knowledge_graph.store import EpisodeInput
from mindroom.logging_config import get_logger
from mindroom.path_confinement import MAX_READ_BYTES
from mindroom.redaction import redact_sensitive_text
from mindroom.runtime_resolution import resolve_knowledge_binding

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    from mindroom.config.knowledge_graph import KnowledgeGraphIngestConfig
    from mindroom.config.main import Config
    from mindroom.constants import RuntimePaths
    from mindroom.knowledge_graph.scope import GraphScope

logger = get_logger(__name__)

_READ_CHUNK_BYTES = 1 << 20
_EPOCH = datetime.fromtimestamp(0, UTC)

# A file's size, modification time in nanoseconds, and inode; while all three are unchanged its content is assumed to be.
type _FileStat = tuple[int, int, int]


@dataclass(frozen=True)
class DocumentFile:
    """One file of a flagged knowledge base, keyed ``<base_id>/<relative posix path>``, with its content hash."""

    key: str
    path: Path
    sha256: str
    modified_at: datetime
    stat: _FileStat


def document_files(
    config: Config,
    scope: GraphScope,
    runtime_paths: RuntimePaths,
    known: Mapping[str, tuple[_FileStat, str]],
) -> tuple[list[DocumentFile], dict[str, Exception]]:
    """Return the files of the flagged knowledge bases the scope's agent uses, and the bases that could not be listed.

    ``known`` maps a file's key to the status and hash it last had; a file whose status is unchanged is not read again.
    """
    files: list[DocumentFile] = []
    errors: dict[str, Exception] = {}
    for base_id in config.resolve_entity(scope.agent_name).knowledge_base_ids:
        base_config = config.get_knowledge_base_config(base_id)
        if not base_config.knowledge_graph:
            continue
        # One base whose path or Git listing fails must not keep the agent's other bases out of its graph.
        try:
            root = resolve_knowledge_binding(base_id, config, runtime_paths, scope.execution_identity).knowledge_path
            if base_config.git is None:
                paths = list_knowledge_files(config, base_id, root)
            else:
                git_dir = knowledge_git_dir(runtime_paths.storage_root, root)
                paths = list_git_tracked_knowledge_files(config, base_id, root, git_dir)
        except (OSError, RuntimeError, ValueError) as exc:
            errors[base_id] = exc
            continue
        for path in paths:
            key = f"{base_id}/{path.relative_to(root).as_posix()}"
            try:
                sha256, status = _signature(path, known.get(key))
            except (OSError, ValueError):
                # A file removed, replaced, or grown past the read cap since it was listed is looked at again next pass.
                continue
            modified_at = _EPOCH + timedelta(microseconds=status[1] // 1000)
            files.append(DocumentFile(key, path, sha256, modified_at, status))
    return files, errors


def read_document(file: DocumentFile) -> str | None:
    """Return a listed file's text, empty when it is not UTF-8 text, or None when its content changed since listing."""
    chunks: list[bytes] = []
    try:
        sha256, _ = _signature(file.path, None, chunks)
    except (OSError, ValueError):
        return None
    if sha256 != file.sha256:
        return None
    try:
        return b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("Knowledge graph skipped a knowledge-base file that is not UTF-8 text", document=file.key)
        return ""


def _signature(
    path: Path,
    known: tuple[_FileStat, str] | None,
    chunks: list[bytes] | None = None,
) -> tuple[str, _FileStat]:
    """Return a listed file's hash and status through one descriptor that follows no link.

    A ``known`` status that still matches reuses its hash without reading. Otherwise the file is read, refused like the
    confined reader refuses it once it holds more than the read cap, and collected into ``chunks`` when given.
    """
    with open_knowledge_file(path) as descriptor:
        status = os.fstat(descriptor)
        stat: _FileStat = (status.st_size, status.st_mtime_ns, status.st_ino)
        if known is not None and known[0] == stat:
            return known[1], stat
        if status.st_size > MAX_READ_BYTES:
            msg = f"Knowledge file exceeds its size limit: {path}"
            raise ValueError(msg)
        digest = hashlib.sha256()
        remaining = MAX_READ_BYTES + 1
        while remaining > 0 and (chunk := os.read(descriptor, min(remaining, _READ_CHUNK_BYTES))):
            remaining -= len(chunk)
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
    if remaining <= 0:
        msg = f"Knowledge file exceeds its size limit: {path}"
        raise ValueError(msg)
    return digest.hexdigest(), stat


def render_document(
    file: DocumentFile,
    text: str,
    settings: KnowledgeGraphIngestConfig,
    *,
    limit: int | None = None,
) -> list[EpisodeInput]:
    """Render a file's text as redacted episodes of at most ``max_episode_chars``, split at line boundaries.

    Each piece is dated by the file's modification time, so extraction resolves relative dates against when the file
    was last written, and named by the content hash, so a new version never reuses an old version's name. With
    ``limit``, only the first ``limit`` pieces are rendered; where a piece starts depends on every line before it.
    """
    return [
        EpisodeInput(
            kind="document",
            name=f"document {file.key} {file.sha256[:12]} {index}",
            body=piece,
            source_description=f"document: {file.key}",
            reference_time=file.modified_at,
        )
        for index, piece in enumerate(islice(_pieces(text, settings.max_episode_chars), limit))
    ]


def _pieces(text: str, cap: int) -> Iterator[str]:
    """Yield redacted text in pieces of at most ``cap`` characters, split at line boundaries, cutting only longer lines.

    Each line is redacted before anything is cut, so a credential split across two pieces is still caught; a line above
    the redaction size limit becomes the redaction-failed placeholder.
    """
    lines: list[str] = []
    size = 0
    for raw_line in text.splitlines():
        line = redact_sensitive_text(raw_line)
        for start in range(0, max(len(line), 1), cap):
            part = line[start : start + cap]
            if lines and size + len(part) + 1 > cap:
                if (piece := "\n".join(lines)).strip():
                    yield piece
                lines, size = [], 0
            lines.append(part)
            size += len(part) + 1
    if (piece := "\n".join(lines)).strip():
        yield piece
