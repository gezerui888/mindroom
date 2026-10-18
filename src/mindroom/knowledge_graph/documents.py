"""Files of knowledge bases flagged ``knowledge_graph: true``, rendered as ``document`` episodes.

Code reads each file, so a document's evidence is its own text, never a model's paraphrase of it. Files are listed with
the knowledge-base listing rules, which never follow a link a worker could plant, and named by their base and relative
path; a content hash, not the modification time a checkout rewrites, decides whether a file changed.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from mindroom.knowledge.file_listing import list_git_tracked_knowledge_files, list_knowledge_files, open_knowledge_file
from mindroom.knowledge.indexing_config import knowledge_git_dir
from mindroom.knowledge_graph.store import EpisodeInput
from mindroom.logging_config import get_logger
from mindroom.path_confinement import MAX_READ_BYTES
from mindroom.redaction import redact_sensitive_text
from mindroom.runtime_resolution import resolve_knowledge_binding

if TYPE_CHECKING:
    from pathlib import Path

    from mindroom.config.knowledge_graph import KnowledgeGraphIngestConfig
    from mindroom.config.main import Config
    from mindroom.constants import RuntimePaths
    from mindroom.knowledge_graph.scope import GraphScope

logger = get_logger(__name__)

_READ_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class DocumentFile:
    """One file of a flagged knowledge base, keyed ``<base_id>/<relative posix path>``, with its content hash."""

    key: str
    path: Path
    sha256: str
    modified_at: datetime


def document_files(config: Config, scope: GraphScope, runtime_paths: RuntimePaths) -> list[DocumentFile]:
    """Return the files of the flagged knowledge bases the scope's agent uses, hashed as they are now."""
    files: list[DocumentFile] = []
    for base_id in config.resolve_entity(scope.agent_name).knowledge_base_ids:
        base_config = config.get_knowledge_base_config(base_id)
        if not base_config.knowledge_graph:
            continue
        root = resolve_knowledge_binding(base_id, config, runtime_paths, scope.execution_identity).knowledge_path
        if base_config.git is None:
            paths = list_knowledge_files(config, base_id, root)
        else:
            git_dir = knowledge_git_dir(runtime_paths.storage_root, root)
            paths = list_git_tracked_knowledge_files(config, base_id, root, git_dir)
        for path in paths:
            try:
                sha256, modified_at = _hash(path)
            except (OSError, ValueError):
                # A file removed or replaced since it was listed is looked at again on the next pass.
                continue
            key = f"{base_id}/{path.relative_to(root).as_posix()}"
            files.append(DocumentFile(key, path, sha256, datetime.fromtimestamp(modified_at, UTC)))
    return files


def read_document(file: DocumentFile) -> str | None:
    """Return a listed file's text, empty when it is not UTF-8 text, or None when its content changed since listing."""
    chunks: list[bytes] = []
    try:
        sha256, _ = _hash(file.path, chunks)
    except (OSError, ValueError):
        return None
    if sha256 != file.sha256:
        return None
    try:
        return b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("Knowledge graph skipped a knowledge-base file that is not UTF-8 text", document=file.key)
        return ""


def _hash(path: Path, chunks: list[bytes] | None = None) -> tuple[str, float]:
    """Hash a listed file and return its mtime, reading through one descriptor that follows no link.

    The content is collected into ``chunks`` when given.
    """
    digest = hashlib.sha256()
    with open_knowledge_file(path) as descriptor:
        status = os.fstat(descriptor)
        if status.st_size > MAX_READ_BYTES:
            msg = f"Knowledge file exceeds its size limit: {path}"
            raise ValueError(msg)
        while chunk := os.read(descriptor, _READ_CHUNK_BYTES):
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
    return digest.hexdigest(), status.st_mtime


def render_document(file: DocumentFile, text: str, settings: KnowledgeGraphIngestConfig) -> list[EpisodeInput]:
    """Render a file's text as redacted episodes of at most ``max_episode_chars``, split at line boundaries.

    Each piece is dated by the file's modification time, so extraction resolves relative dates against when the file
    was last written, and named by the content hash, so a new version never reuses an old version's name.
    """
    cap = settings.max_episode_chars
    return [
        EpisodeInput(
            kind="document",
            name=f"document {file.key} {file.sha256[:12]} {index}",
            # Redaction refuses input above its size limit, so it runs per piece; a placeholder longer than the value
            # it replaces is clipped to the cap.
            body=redact_sensitive_text(piece, max_length=cap),
            source_description=f"document: {file.key}",
            reference_time=file.modified_at,
        )
        for index, piece in enumerate(_pieces(text, cap))
    ]


def _pieces(text: str, cap: int) -> list[str]:
    """Split text at line boundaries into pieces of at most ``cap`` characters, cutting only a longer line."""
    pieces: list[str] = []
    lines: list[str] = []
    size = 0
    for line in text.splitlines():
        for start in range(0, max(len(line), 1), cap):
            part = line[start : start + cap]
            if lines and size + len(part) + 1 > cap:
                pieces.append("\n".join(lines))
                lines, size = [], 0
            lines.append(part)
            size += len(part) + 1
    if lines:
        pieces.append("\n".join(lines))
    return [piece for piece in pieces if piece.strip()]
