"""Files of knowledge bases flagged ``knowledge_graph: true``, rendered as ``document`` episodes.

Code reads each file, so a document's evidence is its own text, never a model's paraphrase of it. Files are listed with
the knowledge-base listing rules, which never follow a link a worker could plant, and named by their base and relative
path; a content hash, not the modification time a checkout rewrites, decides whether a file changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import islice
from typing import TYPE_CHECKING

from mindroom.knowledge.file_listing import hash_knowledge_file, list_git_tracked_knowledge_files, list_knowledge_files
from mindroom.knowledge.indexing_config import knowledge_git_dir
from mindroom.knowledge_graph.store import EpisodeInput
from mindroom.logging_config import get_logger
from mindroom.redaction import REDACTED, REDACTION_FAILED, redact_private_keys, redact_sensitive_text
from mindroom.runtime_resolution import resolve_knowledge_binding

if TYPE_CHECKING:
    import os
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    from mindroom.config.main import Config
    from mindroom.constants import RuntimePaths
    from mindroom.knowledge_graph.scope import GraphScope

logger = get_logger(__name__)

# Unreadable files already logged, by key and error, so a file that stays unreadable is logged once, not every pass.
_UNREADABLE_REPORTED: set[tuple[str, str]] = set()
# Text before a block that redaction sees with it, so a credential named at the end of the previous block, such as a
# secret key whose value is on the next line or "Bearer" before a token, is still caught.
_REDACTION_CONTEXT_CHARS = 4096
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
) -> tuple[list[DocumentFile], dict[str, Exception]] | None:
    """Return the files of the flagged knowledge bases the scope's agent uses, and the bases that could not be listed.

    Return None when the agent uses no flagged base. ``known`` maps a file's key to the status and hash it last had;
    a file whose status is unchanged is not read again.
    """
    base_ids = config.resolve_entity(scope.agent_name).knowledge_base_ids
    flagged = [base_id for base_id in base_ids if config.get_knowledge_base_config(base_id).knowledge_graph]
    if not flagged:
        return None
    files: list[DocumentFile] = []
    errors: dict[str, Exception] = {}
    for base_id in flagged:
        base_config = config.get_knowledge_base_config(base_id)
        # One base whose path or Git listing fails must not keep the agent's other bases out of its graph.
        try:
            root = resolve_knowledge_binding(
                base_id,
                config,
                runtime_paths,
                execution_identity=scope.execution_identity,
            ).knowledge_path
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
            except FileNotFoundError:
                # A file removed since it was listed is left alone, like a deleted one.
                continue
            except OSError as exc:
                # Unreadable, or swapped for a link since it was listed; reported once per file and error, retried each pass.
                if (report := (key, f"{type(exc).__name__}:{exc.errno}")) not in _UNREADABLE_REPORTED:
                    _UNREADABLE_REPORTED.add(report)
                    logger.warning(
                        "Knowledge graph could not read a knowledge-base file",
                        document=key,
                        error=report[1],
                    )
                continue
            except ValueError:
                # Grown past the read cap since it was listed, which listing reports itself.
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
    """Return a listed file's hash and status; a ``known`` status that still matches reuses its hash without reading."""

    def known_digest(status: os.stat_result) -> str | None:
        return known[1] if known is not None and known[0] == _stat(status) else None

    digest, status = hash_knowledge_file(
        path,
        known_digest=known_digest,
        sink=chunks.append if chunks is not None else None,
    )
    return digest, _stat(status)


def _stat(status: os.stat_result) -> _FileStat:
    return status.st_size, status.st_mtime_ns, status.st_ino


def render_document(
    file: DocumentFile,
    text: str,
    piece_chars: int,
    *,
    limit: int | None = None,
) -> list[EpisodeInput]:
    """Render a file's text as redacted episodes of at most ``piece_chars`` characters, split at line boundaries.

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
        for index, piece in enumerate(islice(_pieces(text, piece_chars), limit))
    ]


def _pieces(text: str, cap: int) -> Iterator[str]:
    """Yield redacted text in pieces of at most ``cap`` characters, split at line boundaries, cutting only longer lines.

    The text is redacted before anything is cut, so a credential split across two pieces is still caught.
    """
    lines: list[str] = []
    size = 0
    for line in _redacted_lines(text, cap):
        for start in range(0, max(len(line), 1), cap):
            part = line[start : start + cap]
            if lines and size + len(part) + 1 > cap:
                if _has_text(piece := "\n".join(lines)):
                    yield piece
                lines, size = [], 0
            lines.append(part)
            size += len(part) + 1
    if _has_text(piece := "\n".join(lines)):
        yield piece


def _has_text(piece: str) -> bool:
    """Return whether a piece holds anything besides whitespace and redaction placeholders, which is worth extracting."""
    return bool(piece.replace(REDACTED, "").replace(REDACTION_FAILED, "").strip())


def _redacted_lines(text: str, block_chars: int) -> Iterator[str]:
    """Yield the text's lines, redacted in blocks of whole lines each seen together with the text before it.

    Redaction keeps every line break, so the block's lines follow the context's. A block whose redaction fails closed,
    such as one holding a secret key whose value is on the next line or one too long for the redactor, becomes one
    placeholder line, and none of it is kept.
    """
    # A private key can be longer than a block and its context, so keys are removed from the whole text first.
    joined = redact_private_keys("\n".join(text.splitlines()))
    lines = joined.split("\n")
    for start, block in _blocks(lines, block_chars):
        context_start = max(0, start - _REDACTION_CONTEXT_CHARS)
        # Context starts at a line where it can, so it holds whole lines; a longer previous line gives only its end.
        if context_start > 0 and (line_break := joined.find("\n", context_start, start - 1)) != -1:
            context_start = line_break + 1
        context = joined[context_start:start]
        window = context + "\n".join(block)
        redacted = redact_sensitive_text(window)
        if redacted.count("\n") != window.count("\n"):
            yield REDACTION_FAILED
            continue
        yield from redacted.split("\n")[context.count("\n") :]


def _blocks(lines: list[str], block_chars: int) -> Iterator[tuple[int, list[str]]]:
    """Yield runs of whole lines of at most ``block_chars`` characters, a longer line alone, with where each starts."""
    block: list[str] = []
    block_start = offset = size = 0
    for line in lines:
        if block and size + len(line) + 1 > block_chars:
            yield block_start, block
            block, block_start, size = [], offset, 0
        block.append(line)
        size += len(line) + 1
        offset += len(line) + 1
    if block:
        yield block_start, block
