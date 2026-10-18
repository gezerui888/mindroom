"""Background ingestion: the finished turns of graph-enabled agents, and files of flagged knowledge bases, become episodes.

Stored sessions and the files are the source of truth; a per-graph state file records how far each conversation has
been ingested and which content of each file was, so a restart resumes where the last pass stopped and changing the
embedder, which names a new graph, starts over.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypedDict

from mindroom.automations.threads import automation_threads
from mindroom.durable_write import write_json_file_durable
from mindroom.entity_resolution import prepared_entity_names_by_user_id
from mindroom.history.session_runs import finished_runs, run_key, session_runs, unredacted_runs, updated_sessions
from mindroom.knowledge_graph.documents import document_files, read_document, render_document
from mindroom.knowledge_graph.episodes import (
    RUN_DONE,
    Progress,
    RenderedEpisode,
    classify_thread,
    render_episodes,
)
from mindroom.knowledge_graph.scope import graph_scopes
from mindroom.knowledge_graph.store import UnusableExtractionError, record_knowledge_graph_failure
from mindroom.knowledge_graph.usage import UsageTarget
from mindroom.logging_config import get_logger
from mindroom.session_storage.agent_storage import create_session_storage

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from agno.run.agent import RunOutput

    from mindroom.config.main import Config
    from mindroom.constants import RuntimePaths
    from mindroom.history.session_runs import RedactedEventIds
    from mindroom.knowledge_graph.documents import DocumentFile
    from mindroom.knowledge_graph.scope import GraphScope
    from mindroom.knowledge_graph.store import GraphStore

logger = get_logger(__name__)

_DAY_SECONDS = 86_400
_IDLE_POLL_SECONDS = 30
_FIRST_RETRY_SECONDS = 60
_MAX_RETRY_SECONDS = 3600
# An episode whose extraction output is unusable this often, or that has failed for this long, is dropped, so it
# cannot cost model calls forever; an outage shorter than that loses nothing.
_MAX_BAD_OUTPUTS = 5
_MAX_FAILING_SECONDS = 3 * 86_400
# Episodes one conversation or file adds per pass, so a long backlog cannot hold up every other one.
_MAX_EPISODES_PER_PASS = 20
_EPISODE_KINDS = ("conversation", "tool_results")
_WAKE_EVENTS: set[asyncio.Event] = set()


class _Backoff(TypedDict):
    # Consecutive failures, which set the backoff.
    failures: int
    # Consecutive failures of the current episode whose model output could not be used, which drop it.
    bad_outputs: int
    # When the current episode started failing, so one that never succeeds is dropped after a while.
    failing_since: float
    retry_at: float


class _SessionState(_Backoff):
    updated_at: int
    # The last run each episode kind has ingested, so a failing kind never repeats the other.
    markers: dict[str, list[int | str]]
    # The failing episode, as its kind and where that kind stopped, which stays put while it fails; empty when none.
    failing_piece: list[int | str]
    # When a turn holding the conversation back stops waiting for approval, or 0.
    recheck_at: float


class _DocumentState(_Backoff):
    # The content last extracted in full; a file with this hash is never extracted again.
    sha256: str
    # The file's hash when last read and its `[size, mtime_ns, inode]` then, so an unchanged file is not read again;
    # `pieces_done` and the failure counts belong to this content, so an edit during a retry starts over.
    seen_sha256: str
    seen_stat: list[int]
    # Pieces of the seen content added or dropped, so a retry resumes at the failed piece.
    pieces_done: int


class _GraphState(TypedDict):
    started_at: float
    sessions: dict[str, _SessionState]
    # Keyed `<base_id>/<relative path>`.
    documents: dict[str, _DocumentState]


@dataclass(frozen=True)
class _DueSession:
    session_id: str
    updated_at: int
    runs: list[RunOutput]


def wake_knowledge_graph_ingestion(*, after_seconds: float = 0) -> None:
    """Run an ingestion pass after ``after_seconds`` instead of at the next interval; call on the event loop."""

    def wake() -> None:
        for event in _WAKE_EVENTS:
            event.set()

    asyncio.get_running_loop().call_later(after_seconds, wake)


def _new_session_state() -> _SessionState:
    return {
        "updated_at": 0,
        "markers": {},
        "failures": 0,
        "bad_outputs": 0,
        "failing_since": 0,
        "failing_piece": [],
        "retry_at": 0,
        "recheck_at": 0,
    }


def _start_after_backfill_window(entry: _SessionState, runs: list[RunOutput], *, cutoff: float) -> None:
    """Mark the runs before the backfill window as done, so a long conversation adds only its recent turns."""
    older = [run for run in runs if (run.created_at or 0) < cutoff]
    if older:
        done = [*run_key(older[-1]), RUN_DONE]
        entry["markers"] = {kind: list(done) for kind in _EPISODE_KINDS}


def _marker(entry: _SessionState, kind: str) -> Progress | None:
    marker = entry["markers"].get(kind)
    return (int(marker[0]), str(marker[1]), int(marker[2])) if marker is not None else None


def _new_document_state() -> _DocumentState:
    return {
        "sha256": "",
        "seen_sha256": "",
        "seen_stat": [],
        "pieces_done": 0,
        "failures": 0,
        "bad_outputs": 0,
        "failing_since": 0,
        "retry_at": 0,
    }


def _report_failure(message: str, exc: Exception, **context: str) -> None:
    logger.warning(message, error=f"{type(exc).__name__}: {str(exc)[:200]}", **context)
    # Health is unauthenticated and error text can quote conversation content, so it gets only the failure's kind.
    record_knowledge_graph_failure(f"{message}: {type(exc).__name__}")


def _back_off(entry: _Backoff, exc: Exception, now: float) -> bool:
    """Count a failure of the entry's current episode; return whether it waits for a retry rather than being dropped."""
    entry["failures"] += 1
    # Only output the model got wrong counts here; a setup error such as a bad embedder waits like an outage.
    entry["bad_outputs"] = entry["bad_outputs"] + 1 if isinstance(exc, UnusableExtractionError) else 0
    entry["failing_since"] = entry["failing_since"] or now
    if entry["bad_outputs"] < _MAX_BAD_OUTPUTS and now - entry["failing_since"] < _MAX_FAILING_SECONDS:
        backoff = _FIRST_RETRY_SECONDS * 2 ** (entry["failures"] - 1)
        entry["retry_at"] = now + min(backoff, _MAX_RETRY_SECONDS)
        return True
    return False


@dataclass
class KnowledgeGraphIngestionWorker:
    """Periodically ingest the new finished turns of every graph-enabled agent and private instance."""

    runtime_paths: RuntimePaths
    config_provider: Callable[[], Config | None]
    store: GraphStore
    redacted_event_ids: RedactedEventIds
    _wake_event: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _passes: int = field(default=0, init=False)
    # Graphs with a conversation waiting to retry a failed episode, which keep health failing until it succeeds.
    _failing_graphs: set[str] = field(default_factory=set, init=False)

    async def run(self) -> None:
        """Run ingestion passes until cancelled."""
        _WAKE_EVENTS.add(self._wake_event)
        try:
            while True:
                # Cleared before the pass, so a wake that arrives during it starts the next one at once.
                self._wake_event.clear()
                config = self.config_provider()
                interval = _IDLE_POLL_SECONDS
                if config is not None and config.knowledge_graph_agents():
                    try:
                        await self._run_cycle(config, now=time.time())
                    except Exception as exc:
                        _report_failure("Knowledge graph ingestion pass failed", exc)
                    interval = config.knowledge_graph.ingest.interval_seconds
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._wake_event.wait(), timeout=interval)
        finally:
            _WAKE_EVENTS.discard(self._wake_event)

    async def _run_cycle(self, config: Config, *, now: float) -> None:
        """Ingest up to ``max_sessions_per_cycle`` new or changed files, then idle conversations with new turns."""
        budget = config.knowledge_graph.ingest.max_sessions_per_cycle
        # These read files, including every private instance's records, so they stay off the event loop.
        automation_thread_ids = await asyncio.to_thread(automation_threads, self.runtime_paths)
        entity_names = await asyncio.to_thread(prepared_entity_names_by_user_id, config, self.runtime_paths)
        outcomes: list[bool] = []
        scopes = await asyncio.to_thread(graph_scopes, config, self.runtime_paths)
        # A graph no longer maintained, such as one named for a previous embedder, holds no failure.
        self._failing_graphs &= {scope.graph_name for scope in scopes}
        # Each pass starts with the next scope, so one with a long backlog cannot take every pass's budget.
        start = self._passes % len(scopes) if scopes else 0
        self._passes += 1
        for scope in [*scopes[start:], *scopes[:start]]:
            if budget <= 0:
                break
            # One agent or private instance whose conversations cannot be read must not stop the others.
            try:
                state = await asyncio.to_thread(self._load_state, scope.graph_name, now)
                # Files come first, each new or changed one spending one unit; an unchanged file spends nothing.
                document_outcomes, documents_failing = await self._ingest_documents(config, scope, state, now, budget)
                budget -= len(document_outcomes)
                outcomes.extend(document_outcomes)
                due, readable = await asyncio.to_thread(self._due_sessions, config, scope, state, now, budget)
                if not readable:
                    outcomes.append(False)
                results = [
                    await self._ingest(config, scope, state, session, automation_thread_ids, entity_names.get, now)
                    for session in due
                ]
                # A conversation whose agent is not running is looked at again later and spends none of the budget.
                budget -= sum(result is not None for result in results)
                outcomes.extend(result for result in results if result is not None)
                if documents_failing or any(entry["failing_piece"] for entry in state["sessions"].values()):
                    self._failing_graphs.add(scope.graph_name)
                else:
                    self._failing_graphs.discard(scope.graph_name)
            except Exception as exc:
                _report_failure("Knowledge graph ingestion failed for an agent", exc, agent=scope.agent_name)
                outcomes.append(False)
        # Health reports a failure until a pass ingests something, nothing in it fails, and no failed episode waits.
        if outcomes and all(outcomes) and not self._failing_graphs:
            record_knowledge_graph_failure(None)

    def _due_sessions(
        self,
        config: Config,
        scope: GraphScope,
        state: _GraphState,
        now: float,
        limit: int,
    ) -> tuple[list[_DueSession], bool]:
        """Return up to ``limit`` idle conversations with new turns, and whether every conversation could be read."""
        settings = config.knowledge_graph.ingest
        storage = create_session_storage(
            scope.agent_name,
            config,
            self.runtime_paths,
            execution_identity=scope.execution_identity,
        )
        try:
            since = int(state["started_at"]) - settings.backfill_days * _DAY_SECONDS
            due: list[_DueSession] = []
            readable = True
            for session_id, updated_at in updated_sessions(storage, scope.agent_name, since=since):
                if len(due) >= limit:
                    break
                entry = state["sessions"].get(session_id)
                if entry is not None and (
                    entry["retry_at"] > now
                    or (updated_at <= entry["updated_at"] and not 0 < entry["recheck_at"] <= now)
                ):
                    continue
                if now - updated_at < settings.idle_seconds:
                    continue
                try:
                    runs = session_runs(storage, session_id)
                except Exception as exc:
                    # One conversation that cannot be read must not stop the agent's others.
                    _report_failure(
                        "Knowledge graph ingestion could not read a conversation",
                        exc,
                        agent=scope.agent_name,
                    )
                    readable = False
                    continue
                due.append(_DueSession(session_id, updated_at, runs))
            return due, readable
        finally:
            storage.close()

    async def _ingest(
        self,
        config: Config,
        scope: GraphScope,
        state: _GraphState,
        session: _DueSession,
        automation_thread_ids: set[str],
        agent_for_sender: Callable[[str], str | None],
        now: float,
    ) -> bool | None:
        """Ingest a bounded number of one conversation's new episodes and return whether none failed or was dropped.

        Return None when its agent cannot say yet what was redacted, which leaves the conversation for a later pass.
        """
        settings = config.knowledge_graph.ingest
        finished, blocked_until = finished_runs(session.runs, config, now=now)
        finished = await unredacted_runs(finished, scope.agent_name, self.redacted_event_ids)
        if finished is None:
            return None
        entry = state["sessions"].get(session.session_id)
        if entry is None:
            entry = state["sessions"][session.session_id] = _new_session_state()
            _start_after_backfill_window(
                entry,
                finished,
                cutoff=state["started_at"] - settings.backfill_days * _DAY_SECONDS,
            )
        thread_kind = classify_thread(session.runs, automation_thread_ids)
        clean = True
        added = 0
        for kind in _EPISODE_KINDS:
            marker = _marker(entry, kind)
            # A run whose pieces are partly done is rendered again, and its done pieces are skipped below.
            new_runs = [run for run in finished if marker is None or (*run_key(run), RUN_DONE) > marker]
            # Rendering redacts every tool result of the backlog, which must not stall the event loop.
            episodes = await asyncio.to_thread(
                render_episodes,
                new_runs,
                thread_kind=thread_kind,
                episode_kind=kind,
                agent_name=scope.agent_name,
                settings=settings,
                agent_for_sender=agent_for_sender,
            )
            for rendered in episodes:
                if marker is not None and rendered.progress <= marker:
                    continue
                usage = UsageTarget(
                    scope.agent_name,
                    session.session_id,
                    scope.execution_identity,
                    rendered.requester_id,
                )
                outcome = await self._add(scope, state, entry, kind, rendered, usage, now)
                if outcome is None:
                    return False
                clean = clean and outcome
                added += 1
                # A long backlog spreads over passes; the conversation stays due because its update time is unchanged.
                if added >= _MAX_EPISODES_PER_PASS:
                    return clean
            if new_runs:
                entry["markers"][kind] = [*run_key(new_runs[-1]), RUN_DONE]
        # Held back by a turn waiting for approval, it is looked at again when those approvals expire, even without news.
        entry.update(
            updated_at=session.updated_at,
            recheck_at=blocked_until or 0,
            failures=0,
            bad_outputs=0,
            failing_since=0,
            failing_piece=[],
            retry_at=0,
        )
        await self._persist(scope.graph_name, state)
        return clean

    async def _add(
        self,
        scope: GraphScope,
        state: _GraphState,
        entry: _SessionState,
        kind: str,
        rendered: RenderedEpisode,
        usage: UsageTarget,
        now: float,
    ) -> bool | None:
        """Add one episode; return True when added, False when dropped, and None when it waits for a later pass."""
        piece = [kind, *entry["markers"].get(kind, [])]
        try:
            await self.store.add_episode(scope, rendered.episode, usage)
            dropped = False
        except Exception as exc:
            if entry["failing_piece"] != piece:
                # The counts belong to one episode, so another kind's successes never reset them.
                entry.update(failures=0, bad_outputs=0, failing_since=0, failing_piece=piece)
            _report_failure("Knowledge graph ingestion failed", exc, agent=scope.agent_name)
            if _back_off(entry, exc, now):
                await self._persist(scope.graph_name, state)
                return None
            logger.exception(
                "Knowledge graph dropped an episode it could not extract",
                agent=scope.agent_name,
                episode=rendered.episode.name,
            )
            dropped = True
        # Every piece records its progress, added or dropped, so a retry never repeats it.
        if entry["failing_piece"] == piece:
            entry.update(failures=0, bad_outputs=0, failing_since=0, failing_piece=[])
        entry["markers"][kind] = list(rendered.progress)
        await self._persist(scope.graph_name, state)
        return not dropped

    async def _ingest_documents(
        self,
        config: Config,
        scope: GraphScope,
        state: _GraphState,
        now: float,
        limit: int,
    ) -> tuple[list[bool], bool]:
        """Ingest up to ``limit`` new or changed files of the scope's flagged knowledge bases.

        Return whether each ingested file had no piece fail or dropped, and whether a base could not be listed or a
        listed file waits for a retry. A deleted file is left alone, and so are the facts it added.
        """
        known = {
            key: ((entry["seen_stat"][0], entry["seen_stat"][1], entry["seen_stat"][2]), entry["seen_sha256"])
            for key, entry in state["documents"].items()
            if len(entry["seen_stat"]) == 3
        }
        # Listing opens every flagged file and reads the changed ones, which must not stall the event loop.
        files, errors = await asyncio.to_thread(document_files, config, scope, self.runtime_paths, known)
        for base_id, exc in errors.items():
            _report_failure(
                "Knowledge graph could not list a knowledge base",
                exc,
                agent=scope.agent_name,
                base=base_id,
            )
        seen_changed = False
        for file in files:
            entry = state["documents"].setdefault(file.key, _new_document_state())
            if entry["seen_sha256"] != file.sha256:
                # Progress and failure counts belong to one content; a backoff still waiting keeps waiting.
                entry.update(seen_sha256=file.sha256, pieces_done=0, failures=0, bad_outputs=0, failing_since=0)
            if entry["seen_stat"] != list(file.stat):
                entry["seen_stat"] = list(file.stat)
                seen_changed = True
        if seen_changed:
            await self._persist(scope.graph_name, state)
        outcomes: list[bool] = []
        for file in files:
            if len(outcomes) >= limit:
                break
            entry = state["documents"][file.key]
            if entry["sha256"] == file.sha256 or entry["retry_at"] > now:
                continue
            outcome = await self._ingest_document(config, scope, state, file, now)
            if outcome is not None:
                outcomes.append(outcome)
        # A file deleted, or changed back to what was extracted, while a piece of it waited no longer keeps health failing.
        failing = bool(errors) or any(state["documents"][file.key]["failures"] > 0 for file in files)
        return outcomes, failing

    async def _ingest_document(
        self,
        config: Config,
        scope: GraphScope,
        state: _GraphState,
        file: DocumentFile,
        now: float,
    ) -> bool | None:
        """Add a bounded number of one file's pieces and return whether none failed or was dropped.

        Return None when the file changed after it was listed, which leaves it for the next pass.
        """
        text = await asyncio.to_thread(read_document, file)
        if text is None:
            return None
        entry = state["documents"][file.key]
        # Rendering redacts the file line by line, which must not stall the event loop; it stops one piece past what
        # this pass may add, which tells whether more remain.
        episodes = await asyncio.to_thread(
            render_document,
            file,
            text,
            config.knowledge_graph.ingest,
            limit=entry["pieces_done"] + _MAX_EPISODES_PER_PASS + 1,
        )
        usage = UsageTarget(scope.agent_name, None, scope.execution_identity)
        clean = True
        for added, index in enumerate(range(entry["pieces_done"], len(episodes))):
            # A long file spreads over passes; it stays due because its content is not recorded as extracted yet.
            if added >= _MAX_EPISODES_PER_PASS:
                return clean
            try:
                await self.store.add_episode(scope, episodes[index], usage)
            except Exception as exc:
                _report_failure(
                    "Knowledge graph ingestion failed for a document",
                    exc,
                    agent=scope.agent_name,
                    document=file.key,
                )
                if _back_off(entry, exc, now):
                    await self._persist(scope.graph_name, state)
                    return False
                logger.exception(
                    "Knowledge graph dropped a document piece it could not extract",
                    agent=scope.agent_name,
                    episode=episodes[index].name,
                )
                clean = False
            # Every piece records its progress, added or dropped, so a retry never repeats it.
            entry.update(pieces_done=index + 1, failures=0, bad_outputs=0, failing_since=0, retry_at=0)
            await self._persist(scope.graph_name, state)
        entry.update(sha256=file.sha256, pieces_done=0)
        await self._persist(scope.graph_name, state)
        return clean

    async def _persist(self, graph_name: str, state: _GraphState) -> None:
        await asyncio.to_thread(write_json_file_durable, self._state_path(graph_name), state)

    def _state_path(self, graph_name: str) -> Path:
        return self.runtime_paths.storage_root / "knowledge_graph" / "ingestion" / f"{graph_name}.json"

    def _load_state(self, graph_name: str, now: float) -> _GraphState:
        try:
            saved: _GraphState = json.loads(self._state_path(graph_name).read_text(encoding="utf-8"))
        except FileNotFoundError:
            pass
        except ValueError:
            # Starting over re-extracts recent conversations, which Graphiti merges with what it already holds.
            logger.warning("Knowledge graph ingestion state was unreadable; starting over", graph=graph_name)
        else:
            # LEGACY_COMPAT: Ingestion state saved before knowledge-base files fed the graph.
            # Legacy format: The state file has no `documents` key.
            # Last legacy release: v2026.10.344, like every release since graph ingestion shipped in v2026.10.317;
            # replacement: the first release with document ingestion saves `documents`, each file's progress by key.
            # Handling: A missing `documents` starts empty, so every flagged file counts as new; conversation progress is kept.
            # Coverage: tests/test_knowledge_graph_ingestion.py::test_state_saved_before_documents_keeps_its_conversations_and_adds_documents.
            saved.setdefault("documents", {})
            return saved
        # Saved at once, so the backfill window stays anchored to when the graph started.
        state: _GraphState = {"started_at": now, "sessions": {}, "documents": {}}
        write_json_file_durable(self._state_path(graph_name), state)
        return state
