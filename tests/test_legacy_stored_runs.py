"""Turns stored by older releases are read with the room and thread their session ID names."""

from __future__ import annotations

from typing import Any

import pytest
from agno.run.agent import RunOutput
from agno.run.base import RunStatus

from mindroom.history.legacy_stored_runs import recover_legacy_run_metadata


def _run(session_id: str, metadata: dict[str, Any] | None, *, parent_run_id: str | None = None) -> RunOutput:
    return RunOutput(
        run_id="r",
        session_id=session_id,
        parent_run_id=parent_run_id,
        metadata=metadata,
        status=RunStatus.completed,
    )


def test_recovers_room_and_thread_from_the_session_id() -> None:
    """A legacy thread turn gets the room and thread root its session ID was built from."""
    run = _run("!abc:mindroom.chat:$root", {"matrix_event_id": "$e"})

    recover_legacy_run_metadata(run)

    assert run.metadata == {"matrix_event_id": "$e", "room_id": "!abc:mindroom.chat", "thread_id": "$root"}


def test_room_mode_session_has_no_thread() -> None:
    """A session ID without a thread root is a whole room, so the turn gets the room and no thread."""
    run = _run("!abc:mindroom.chat", {"matrix_event_id": "$e"})

    recover_legacy_run_metadata(run)

    assert run.metadata is not None
    assert run.metadata["room_id"] == "!abc:mindroom.chat"
    assert run.metadata.get("thread_id") is None


@pytest.mark.parametrize(
    ("session_id", "metadata"),
    [
        ("openai:key:abc", {"matrix_event_id": "$e"}),
        ("delegate:x", {"matrix_event_id": "$e"}),
        ("!abc:mindroom.chat:$root", None),
        ("!abc:mindroom.chat:$root", {}),
        ("!abc:mindroom.chat:$root", {"room_id": "!other:x"}),
        ("!abc:mindroom.chat:$root", {"matrix_event_id": "$e", "room_id": "!other:x", "thread_id": "$t"}),
    ],
)
def test_leaves_other_runs_alone(session_id: str, metadata: dict[str, Any] | None) -> None:
    """Runs outside a Matrix room, without a Matrix event, or already naming their room are unchanged."""
    run = _run(session_id, None if metadata is None else dict(metadata))

    recover_legacy_run_metadata(run)

    assert run.metadata == metadata


def test_member_runs_are_left_alone() -> None:
    """Only a top-level turn is a Matrix turn; a member run inside it keeps its metadata."""
    run = _run("!abc:mindroom.chat:$root", {"matrix_event_id": "$e"}, parent_run_id="top")

    recover_legacy_run_metadata(run)

    assert run.metadata == {"matrix_event_id": "$e"}
