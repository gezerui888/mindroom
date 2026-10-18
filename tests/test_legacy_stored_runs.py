"""Turns stored by older releases are read with the room and thread their session ID names, and people's words."""

from __future__ import annotations

from typing import Any

import pytest
from agno.run.agent import RunOutput
from agno.run.base import RunStatus

from mindroom.history.legacy_stored_runs import (
    is_recovered_legacy_run,
    legacy_prompt_utterances,
    recover_legacy_run_metadata,
)
from mindroom.prompt_message_tags import render_msg_tag


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

    assert run.metadata is not None
    assert (run.metadata["room_id"], run.metadata["thread_id"]) == ("!abc:mindroom.chat", "$root")
    assert is_recovered_legacy_run(run)


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
    assert not is_recovered_legacy_run(run)


def test_member_runs_are_left_alone() -> None:
    """Only a top-level turn is a Matrix turn; a member run inside it keeps its metadata."""
    run = _run("!abc:mindroom.chat:$root", {"matrix_event_id": "$e"}, parent_run_id="top")

    recover_legacy_run_metadata(run)

    assert run.metadata == {"matrix_event_id": "$e"}


MEMORY_FILE = "[File memory entrypoint (agent)]\n# MEMORY.md - Long-term Memory\n\n## Alice Example\n\n- **Prefers short answers**"
MEMORY_LIST = (
    "[Automatically extracted agent file memories - may not be relevant to current context]\n"
    "Previous agent file memories that might be related:\n"
    "- [auto_flush:!r:x:$t:1] User is debugging Atlas.\n"
    "- [auto_flush:!r:x:$u:2] Atlas lacks audio tools."
)
ATTACHMENTS = "Available attachment IDs: att_73f950d84ac762f89e60c6c2. Use tool calls to inspect or process them."
MATRIX_METADATA = (
    "[Matrix metadata for tool calls]\nroom_id: !r:x\nthread_id: $t\nreply_to_event_id: $e\n"
    "Use these IDs when calling matrix_message."
)
INLINE_MEDIA = (
    "[Inline media unavailable for this model] The model rejected inline attachments for this turn. "
    "Use available attachment IDs and tools to inspect files instead."
)
CURRENT = (
    "Messages from other participants since your last response:\n"
    "@alice:example.org: [2026-04-09 16:11 PDT] hi\n\n"
    f"Current message:\n{MEMORY_FILE}\n\n{MEMORY_LIST}\n\nWhat can you do?\n\n{ATTACHMENTS}\n\n{MATRIX_METADATA}"
)


def test_legacy_prompt_keeps_only_peoples_words() -> None:
    """Earlier messages are their senders' words and the current message is what follows the memories."""
    assert legacy_prompt_utterances(CURRENT, "@alice:example.org") == [
        ("@alice:example.org", "hi"),
        ("@alice:example.org", "What can you do?"),
    ]


def test_previous_conversation_copies_are_skipped() -> None:
    """A copy of the thread so far repeats earlier turns, so only the current message is read."""
    prompt = (
        "Previous conversation in this thread:\n"
        "@r:x: [2026-03-10 09:00 PDT] Atlas ships Friday?\n"
        "@mindroom_mind:x: It ships Friday.\n\n"
        "Current message:\n"
        "[2026-03-10 09:05 PDT] And Bolt?"
    )

    assert legacy_prompt_utterances(prompt, "@r:x") == [("@r:x", "And Bolt?")]


def test_interrupted_draft_notes_are_dropped() -> None:
    """The agent's own interrupted draft and the note about it are MindRoom's, not anyone's words."""
    prompt = (
        "Messages since your last response:\n"
        "Your previous response was interrupted before completion. The partial content below may be incomplete. "
        "Continue from where you left off if appropriate.\n"
        "@r:x: [2026-03-28 10:00 PDT] first line\n"
        "second line\n"
        "You (interrupted reply draft): I was checking\n"
        "the deploy logs\n"
        "@carol:x: Bolt is late.\n\n"
        "Current message:\n"
        "[2026-03-28 10:02 PDT] go on"
    )

    assert legacy_prompt_utterances(prompt, "@r:x") == [
        ("@r:x", "first line\nsecond line"),
        ("@carol:x", "Bolt is late."),
        ("@r:x", "go on"),
    ]


def test_memory_entrypoint_without_bullet_list_drops_the_current_message() -> None:
    """Where a memory file ends cannot be told without a list after it, so nothing is attributed."""
    assert (
        legacy_prompt_utterances("Current message:\n[File memory entrypoint (agent)]\n# MEMORY.md\n\nhello", "@r:x")
        == []
    )


def test_a_memory_entrypoint_after_the_last_memory_list_drops_the_current_message() -> None:
    """A room memory file after the agent's memory list also has no end to cut at."""
    prompt = (
        "[Automatically extracted agent file memories - may not be relevant to current context]\n"
        "Previous agent file memories that might be related:\n"
        "- Atlas ships Friday.\n\n"
        "[File memory entrypoint (room)]\n# Room notes\n\nBolt is late.\n\n"
        "hello"
    )

    assert legacy_prompt_utterances(prompt, "@r:x") == []


def test_plain_legacy_message_is_the_requesters() -> None:
    """A prompt with no sections is the requester's message without its time prefix."""
    assert legacy_prompt_utterances("[2026-04-13 10:53 PDT] continue", "@r:x") == [("@r:x", "continue")]
    assert legacy_prompt_utterances("The user selected: shell", "@r:x") == [("@r:x", "The user selected: shell")]


def test_unknown_sections_are_dropped() -> None:
    """Text under a heading no release wrote, or earlier messages without a current message, is nobody's words."""
    prompt = "Some other heading:\n@bob:x: hi\n\nCurrent message:\nhey"

    assert legacy_prompt_utterances(prompt, "@r:x") == [("@r:x", "hey")]
    assert legacy_prompt_utterances("Messages since your last response:\n@bob:x: hi", "@r:x") == []


def test_a_bare_time_prefix_is_nobodys_words() -> None:
    """A message that is only its time prefix says nothing, so it yields no utterance."""
    assert legacy_prompt_utterances("[2026-04-13 10:53 PDT]", "@r:x") == []
    assert legacy_prompt_utterances("Current message:\n[2026-04-13 10:53 PDT]\n\n" + ATTACHMENTS, "@r:x") == []


def test_attachment_metadata_and_inline_media_blocks_are_dropped() -> None:
    """The attachment line, the Matrix IDs block, and the inline media note are MindRoom's, in the order it added them."""
    prompt = f"[2026-03-20 08:00 PDT] What is in this file?\n\n{ATTACHMENTS}\n\n{MATRIX_METADATA}\n\n{INLINE_MEDIA}"

    assert legacy_prompt_utterances(prompt, "@r:x") == [("@r:x", "What is in this file?")]
    assert legacy_prompt_utterances(f"Current message:\n{prompt}", "@r:x") == [("@r:x", "What is in this file?")]


def test_without_a_current_message_heading_only_plain_text_is_kept() -> None:
    """Without the heading, MindRoom's blocks are dropped, and a transcript leaves nothing that can be placed."""
    assert legacy_prompt_utterances("Thread Context:\n@bob:x: secret\n\nhello", "@r:x") == []
    assert legacy_prompt_utterances("@bob:x: secret\n\nhello", "@r:x") == []
    prompt = f"{MEMORY_FILE}\n\n{MEMORY_LIST}\n\n[2026-03-20 08:00 PDT] hello\n\nsecond paragraph\n\n{ATTACHMENTS}"
    assert legacy_prompt_utterances(prompt, "@r:x") == [("@r:x", "hello\n\nsecond paragraph")]


def test_tagged_legacy_messages_lose_mindroom_blocks_and_drafts() -> None:
    """Tagged releases kept MindRoom's blocks inside the requester's tag, before or after the text, and tagged drafts."""
    through_v2026_4_198 = f"{MEMORY_FILE}\n\n{MEMORY_LIST}\n\n[2026-04-18 10:00 PDT] Ship it?\n\n{MATRIX_METADATA}"
    from_v2026_4_199 = f"Ship it now?\n\n{MEMORY_LIST}\n\n{ATTACHMENTS}\n\n{MATRIX_METADATA}"
    prompt = (
        "Messages since your last response:\n<conversation>\n"
        + render_msg_tag(sender="@bob:x", body="[2026-04-18 09:58 PDT] Bolt is late.")
        + render_msg_tag(sender="You (interrupted reply draft)", body="I was checking")
        + "\n</conversation>\n\nCurrent message:\n"
        + render_msg_tag(sender="@r:x", body=through_v2026_4_198)
    )

    assert legacy_prompt_utterances(prompt, "@r:x") == [("@bob:x", "Bolt is late."), ("@r:x", "Ship it?")]
    assert legacy_prompt_utterances(render_msg_tag(sender="@r:x", body=from_v2026_4_199), "@r:x") == [
        ("@r:x", "Ship it now?"),
    ]
    assert legacy_prompt_utterances(render_msg_tag(sender="@r:x", body=f"{MEMORY_FILE}\n\nhello"), "@r:x") == []
