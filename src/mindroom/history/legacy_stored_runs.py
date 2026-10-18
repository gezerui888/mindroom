"""Read Matrix turns that older releases stored in an earlier shape."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from mindroom.constants import MATRIX_EVENT_ID_METADATA_KEY

if TYPE_CHECKING:
    from agno.run.agent import RunOutput
    from agno.run.team import TeamRunOutput

# A Matrix thread session ID is the room ID and the thread root event ID joined by `:`.
_THREAD_SEPARATOR = ":$"
# Set on a recovered run's loaded metadata, so readers know its prompts were written in an earlier layout too.
_RECOVERED_KEY = "room_id_from_session_id"

# Headings of the earlier messages a legacy prompt put before the current message: new ones, or a copy of the thread.
_NEW_MESSAGES_HEADERS = (
    "Messages from other participants since your last response:",
    "Messages since your last response:",
)
_THREAD_COPY_HEADER = "Previous conversation in this thread:"
_CURRENT_MESSAGE = re.compile(r"(?:^|\n\n)Current message:\n")
_SENDER_LINE = re.compile(r"(@[^\s:]+:\S+?): (.*)")
_OWN_DRAFT_LINE = re.compile(r"You \([^)]*\): ")
_TIME_PREFIX = re.compile(r"^\[(?:\d{4}-\d{2}-\d{2} )?\d{2}:\d{2} [^\]]+\]\s")
_MEMORY_LIST_HEADER = re.compile(r"^Previous .+ memories that might be related:$", re.MULTILINE)
_MEMORY_ENTRYPOINT = "[File memory entrypoint"
_TRAILING_BLOCK = re.compile(
    r"\n*(?:Available attachment IDs: [^\n]*\. Use tool calls to inspect or process them\."
    r"|\[Matrix metadata for tool calls\]\nroom_id: [^\n]*\nthread_id: [^\n]*\nreply_to_event_id: [^\n]*\n"
    r"Use these IDs when calling matrix_message\.)\s*$",
)


# LEGACY_COMPAT: Run metadata without room_id.
# Legacy format: Matrix turns stored their event IDs (`matrix_event_id`, `matrix_seen_event_ids`) in run metadata but
# no `room_id` or `thread_id`; the rule selects a top-level run whose metadata holds `matrix_event_id` and no `room_id`
# and whose session ID is a Matrix room ID (`!...`), optionally followed by `:` and the thread root event ID.
# A current turn interrupted before its run started is stored from the recorder's seed metadata, which also lacks
# `room_id`; the rule applies to it too, and its session ID names the same room.
# Last legacy release: v2026.5.26; replacement: v2026.5.27 (#665) stores `room_id` and `thread_id` in run metadata.
# Handling: The room and thread root are read from the session ID (split at the first `:$`; none means a room-mode
# session without a thread) and written into the loaded run's metadata, so background readers group, filter, and
# describe the turn by its real room, and marked recovered for the prompt rule below. Stored rows are not rewritten.
# Coverage: tests/test_legacy_stored_runs.py::test_recovers_room_and_thread_from_the_session_id,
# tests/test_legacy_stored_runs.py::test_room_mode_session_has_no_thread,
# tests/test_legacy_stored_runs.py::test_leaves_other_runs_alone,
# tests/test_legacy_stored_runs.py::test_member_runs_are_left_alone,
# tests/test_history_session_runs.py::test_session_runs_returns_legacy_runs_with_their_room_and_thread, and
# tests/test_history_session_runs.py::test_legacy_runs_are_checked_for_redactions_in_their_room.
def recover_legacy_run_metadata(run: RunOutput) -> None:
    """Give a turn stored before run metadata named its room the room and thread its session ID was built from."""
    metadata = run.metadata
    session_id = run.session_id or ""
    if (
        not metadata
        or run.parent_run_id is not None
        or metadata.get("room_id")
        or not metadata.get(MATRIX_EVENT_ID_METADATA_KEY)
        or not session_id.startswith("!")
    ):
        return
    room_id, separator, thread_id = session_id.partition(_THREAD_SEPARATOR)
    metadata["room_id"] = room_id
    if separator:
        metadata["thread_id"] = f"${thread_id}"
    metadata[_RECOVERED_KEY] = True


def is_recovered_legacy_run(run: RunOutput | TeamRunOutput) -> bool:
    """Return whether the rule above gave the run its room, as it does for runs stored before metadata named it."""
    return (run.metadata or {}).get(_RECOVERED_KEY) is True


# LEGACY_COMPAT: Matrix prompt text without <msg> tags.
# Legacy format: A user message held the prompt MindRoom built as plain text: optionally a heading
# (`Messages from other participants since your last response:`, `Messages since your last response:`, or
# `Previous conversation in this thread:`) over `@sender:server: body` lines, then `Current message:`; the current
# message follows memory blocks (`[File memory entrypoint (...)]` files, `Previous ... memories that might be related:`
# lists) and ends with `Available attachment IDs:` lines and a `[Matrix metadata for tool calls]` block. The rule
# selects a user message without `<msg>` tags in a run recovered by the rule above.
# Last legacy release: v2026.4.174; replacement: v2026.4.175 (#645) wraps earlier messages, and the current message
# when its sender is known, in `<msg from="..."><![CDATA[...]]></msg>` tags. Runs through v2026.5.26 are recovered
# too, so an untagged message of theirs is read the same way.
# Handling: New earlier messages are their senders' words (time prefix stripped, continuation lines joined); the
# agent's own `You (...)` drafts, notes under the heading, and thread copies are dropped. The current message is the
# requester's text after the last memory list, or nothing when a memory file has no list after it. Anything that
# matches no known section is dropped, never attributed.
# Coverage: tests/test_legacy_stored_runs.py::test_legacy_prompt_keeps_only_peoples_words,
# tests/test_legacy_stored_runs.py::test_previous_conversation_copies_are_skipped,
# tests/test_legacy_stored_runs.py::test_interrupted_draft_notes_are_dropped,
# tests/test_legacy_stored_runs.py::test_memory_entrypoint_without_bullet_list_drops_the_current_message,
# tests/test_legacy_stored_runs.py::test_a_memory_entrypoint_after_the_last_memory_list_drops_the_current_message,
# tests/test_legacy_stored_runs.py::test_plain_legacy_message_is_the_requesters,
# tests/test_legacy_stored_runs.py::test_unknown_sections_are_dropped, and
# tests/test_conversation_text.py::test_a_legacy_turn_without_msg_tags_keeps_only_peoples_words.
def legacy_prompt_utterances(text: str, requester: str) -> list[tuple[str, str]]:
    """Return the ``(sender, body)`` messages people wrote in a prompt stored before prompts tagged each message."""
    current_heading = _CURRENT_MESSAGE.search(text)
    if current_heading is None:
        if text.startswith((*_NEW_MESSAGES_HEADERS, _THREAD_COPY_HEADER)):
            return []
        earlier: list[tuple[str, str]] = []
        current = _current_message(text)
    else:
        earlier = _earlier_messages(text[: current_heading.start()])
        current = _current_message(text[current_heading.end() :])
    return [*earlier, (requester, current)] if current else earlier


def _earlier_messages(section: str) -> list[tuple[str, str]]:
    """Return the senders' words under a heading of new messages; a thread copy or unknown heading has none."""
    heading, _, lines = section.partition("\n")
    if heading not in _NEW_MESSAGES_HEADERS:
        return []
    messages: list[tuple[str | None, list[str]]] = []
    for line in lines.splitlines():
        if sender_line := _SENDER_LINE.match(line):
            messages.append((sender_line.group(1), [_TIME_PREFIX.sub("", sender_line.group(2), count=1)]))
        elif _OWN_DRAFT_LINE.match(line):
            messages.append((None, []))
        elif messages:
            messages[-1][1].append(line)
    return [(sender, body) for sender, body_lines in messages if sender and (body := "\n".join(body_lines).strip())]


def _current_message(text: str) -> str:
    """Return what the requester typed in a legacy current message, or nothing when it cannot be told apart."""
    while trailing := _TRAILING_BLOCK.search(text):
        text = text[: trailing.start()]
    memory_lists = list(_MEMORY_LIST_HEADER.finditer(text))
    if memory_lists:
        _, blank_line, text = text[memory_lists[-1].end() :].partition("\n\n")
        if not blank_line:
            return ""
    if _MEMORY_ENTRYPOINT in text:
        return ""
    return _TIME_PREFIX.sub("", text.strip(), count=1).strip()
