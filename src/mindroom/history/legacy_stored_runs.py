"""Read Matrix turns that older releases stored in an earlier shape."""

from __future__ import annotations

import re
from datetime import datetime
from typing import TYPE_CHECKING

from mindroom.constants import MATRIX_EVENT_ID_METADATA_KEY
from mindroom.prompt_message_tags import has_msg_tags, parse_msg_tags

if TYPE_CHECKING:
    from agno.run.agent import RunOutput
    from agno.run.team import TeamRunOutput

# A Matrix thread session ID is the room ID and the thread root event ID joined by `:`.
_THREAD_SEPARATOR = ":$"
# Set on a recovered run's loaded metadata only, never stored: readers learn its prompts may use an earlier layout too.
_RECOVERED_KEY = "_mindroom_room_id_from_session_id"

# Headings of the earlier messages a legacy prompt put before the current message: new ones, or a copy of the thread.
_NEW_MESSAGES_HEADERS = (
    "Messages from other participants since your last response:",
    "Messages since your last response:",
)
_HISTORY_HEADINGS = (*_NEW_MESSAGES_HEADERS, "Previous conversation in this thread:", "Thread Context:")
_CURRENT_MESSAGE = re.compile(r"(?:^|\n\n)Current message:\n")
_SENDER_LINE = re.compile(r"(@[^\s:]+:\S+?): (.*)")
_OWN_DRAFT_LINE = re.compile(r"You \([^)]*\): ")
_TIME_PREFIX = re.compile(r"^\[(?:\d{4}-\d{2}-\d{2} )?\d{2}:\d{2} [^\]]+\](?:\s+|$)")
# Paragraphs, separated by blank lines, that MindRoom added to stored messages.
_MEMORY_FILE = "[File memory entrypoint"
_MEMORY_LIST = (
    r"\[Automatically extracted [^\n]* memories - may not be relevant to current context\]\n"
    r"Previous [^\n]* memories that might be related:(?:\n[^\n]*)*"
)
_MEMORY_LIST_BLOCK = re.compile(_MEMORY_LIST)
_MEMORY_SEARCH_NOTICE = (
    r"Semantic memory search is unavailable this turn \([^\n]*\); stored memories may be missing or keyword-only\. "
    r"Do not claim to have checked stored memories\."
)
_ATTACHMENTS = r"Available attachment IDs: [^\n]*\. Use tool calls to inspect or process them\."
_MATRIX_METADATA = (
    r"\[Matrix metadata for tool calls\]\nroom_id: [^\n]*\nthread_id: [^\n]*\nreply_to_event_id: [^\n]*\n"
    r"Use these IDs when calling matrix_message\."
)
_INLINE_MEDIA = (
    r"\[Inline media unavailable for this model\][ \n]The model rejected inline attachments for this turn\. "
    r"Use available attachment IDs and tools to inspect files instead\."
)
_ADDED_BLOCK = re.compile(f"{_MEMORY_LIST}|{_MEMORY_SEARCH_NOTICE}|{_ATTACHMENTS}|{_MATRIX_METADATA}|{_INLINE_MEDIA}")
# Each block, with the commit time of the first release that no longer stored it in messages.
_STORED_BLOCKS = tuple(
    (re.compile(pattern), datetime.fromisoformat(stopped).timestamp())
    for pattern, stopped in (
        (_MEMORY_LIST, "2026-07-19T20:29:11-07:00"),  # v2026.7.214
        (_MEMORY_SEARCH_NOTICE, "2026-07-19T20:29:11-07:00"),  # v2026.7.214
        (_ATTACHMENTS, "2026-06-11T10:23:51-07:00"),  # v2026.6.83
        (_MATRIX_METADATA, "2026-07-20T15:18:33-07:00"),  # v2026.7.221
        (_INLINE_MEDIA, "2026-08-24T13:25:21+02:00"),  # v2026.8.110
    )
)


# LEGACY_COMPAT: Run metadata without room_id.
# Legacy format: Matrix turns stored their event IDs (`matrix_event_id`, `matrix_seen_event_ids`) in run metadata but
# no `room_id` or `thread_id`; the rule selects a run whose metadata holds `matrix_event_id` and no `room_id` and
# whose session ID is a Matrix room ID (`!...`), optionally followed by `:` and the thread root event ID.
# Last legacy release: v2026.5.26; replacement: v2026.5.27 (#665) stores `room_id` and `thread_id` in run metadata.
# A turn that failed before its run started was stored from the turn recorder's seed metadata, which kept no room
# through v2026.10.344; the unreleased writer seeds `room_id` and `thread_id` there too.
# Handling: The room and thread root are read from the session ID (split at the first `:$`; none means a room-mode
# session without a thread) and written into the loaded run's metadata, so background readers group, filter, and
# describe the turn by its real room, and marked recovered for the prompt rule below. Stored rows are not rewritten.
# Coverage: tests/test_legacy_stored_runs.py::test_recovers_room_and_thread_from_the_session_id,
# tests/test_legacy_stored_runs.py::test_room_mode_session_has_no_thread,
# tests/test_legacy_stored_runs.py::test_leaves_other_runs_alone,
# tests/test_history_session_runs.py::test_session_runs_returns_legacy_runs_with_their_room_and_thread, and
# tests/test_history_session_runs.py::test_legacy_runs_are_checked_for_redactions_in_their_room.
def recover_legacy_run_metadata(run: RunOutput) -> None:
    """Give a turn stored before run metadata named its room the room and thread its session ID was built from."""
    metadata = run.metadata
    session_id = run.session_id or ""
    if (
        not metadata
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
    """Return whether the rule above gave the run its room, as it does for runs stored before metadata named it.

    The mark lives only in the loaded run's metadata; stored runs never hold it.
    """
    return (run.metadata or {}).get(_RECOVERED_KEY) is True


# LEGACY_COMPAT: MindRoom blocks stored inside Matrix messages.
# Legacy format: Paragraphs MindRoom appended to the current message and stored with it, inside the requester's
# `<msg>` tag from v2026.4.175: a retrieved-memory list (`[Automatically extracted ... memories ...]` then
# `Previous ... memories that might be related:` and its lines), the semantic memory search notice (from v2026.7.141),
# the `Available attachment IDs:` line, the `[Matrix metadata for tool calls]` block, and the
# `[Inline media unavailable for this model]` note (followed by a space before v2026.5.134 and a line break after).
# The rule selects each user message body of a stored turn created before the release that stopped writing a block.
# Last legacy release: the memory list and search notice v2026.7.213, replaced by v2026.7.214 (#1596, cut
# 2026-07-19T20:29:11-07:00), which keeps retrieved memory out of stored turns; the attachment line v2026.6.82,
# replaced by v2026.6.83 (#1214, cut 2026-06-11T10:23:51-07:00), which places attachments by time; the Matrix
# metadata block v2026.7.220, replaced by v2026.7.221 (#1599, cut 2026-07-20T15:18:33-07:00), which moves it into
# structured context; the inline media note v2026.8.109, replaced by v2026.8.110 (#1890, cut
# 2026-08-24T13:25:21+02:00), which adds it at the provider call only. The cut times are the tags' commit times.
# Handling: In a turn whose `created_at` is before a block's cut time, a paragraph that matches that block exactly is
# removed, so turns of those releases credit only the person's words. A turn created later, or without a creation
# time, is read byte for byte, so a person who pastes one of these blocks today keeps it.
# Coverage: tests/test_legacy_stored_runs.py::test_blocks_are_dropped_only_from_turns_stored_while_a_release_wrote_them,
# tests/test_legacy_stored_runs.py::test_a_current_message_is_read_byte_for_byte,
# tests/test_conversation_text.py::test_blocks_mindroom_added_inside_a_tag_are_nobodys_words, and
# tests/test_conversation_text.py::test_a_current_message_quoting_an_old_block_is_read_as_written.
def without_added_blocks(body: str, created_at: int | None) -> str:
    """Return a stored message body without the paragraphs MindRoom added when its turn was created, else unchanged."""
    if not created_at:
        return body
    written = [pattern for pattern, stopped in _STORED_BLOCKS if created_at < stopped]
    if not written:
        return body
    paragraphs = body.split("\n\n")
    kept = [paragraph for paragraph in paragraphs if not any(p.fullmatch(paragraph.strip()) for p in written)]
    return body if len(kept) == len(paragraphs) else "\n\n".join(kept).strip()


# LEGACY_COMPAT: Matrix prompt text without <msg> tags, and MindRoom blocks inside them.
# Legacy format: A user message held the prompt MindRoom built as plain text: optionally a heading
# (`Messages from other participants since your last response:`, `Messages since your last response:`, or
# `Previous conversation in this thread:`; a team's `Thread Context:`) over `@sender:server: body` lines, then
# `Current message:`. MindRoom added
# blocks, each separated by a blank line: memory files (`[File memory entrypoint (...)]`) and memory lists
# (`[Automatically extracted ... memories ...]` over `Previous ... memories that might be related:`) before the person's
# text, then `Available attachment IDs:` lines, a `[Matrix metadata for tool calls]` block, and, from v2026.3.14, an
# `[Inline media unavailable for this model]` note. The rule selects every user message of a run recovered above.
# Last legacy release: v2026.4.174 for untagged prompts; replacement: v2026.4.175 (#645) wraps earlier messages, and
# the current message when its sender is known, in `<msg from="..."><![CDATA[...]]></msg>` tags. From v2026.4.175
# through v2026.4.198 the requester's tag held the whole prompt with its memory file and list; v2026.4.199 (#576)
# moved the memory file to the system context and put the memory list after the person's text. Runs through
# v2026.5.26 are recovered, so their tagged and untagged messages both take this rule.
# Handling: Every message body loses the blocks above; a memory file runs until the next memory list, and a body
# whose memory file has no list after it is nothing. The time prefix is stripped and an empty body yields nothing.
# Tagged messages keep their senders when the sender is a Matrix ID; `You (...)` drafts are dropped, while releases
# before v2026.3.122 (#404) listed the agent's own drafts under its Matrix ID, so those read as its words. Untagged, new
# earlier messages under their heading are their senders' (continuation lines joined; `You (...)` drafts, notes
# under the heading, thread copies, and unknown headings dropped) and the text after the first `Current message:` is
# the requester's. Without that heading, the cleaned text is the requester's unless it holds a history heading or an
# `@sender:server: ` line, which leaves no way to tell earlier messages from the current one, so nothing is kept.
# Coverage: tests/test_legacy_stored_runs.py::test_legacy_prompt_keeps_only_peoples_words,
# tests/test_legacy_stored_runs.py::test_previous_conversation_copies_are_skipped,
# tests/test_legacy_stored_runs.py::test_interrupted_draft_notes_are_dropped,
# tests/test_legacy_stored_runs.py::test_memory_entrypoint_without_bullet_list_drops_the_current_message,
# tests/test_legacy_stored_runs.py::test_a_memory_entrypoint_after_the_last_memory_list_drops_the_current_message,
# tests/test_legacy_stored_runs.py::test_plain_legacy_message_is_the_requesters,
# tests/test_legacy_stored_runs.py::test_a_bare_time_prefix_is_nobodys_words,
# tests/test_legacy_stored_runs.py::test_attachment_metadata_and_inline_media_blocks_are_dropped,
# tests/test_legacy_stored_runs.py::test_unknown_sections_are_dropped,
# tests/test_legacy_stored_runs.py::test_without_a_current_message_heading_only_plain_text_is_kept,
# tests/test_legacy_stored_runs.py::test_tagged_legacy_messages_lose_mindroom_blocks_and_drafts, and
# tests/test_conversation_text.py::test_a_legacy_turn_keeps_only_peoples_words.
def legacy_prompt_utterances(text: str, requester: str) -> list[tuple[str, str]]:
    """Return the ``(sender, body)`` messages people wrote in a user message of a run stored in a legacy layout."""
    if has_msg_tags(text):
        return [
            (sender, body)
            for sender, tagged in parse_msg_tags(text, default_sender=requester)
            if sender.startswith("@") and (body := _person_text(tagged))
        ]
    # The first heading is the legacy writer's; one pasted into an earlier message would be read here (a known limit).
    current_heading = _CURRENT_MESSAGE.search(text)
    if current_heading is not None:
        earlier = _earlier_messages(text[: current_heading.start()])
        current = _person_text(text[current_heading.end() :])
    else:
        earlier, current = [], _person_text(text)
        if any(line in _HISTORY_HEADINGS or _SENDER_LINE.match(line) for line in current.splitlines()):
            return []
    return [*earlier, (requester, current)] if current else earlier


def _earlier_messages(section: str) -> list[tuple[str, str]]:
    """Return the senders' words under a heading of new messages; a thread copy or unknown heading has none."""
    heading, _, lines = section.partition("\n")
    if heading not in _NEW_MESSAGES_HEADERS:
        return []
    messages: list[tuple[str | None, list[str]]] = []
    for line in lines.splitlines():
        if sender_line := _SENDER_LINE.match(line):
            messages.append((sender_line.group(1), [sender_line.group(2)]))
        elif _OWN_DRAFT_LINE.match(line):
            messages.append((None, []))
        elif messages:
            messages[-1][1].append(line)
    return [
        (sender, body) for sender, body_lines in messages if sender and (body := _person_text("\n".join(body_lines)))
    ]


def _person_text(text: str) -> str:
    """Return a legacy message body without the blocks MindRoom added, or nothing when a memory file never ends."""
    kept: list[str] = []
    in_memory_file = False
    for block in text.split("\n\n"):
        stripped = block.strip()
        if stripped.startswith(_MEMORY_FILE):
            in_memory_file = True
        elif _MEMORY_LIST_BLOCK.fullmatch(stripped):
            in_memory_file = False
        elif not in_memory_file and stripped and not _ADDED_BLOCK.fullmatch(stripped):
            kept.append(block)
    if in_memory_file:
        return ""
    return _TIME_PREFIX.sub("", "\n\n".join(kept).strip(), count=1).strip()
