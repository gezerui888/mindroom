"""Read Matrix turns that older releases stored in an earlier shape."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mindroom.constants import MATRIX_EVENT_ID_METADATA_KEY

if TYPE_CHECKING:
    from agno.run.agent import RunOutput

# A Matrix thread session ID is the room ID and the thread root event ID joined by `:`.
_THREAD_SEPARATOR = ":$"


# LEGACY_COMPAT: Run metadata without room_id.
# Legacy format: Matrix turns stored their event IDs (`matrix_event_id`, `matrix_seen_event_ids`) in run metadata but
# no `room_id` or `thread_id`; the rule selects a run whose metadata holds `matrix_event_id` and no `room_id` and whose
# session ID is a Matrix room ID (`!...`), optionally followed by `:` and the thread root event ID.
# Last legacy release: v2026.5.26; replacement: v2026.5.27 (#665) stores `room_id` and `thread_id` in run metadata.
# Handling: The room and thread root are read from the session ID (split at the first `:$`; none means a room-mode
# session without a thread) and written into the loaded run's metadata, so background readers group, filter, and
# describe the turn by its real room. Stored rows are not rewritten.
# Coverage: tests/test_legacy_stored_runs.py::test_recovers_room_and_thread_from_the_session_id,
# tests/test_legacy_stored_runs.py::test_room_mode_session_has_no_thread,
# tests/test_legacy_stored_runs.py::test_leaves_other_runs_alone, and
# tests/test_history_session_runs.py::test_session_runs_returns_legacy_runs_with_their_room_and_thread.
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
