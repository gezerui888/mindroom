"""Tests for finding a Matrix user ID in text the way the homeserver's in-place rename does."""

from __future__ import annotations

import pytest

from mindroom.agent_rename.matching import contains_user_id, replace_user_id

USER_ID = "@a:example.org"


@pytest.mark.parametrize(
    "text",
    [
        "@a:example.org",
        '"@a:example.org"',
        "x@a:example.org\N{LATIN SMALL LETTER E WITH ACUTE}y",
        "ask @a:example.org.",
        "@a:example.org. Then",
        "@a:example.org: hi",
        "@a:example.org:",
        "@a:example.org- x",
        "@a:example.org...",
        '"@a:example.org:cA6GgYzlTQ"',
    ],
)
def test_matches_when_nothing_continues_the_server_name(text: str) -> None:
    """Trailing punctuation and a ``:<device_id>`` suffix end the ID."""
    assert contains_user_id(text, USER_ID)


@pytest.mark.parametrize(
    "text",
    [
        "@a:example.org.uk",
        "@a:example.org.uk.",
        "@a:example.org:8448",
        "@a:example.org-x",
        "@a:example.org-gateway:x",
        "@a:example.orgx",
        "@ab:example.org",
        "",
    ],
)
def test_no_match_when_the_server_name_continues(text: str) -> None:
    """A letter or digit, a ``.`` or ``-`` before one, or a ``:`` before a port digit extends the server name."""
    assert not contains_user_id(text, USER_ID)


def test_replaces_every_bounded_occurrence_when_lengths_differ() -> None:
    """Only bounded occurrences change, and the replacement is taken literally."""
    text = "@a:example.org|@a:example.org.uk|@a:example.org:8448|@a:example.org"

    assert replace_user_id(text, USER_ID, r"@b\1:example.org") == (
        r"@b\1:example.org|@a:example.org.uk|@a:example.org:8448|@b\1:example.org"
    )


def test_empty_user_id_matches_nothing() -> None:
    """An empty ID would otherwise match between every pair of characters."""
    assert not contains_user_id("anything", "")
    assert replace_user_id("anything", "", "@b:example.org") == "anything"
