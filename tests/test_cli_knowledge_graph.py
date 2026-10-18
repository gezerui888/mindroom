"""The knowledge-graph CLI sends operator requests to the running MindRoom and prints their reports."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest
from typer.testing import CliRunner

from mindroom.cli.main import app

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()

_DRY_RUN = "Dry run: nothing changed. Re-run with --apply."
_MERGE_REPORT = {
    "keep": {"name": "Alice Example", "uuid": "u1", "facts": 3, "mentions": 2},
    "merged": [
        {
            "name": "Alice",
            "uuid": "u2",
            "facts_moved": 4,
            "self_loops_dropped": 1,
            "mentions_moved": 2,
            "mentions_dropped": 1,
        },
    ],
    "facts_moved": 4,
    "self_loops_dropped": 1,
    "unmatched": ["Alcie"],
    "applied": False,
}


class _Api:
    """Records each request and answers with the queued responses."""

    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []
        self.responses: list[httpx.Response] = []

    def get(self, url: str, **kwargs: object) -> httpx.Response:
        self.sent.append({"method": "GET", "url": url, **kwargs})
        return self.responses.pop(0)

    def post(self, url: str, **kwargs: object) -> httpx.Response:
        self.sent.append({"method": "POST", "url": url, **kwargs})
        return self.responses.pop(0)


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> _Api:
    """The running MindRoom's API, reached with the operator key."""
    monkeypatch.setenv("MINDROOM_API_KEY", "operator-key")
    fake = _Api()
    monkeypatch.setattr(httpx, "get", fake.get)
    monkeypatch.setattr(httpx, "post", fake.post)
    return fake


def test_merge_dry_run_posts_to_the_route_and_says_nothing_changed(
    api: _Api,
    tmp_path: Path,
) -> None:
    """Without --apply the CLI asks for a dry run, prints what would move, and ends with the dry-run line."""
    api.responses.append(httpx.Response(200, json=_MERGE_REPORT))

    result = runner.invoke(
        app,
        ["knowledge-graph", "merge", "mind", "Alice Example", "Alice", "Alcie", "--config", str(tmp_path / "c.yaml")],
    )

    assert result.exit_code == 0, result.output
    [sent] = api.sent
    assert sent["method"] == "POST"
    assert str(sent["url"]).endswith("/api/knowledge-graph/mind/merge")
    assert sent["headers"] == {"Authorization": "Bearer operator-key"}
    assert sent["json"] == {"keep": "Alice Example", "duplicates": ["Alice", "Alcie"], "apply": False}
    assert sent["follow_redirects"] is False
    assert "Alice Example" in result.stdout
    assert "Alice: 4 facts move, 1 self-loops dropped, 2 mentions move, 1 dropped" in result.stdout
    assert "No entity is named: Alcie" in result.stdout
    assert result.stdout.rstrip().endswith(_DRY_RUN)


def test_applied_delete_sends_apply_and_drops_the_dry_run_line(
    api: _Api,
    tmp_path: Path,
) -> None:
    """With --apply the request applies the change and the output says it was done."""
    report = {
        "deleted": [{"name": "compaction.py", "uuid": "u3", "facts": 2, "mentions": 1}],
        "unmatched": [],
        "applied": True,
    }
    api.responses.append(httpx.Response(200, json=report))

    result = runner.invoke(
        app,
        ["knowledge-graph", "delete", "mind", "--junk", "--apply", "--config", str(tmp_path / "c.yaml")],
    )

    assert result.exit_code == 0, result.output
    [sent] = api.sent
    assert sent["json"] == {"names": [], "junk": True, "apply": True}
    assert "compaction.py (2 facts, 1 mentions)" in result.stdout
    assert _DRY_RUN not in result.stdout


def test_entities_lists_names_with_their_counts(api: _Api, tmp_path: Path) -> None:
    """The entity list shows each name with its facts and mentions, and passes the query and limit."""
    api.responses.append(
        httpx.Response(200, json=[{"name": "Alice Example", "uuid": "u1", "facts": 12, "mentions": 3}]),
    )

    result = runner.invoke(
        app,
        ["knowledge-graph", "entities", "mind", "alice", "--limit", "5", "--config", str(tmp_path / "c.yaml")],
    )

    assert result.exit_code == 0, result.output
    [sent] = api.sent
    assert (sent["method"], sent["params"]) == ("GET", {"query": "alice", "limit": 5})
    assert "Alice Example" in result.stdout
    assert "12 facts" in result.stdout


def test_delete_needs_names_or_junk(api: _Api, tmp_path: Path) -> None:
    """A delete that names nothing and is not a junk sweep would do nothing, so it is refused before any request."""
    result = runner.invoke(app, ["knowledge-graph", "delete", "mind", "--config", str(tmp_path / "c.yaml")])

    assert result.exit_code == 2
    assert api.sent == []


def test_a_refused_request_shows_the_reason(api: _Api, tmp_path: Path) -> None:
    """The route's error detail, such as a merge it rejects, reaches the operator."""
    api.responses.append(httpx.Response(409, json={"detail": "No entity is named 'Alice N.'"}))

    result = runner.invoke(
        app,
        ["knowledge-graph", "merge", "mind", "Alice N.", "Alice", "--config", str(tmp_path / "c.yaml")],
    )

    assert result.exit_code == 2
    assert "HTTP 409" in result.output
    assert "No entity is named 'Alice N.'" in result.output


def test_an_infinite_timeout_is_refused(api: _Api, tmp_path: Path) -> None:
    """`--timeout inf` passes the option's minimum but no HTTP client accepts it, so it fails before any request."""
    result = runner.invoke(
        app,
        ["knowledge-graph", "delete", "mind", "--junk", "--timeout", "inf", "--config", str(tmp_path / "c.yaml")],
    )

    assert result.exit_code == 2
    assert "--timeout must be finite." in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert api.sent == []
