"""Operator routes that list, merge, and delete entities in a shared agent's knowledge graph."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
import yaml

from mindroom import constants
from mindroom.api import config_lifecycle, main
from mindroom.api import knowledge_graph as knowledge_graph_api
from mindroom.knowledge_graph.maintenance import (
    DeleteReport,
    EntityCount,
    MergeInputError,
    MergeReport,
    _MergedEntity,
)
from mindroom.knowledge_graph.scope import graph_scope
from mindroom.knowledge_graph.store import set_knowledge_graph_store

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator
    from pathlib import Path

    from fastapi.testclient import TestClient

    from mindroom.knowledge_graph.scope import GraphScope

OPERATOR = {"Authorization": "Bearer operator-key"}
ROUTES = [
    ("GET", "/api/knowledge-graph/mind/entities", None),
    ("POST", "/api/knowledge-graph/mind/merge", {"keep": "Alice Example", "duplicates": ["Alice"]}),
    ("POST", "/api/knowledge-graph/mind/delete", {"junk": True}),
    # The key is checked before the body, so an invalid body tells an unauthenticated caller nothing.
    ("POST", "/api/knowledge-graph/mind/merge", {}),
]


class _Store:
    """Runs each operation with a stand-in driver and records the scope it was given."""

    def __init__(self) -> None:
        self.scopes: list[GraphScope] = []

    async def maintain(self, scope: GraphScope, operation: Callable[[object], Awaitable[Any]]) -> Any:  # noqa: ANN401
        self.scopes.append(scope)
        return await operation("driver")


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """A config with a shared graph agent, a private graph agent, and an agent without the graph."""
    config_dir = tmp_path / "api-runtime"
    config_dir.mkdir()
    path = config_dir / "config.yaml"
    path.write_text(
        yaml.dump(
            {
                "models": {"default": {"provider": "ollama", "id": "test-model"}},
                "knowledge_graph": {"model": "default"},
                "agents": {
                    "mind": {"display_name": "Mind", "knowledge_graph": True},
                    "pal": {"display_name": "Pal", "knowledge_graph": True, "private": {"per": "user"}},
                    "plain": {"display_name": "Plain"},
                },
            },
        ),
        encoding="utf-8",
    )
    return path


def _load(test_client: TestClient, config_path: Path, env: dict[str, str]) -> None:
    paths = constants.resolve_primary_runtime_paths(
        config_path=config_path,
        process_env={"MATRIX_HOMESERVER": "http://localhost:8008", **env},
    )
    main.initialize_api_app(test_client.app, paths)
    assert config_lifecycle.load_config_into_app(paths, test_client.app)


@pytest.fixture(autouse=True)
def runtime(test_client: TestClient, temp_config_file: Path) -> Iterator[None]:
    """Give the operator a configured key, and leave no store registered afterwards."""
    _load(test_client, temp_config_file, {"MINDROOM_API_KEY": "operator-key"})
    yield
    set_knowledge_graph_store(None)


@pytest.fixture
def store() -> _Store:
    """A running graph store."""
    fake = _Store()
    set_knowledge_graph_store(fake)
    return fake


@pytest.mark.parametrize(("method", "path", "body"), ROUTES)
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}])
def test_operator_key_required(
    test_client: TestClient,
    store: _Store,
    method: str,
    path: str,
    body: dict[str, object] | None,
    headers: dict[str, str],
) -> None:
    """Merging and deleting rewrite a graph every requester of the agent reads, so only the operator may."""
    response = test_client.request(method, path, json=body, headers=headers)

    assert response.status_code == 401
    assert store.scopes == []


@pytest.mark.parametrize(("method", "path", "body"), ROUTES)
def test_routes_are_closed_without_a_configured_operator_key(
    test_client: TestClient,
    temp_config_file: Path,
    store: _Store,
    method: str,
    path: str,
    body: dict[str, object] | None,
) -> None:
    """A runtime without MINDROOM_API_KEY has no operator, so the routes are unavailable."""
    _load(test_client, temp_config_file, {})

    response = test_client.request(method, path, json=body, headers=OPERATOR)

    assert response.status_code == 503
    assert store.scopes == []


def test_merge_dry_run_runs_through_the_store(
    test_client: TestClient,
    store: _Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A merge without `apply` is a dry run in the agent's graph, and the report comes back whole."""
    report = MergeReport(
        keep=EntityCount(name="Alice Example", uuid="u1", facts=3, mentions=2),
        merged=[_MergedEntity("Alice", "u2", 1, 1, 1, 0)],
        facts_moved=1,
        self_loops_dropped=1,
        unmatched=["Alcie"],
        applied=False,
    )
    merge = AsyncMock(return_value=report)
    monkeypatch.setattr(knowledge_graph_api, "merge_entities", merge)

    response = test_client.post(
        "/api/knowledge-graph/mind/merge",
        json={"keep": "Alice Example", "duplicates": ["Alice", "Alcie"]},
        headers=OPERATOR,
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "keep": {"name": "Alice Example", "uuid": "u1", "facts": 3, "mentions": 2},
        "merged": [
            {
                "name": "Alice",
                "uuid": "u2",
                "facts_moved": 1,
                "self_loops_dropped": 1,
                "mentions_moved": 1,
                "mentions_dropped": 0,
            },
        ],
        "facts_moved": 1,
        "self_loops_dropped": 1,
        "unmatched": ["Alcie"],
        "applied": False,
    }
    merge.assert_awaited_once_with("driver", "Alice Example", ["Alice", "Alcie"], apply=False)
    config, runtime_paths = config_lifecycle.read_app_committed_runtime_config(test_client.app)
    assert store.scopes == [graph_scope("mind", config, runtime_paths, None)]


def test_entities_and_delete_reach_the_operations(
    test_client: TestClient,
    store: _Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Listing passes the query and limit; deleting passes the names, the junk flag, and `apply`."""
    listed = AsyncMock(return_value=[EntityCount(name="Alice", uuid="u2", facts=4, mentions=1)])
    deleted = AsyncMock(return_value=DeleteReport(deleted=[], unmatched=["Bob"], applied=True))
    monkeypatch.setattr(knowledge_graph_api, "list_entities", listed)
    monkeypatch.setattr(knowledge_graph_api, "delete_entities", deleted)

    entities = test_client.get("/api/knowledge-graph/mind/entities?query=ali&limit=5", headers=OPERATOR)
    delete = test_client.post(
        "/api/knowledge-graph/mind/delete",
        json={"names": ["Bob"], "junk": True, "apply": True},
        headers=OPERATOR,
    )

    assert entities.json() == [{"name": "Alice", "uuid": "u2", "facts": 4, "mentions": 1}]
    listed.assert_awaited_once_with("driver", "ali", 5)
    assert delete.json() == {"deleted": [], "unmatched": ["Bob"], "applied": True}
    deleted.assert_awaited_once_with("driver", ["Bob"], junk=True, apply=True)
    assert [scope.agent_name for scope in store.scopes] == ["mind", "mind"]


@pytest.mark.parametrize(
    ("agent", "status"),
    [("nobody", 404), ("plain", 404), ("pal", 400)],
)
def test_only_a_shared_graph_agent_can_be_maintained(
    test_client: TestClient,
    store: _Store,
    agent: str,
    status: int,
) -> None:
    """An unknown agent or one without the graph has no graph; a private agent has one per requester."""
    response = test_client.get(f"/api/knowledge-graph/{agent}/entities", headers=OPERATOR)

    assert response.status_code == status
    assert store.scopes == []


def test_a_store_that_fails_is_unavailable(test_client: TestClient) -> None:
    """A store closing mid-request, or a database that cannot start, answers 503 with no internals."""

    class _ClosingStore:
        async def maintain(self, _scope: GraphScope, _operation: object) -> None:
            msg = "The work was cancelled on the other event loop"
            raise RuntimeError(msg)

    set_knowledge_graph_store(_ClosingStore())

    response = test_client.get("/api/knowledge-graph/mind/entities", headers=OPERATOR)

    assert response.status_code == 503
    assert response.json()["detail"] == "The knowledge graph is unavailable; see the MindRoom log"


def test_a_stopped_store_is_unavailable(test_client: TestClient) -> None:
    """With no graph store running there is nothing to maintain."""
    response = test_client.get("/api/knowledge-graph/mind/entities", headers=OPERATOR)

    assert response.status_code == 503


@pytest.mark.usefixtures("store")
def test_a_rejected_merge_is_a_conflict(test_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing entity to keep, or one also listed as a duplicate, is reported with the reason."""
    monkeypatch.setattr(
        knowledge_graph_api,
        "merge_entities",
        AsyncMock(side_effect=MergeInputError("No entity is named 'Alice N.'")),
    )

    response = test_client.post(
        "/api/knowledge-graph/mind/merge",
        json={"keep": "Alice N.", "duplicates": ["Alice"], "apply": True},
        headers=OPERATOR,
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "No entity is named 'Alice N.'"
