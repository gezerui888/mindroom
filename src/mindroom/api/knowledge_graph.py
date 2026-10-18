"""Operator routes that list, merge, and delete entities in a shared agent's knowledge graph.

The operations run inside the runtime, on the graph store's own loop while no episode is being added, because only
the runtime may talk to the embedded database. Every route requires the operator key, since a shared agent's graph
answers all of its requesters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from mindroom.api import config_lifecycle
from mindroom.api.auth import require_operator_key
from mindroom.knowledge_graph.maintenance import (
    DeleteReport,
    EntityCount,
    MergeInputError,
    MergeReport,
    delete_entities,
    list_entities,
    merge_entities,
)
from mindroom.knowledge_graph.scope import graph_scope
from mindroom.knowledge_graph.store import knowledge_graph_store
from mindroom.logging_config import get_logger

if TYPE_CHECKING:
    from mindroom.knowledge_graph.scope import GraphScope
    from mindroom.knowledge_graph.store import GraphStore

logger = get_logger(__name__)


def _require_operator(request: Request, authorization: Annotated[str | None, Header()] = None) -> None:
    require_operator_key(request, authorization)


# The key is checked before the request body is validated.
router = APIRouter(prefix="/api/knowledge-graph", tags=["knowledge-graph"], dependencies=[Depends(_require_operator)])


class _MergeRequest(BaseModel):
    keep: str
    duplicates: list[str]
    apply: bool = False


class _DeleteRequest(BaseModel):
    names: list[str] = Field(default_factory=list)
    junk: bool = False
    apply: bool = False


def _shared_graph(request: Request, agent: str) -> tuple[GraphStore, GraphScope]:
    """Return the running store and the graph of a shared agent that enables the knowledge graph."""
    config, runtime_paths = config_lifecycle.read_committed_runtime_config(request)
    if agent not in config.agents or not config.agent_uses_knowledge_graph(agent):
        raise HTTPException(status_code=404, detail=f"No agent named {agent!r} has the knowledge graph enabled")
    if config.get_agent(agent).private is not None:
        raise HTTPException(
            status_code=400,
            detail=f"{agent!r} is a private agent, with one graph per requester; only shared agents can be maintained",
        )
    store = knowledge_graph_store()
    if store is None:
        raise HTTPException(status_code=503, detail="The knowledge graph is not running")
    return store, graph_scope(agent, config, runtime_paths, None)


@router.get("/{agent}/entities")
async def entities(
    request: Request,
    agent: str,
    query: str = "",
    limit: Annotated[int, Query(ge=1, le=1000)] = 50,
) -> list[EntityCount]:
    """List the agent's entities whose name contains ``query``, ignoring case, most facts first."""
    store, scope = _shared_graph(request, agent)
    return await store.maintain(scope, lambda driver: list_entities(driver, query, limit))


@router.post("/{agent}/merge")
async def merge(request: Request, agent: str, body: _MergeRequest) -> MergeReport:
    """Merge the duplicates into the entity to keep, or report what that would do."""
    store, scope = _shared_graph(request, agent)
    try:
        report = await store.maintain(
            scope,
            lambda driver: merge_entities(driver, body.keep, body.duplicates, apply=body.apply),
        )
    except MergeInputError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if report.applied:
        logger.info(
            "Merged knowledge graph entities",
            agent=agent,
            keep=report.keep.name,
            merged=[entity.name for entity in report.merged],
            facts_moved=report.facts_moved,
        )
    return report


@router.post("/{agent}/delete")
async def delete(request: Request, agent: str, body: _DeleteRequest) -> DeleteReport:
    """Delete the named and, with ``junk``, junk-looking entities with their facts, or report what would go."""
    store, scope = _shared_graph(request, agent)
    report = await store.maintain(
        scope,
        lambda driver: delete_entities(driver, body.names, junk=body.junk, apply=body.apply),
    )
    if report.applied:
        logger.info("Deleted knowledge graph entities", agent=agent, deleted=len(report.deleted))
    return report
