"""Operator maintenance of one scope's graph: list entities, merge duplicates into one, and delete junk.

Graphiti has no merge operation, so these work in Cypher on the graph Graphiti writes. Each takes the driver for one
scope's graph; ``GraphitiStore.maintain`` runs them on the graph loop while no episode is being added. With
``apply=False`` they report what would change and write nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from graphiti_core.driver.driver import GraphDriver


@dataclass(frozen=True)
class EntityCount:
    """An entity with the number of facts that touch it and of episodes that mention it."""

    name: str
    uuid: str
    facts: int
    mentions: int


@dataclass(frozen=True)
class MergeReport:
    """The kept entity, the entities merged into it, and what happened to their facts, counted before any write."""

    keep: EntityCount
    merged: list[EntityCount]
    facts_moved: int
    # Facts between the merged entities, which would otherwise point from the kept entity to itself.
    self_loops_dropped: int
    applied: bool


@dataclass(frozen=True)
class DeleteReport:
    """The entities deleted with their facts, or that would be in a dry run."""

    deleted: list[EntityCount]
    applied: bool


class MergeInputError(ValueError):
    """The entity to keep does not exist or is also listed as a duplicate."""


# Appended to a clause that binds `entity`; returns each entity once, most facts first.
_COUNTED = """
WITH DISTINCT entity
OPTIONAL MATCH (entity)-[fact:RELATES_TO]-(:Entity)
WITH entity, count(DISTINCT fact) AS facts
OPTIONAL MATCH (:Episodic)-[mention:MENTIONS]->(entity)
RETURN entity.name AS name, entity.uuid AS uuid, facts, count(mention) AS mentions
ORDER BY facts DESC, name, uuid
"""
_CONTAINING = "MATCH (entity:Entity) WHERE toLower(entity.name) CONTAINS toLower($query)"
_NAMED = "UNWIND $names AS wanted MATCH (entity:Entity) WHERE toLower(entity.name) = toLower(wanted)"
_WITH_UUID = "MATCH (entity:Entity) WHERE entity.uuid IN $uuids"
# FalkorDB returns parallel edges between two nodes as one row unless the query uses the edge, so every query
# here that lists or counts facts or mentions reads the edge itself.
_FACTS_TOUCHING = """
MATCH (source:Entity)-[fact:RELATES_TO]->(target:Entity)
WHERE source.uuid IN $uuids OR target.uuid IN $uuids
RETURN source.uuid AS source, target.uuid AS target, fact.uuid AS uuid
"""
# Cypher cannot change a relationship's endpoints, so a fact moves by creating its copy and deleting it in one query.
_MOVE_FACT = """
MATCH (:Entity {uuid: $source})-[old:RELATES_TO {uuid: $uuid}]->(:Entity {uuid: $target})
MATCH (new_source:Entity {uuid: $new_source})
MATCH (new_target:Entity {uuid: $new_target})
CREATE (new_source)-[moved:RELATES_TO]->(new_target)
SET moved = $properties
"""
_MOVE_MENTIONS = """
MATCH (episode:Episodic)-[old:MENTIONS]->(duplicate:Entity)
WHERE duplicate.uuid IN $uuids
MATCH (keep:Entity {uuid: $keep})
MERGE (episode)-[moved:MENTIONS]->(keep)
ON CREATE SET moved = properties(old)
DELETE old
"""
# Also drops HAS_MEMBER edges from communities, which MindRoom never builds.
_DELETE = "MATCH (entity:Entity) WHERE entity.uuid IN $uuids DETACH DELETE entity"
# Graphiti's two writes of a fact store its endpoints under different keys.
_SOURCE_KEYS = ("source_uuid", "source_node_uuid")
_TARGET_KEYS = ("target_uuid", "target_node_uuid")

# Kinds of names the extraction instructions keep out of the graph, which earlier extraction made into entities.
_URL = re.compile(r"^(?:[a-z][a-z0-9+.-]*://|www\.)", re.IGNORECASE)
# Absolute, home, relative, or Windows paths, and anything with two slashes and no spaces before the second.
_PATH = re.compile(r"^(?:~|\.\.?)?/|^[a-z]:\\|^[^\s/]+/[^\s/]+/", re.IGNORECASE)
# Not `.js`, which ends product names such as Node.js.
_FILE_EXTENSIONS = (
    "py|pyi|ipynb|ts|tsx|jsx|mjs|rs|go|java|kt|swift|rb|php|lua|c|h|cpp|sh|bash|zsh|nix|sql|css|scss|html|vue"
    "|md|rst|txt|log|json|jsonl|yaml|yml|toml|ini|cfg|conf|env|lock|csv|tsv|xml|db|sqlite|sqlite3|parquet"
    "|pid|sock|bak|tmp|patch|diff|png|jpg|jpeg|gif|svg|webp|mp3|mp4|wav|zip|tar|gz|tgz"
)
_FILE = re.compile(rf"^\S*\.(?:{_FILE_EXTENSIONS})$", re.IGNORECASE)
_ENV_VAR = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")
# Commit hashes and digests are lowercase hex with letters and digits; UUIDs too.
_HASH = re.compile(r"^(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,64}$|^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$")
# Room IDs and event IDs; links to them are URLs. User IDs are aliases of a person and stay.
_MATRIX_ID = re.compile(r"^(?:![A-Za-z0-9]+:\S+|\$[\w+/-]{20,})$")
_JUNK = (_URL, _PATH, _FILE, _ENV_VAR, _HASH, _MATRIX_ID)


def looks_like_junk(name: str) -> bool:
    """Whether a name is a path, code or data file, URL, env var, hash, or Matrix room or event ID."""
    return any(pattern.search(name) for pattern in _JUNK)


async def _records(driver: GraphDriver, cypher: str, **params: object) -> list[dict[str, Any]]:
    records, _, _ = await driver.execute_query(cypher, **params)
    return records


async def _entities(driver: GraphDriver, cypher: str, **params: object) -> list[EntityCount]:
    return [
        EntityCount(name=record["name"], uuid=record["uuid"], facts=record["facts"], mentions=record["mentions"])
        for record in await _records(driver, cypher, **params)
    ]


async def list_entities(driver: GraphDriver, query: str, limit: int = 50) -> list[EntityCount]:
    """Return the entities whose name contains ``query``, ignoring case, most facts first."""
    return await _entities(driver, _CONTAINING + _COUNTED + "LIMIT $limit", query=query, limit=limit)


async def merge_entities(driver: GraphDriver, keep: str, duplicates: Sequence[str], *, apply: bool) -> MergeReport:
    """Merge the entities named in ``duplicates`` into the one named ``keep``; names match exactly, ignoring case.

    Among several entities named ``keep``, the one with most facts is kept and the others are merged into it. Each
    duplicate's facts move to the kept entity with every property, their uuid included, since episodes list the facts
    they support by uuid; facts among the merged entities are dropped, then the duplicate's mentions move and the
    duplicate is deleted. Each fact moves in its own query, so an interrupted merge loses no fact and running it
    again finishes it.
    """
    candidates = await _entities(driver, _NAMED + _COUNTED, names=[keep])
    if not candidates:
        msg = f"No entity is named {keep!r}"
        raise MergeInputError(msg)
    kept, *same_name = candidates
    named = await _entities(driver, _NAMED + _COUNTED, names=list(duplicates))
    if any(entity.uuid == kept.uuid for entity in named):
        msg = f"{keep!r} is both the entity to keep and a duplicate"
        raise MergeInputError(msg)
    merged = [*same_name, *named]
    merged_uuids = [entity.uuid for entity in merged]
    group = {kept.uuid, *merged_uuids}
    facts = await _records(
        driver,
        _FACTS_TOUCHING + (", properties(fact) AS properties" if apply else ""),
        uuids=merged_uuids,
    )
    moving = [fact for fact in facts if not (fact["source"] in group and fact["target"] in group)]
    if apply:
        for fact in moving:
            await _move_fact(driver, fact, group, kept.uuid)
        await driver.execute_query(_MOVE_MENTIONS, uuids=merged_uuids, keep=kept.uuid)
        await driver.execute_query(_DELETE, uuids=merged_uuids)
    return MergeReport(
        keep=kept,
        merged=merged,
        facts_moved=len(moving),
        self_loops_dropped=len(facts) - len(moving),
        applied=apply,
    )


async def _move_fact(driver: GraphDriver, fact: dict[str, Any], group: set[str], keep: str) -> None:
    """Recreate a fact with its endpoint in ``group`` replaced by ``keep``, then delete the original."""
    new_source = keep if fact["source"] in group else fact["source"]
    new_target = keep if fact["target"] in group else fact["target"]
    properties = dict(fact["properties"])
    embedding = properties.pop("fact_embedding", None)
    for key in _SOURCE_KEYS:
        if key in properties:
            properties[key] = new_source
    for key in _TARGET_KEYS:
        if key in properties:
            properties[key] = new_target
    # A property set from a parameter is a list; Graphiti's vector search needs the embedding stored as a vector.
    cypher = _MOVE_FACT + ("SET moved.fact_embedding = vecf32($embedding)\n" if embedding is not None else "")
    await driver.execute_query(
        cypher + "DELETE old",
        source=fact["source"],
        target=fact["target"],
        uuid=fact["uuid"],
        new_source=new_source,
        new_target=new_target,
        properties=properties,
        embedding=embedding,
    )


async def delete_entities(
    driver: GraphDriver,
    names: Sequence[str] = (),
    *,
    junk: bool = False,
    apply: bool,
) -> DeleteReport:
    """Delete the entities named in ``names`` (exactly, ignoring case) and, with ``junk``, every junk-looking one.

    An entity goes with its facts and mentions. Facts it shared with other entities are gone from those too.
    """
    deleted = await _entities(driver, _NAMED + _COUNTED, names=list(names))
    if junk:
        listed = {entity.uuid for entity in deleted}
        everything = await _records(driver, "MATCH (entity:Entity) RETURN entity.name AS name, entity.uuid AS uuid")
        uuids = [row["uuid"] for row in everything if row["uuid"] not in listed and looks_like_junk(row["name"])]
        deleted += await _entities(driver, _WITH_UUID + _COUNTED, uuids=uuids)
    if apply and deleted:
        await driver.execute_query(_DELETE, uuids=[entity.uuid for entity in deleted])
    return DeleteReport(deleted=deleted, applied=apply)
