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
class _MergedEntity:
    """A duplicate and what merging it moves to the kept entity or drops.

    A fact or mention shared by several duplicates is counted under the first of them in merge order, so the numbers
    add up to what the merge does.
    """

    name: str
    uuid: str
    facts_moved: int
    # Facts with the kept entity or another duplicate, which would otherwise point from the kept entity to itself.
    self_loops_dropped: int
    mentions_moved: int
    # Mentions from episodes that already mention the kept entity, or an earlier duplicate.
    mentions_dropped: int


@dataclass(frozen=True)
class MergeReport:
    """The kept entity, the duplicates merged into it, and the names that matched nothing, counted before any write."""

    keep: EntityCount
    merged: list[_MergedEntity]
    facts_moved: int
    self_loops_dropped: int
    unmatched: list[str]
    applied: bool


@dataclass(frozen=True)
class DeleteReport:
    """The entities deleted with their facts, or that would be in a dry run, and the names that matched nothing."""

    deleted: list[EntityCount]
    unmatched: list[str]
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
_UNMATCHED = """
UNWIND $names AS wanted
OPTIONAL MATCH (entity:Entity) WHERE toLower(entity.name) = toLower(wanted)
WITH wanted, count(entity) AS found
WHERE found = 0
RETURN wanted
"""
# FalkorDB returns parallel edges between two nodes as one row unless the query uses the edge, so every query
# here that lists or counts facts or mentions reads the edge itself.
# Named `fact`, not `e`: MindRoomFalkorDriver rewrites `properties(e)` to drop the embedding a moved fact needs.
_FACTS_MATCH = """
MATCH (source:Entity)-[fact:RELATES_TO]->(target:Entity)
WHERE source.uuid IN $uuids OR target.uuid IN $uuids
"""
_FACTS_TOUCHING = _FACTS_MATCH + "RETURN source.uuid AS source, target.uuid AS target, fact.uuid AS uuid"
_FACTS_TOUCHING_WITH_PROPERTIES = (
    _FACTS_MATCH
    + "RETURN source.uuid AS source, target.uuid AS target, fact.uuid AS uuid, properties(fact) AS properties"
)
_MENTIONS_OF = """
MATCH (episode:Episodic)-[mention:MENTIONS]->(entity:Entity)
WHERE entity.uuid IN $uuids
RETURN episode.uuid AS episode, entity.uuid AS entity, mention.uuid AS uuid
ORDER BY episode, uuid
"""
# Cypher cannot change a relationship's endpoints, so a fact moves by creating its copy and deleting it in one query.
# A property set from a parameter is a list; Graphiti's vector search needs the embedding stored as a vector, and
# `vecf32(NULL)` leaves a fact without an embedding as it was.
_MOVE_FACT = """
MATCH (:Entity {uuid: $source})-[old:RELATES_TO {uuid: $uuid}]->(:Entity {uuid: $target})
MATCH (new_source:Entity {uuid: $new_source})
MATCH (new_target:Entity {uuid: $new_target})
CREATE (new_source)-[moved:RELATES_TO]->(new_target)
SET moved = $properties
SET moved.fact_embedding = vecf32($embedding)
DELETE old
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

# Kinds of names the extraction instructions keep out of the graph, which earlier extraction made into entities. The
# patterns catch the common shapes, not every one; an operator reads the dry run's list before deleting.
_URL = re.compile(r"^(?:[a-z][a-z0-9+.-]*://|www\.)", re.IGNORECASE)
_ABSOLUTE_PATH = re.compile(r"^(?:~|\.\.?)?/|^[a-z]:\\", re.IGNORECASE)
# Lowercase segments with a letter (`src/mindroom`), or three or more segments of two or more characters
# (`Users/Ann/Desktop`); not `TCP/IP`, `A/B/C`, or `1/2/2026`.
_RELATIVE_PATH = re.compile(r"^(?=.*[a-z])[a-z0-9_.-]+(?:/[a-z0-9_.-]*)+$")
_DEEP_PATH = re.compile(r"^(?=.*[A-Za-z])[\w.-]{2,}(?:/[\w.-]{2,}){2,}/?$")
_FILE = re.compile(
    r"^(?P<stem>\S*)\.(?P<extension>py|pyi|ipynb|js|ts|tsx|jsx|mjs|rs|go|java|kt|swift|rb|php|lua|c|h|cpp|sh|bash|zsh"
    r"|nix|sql|css|scss|html|vue|md|rst|txt|log|json|jsonl|yaml|yml|toml|ini|cfg|conf|env|lock|csv|tsv|xml|db|sqlite"
    r"|sqlite3|parquet|pid|sock|bak|tmp|patch|diff|png|jpg|jpeg|gif|svg|webp|mp3|mp4|wav|zip|tar|gz|tgz)$",
    re.IGNORECASE,
)
# A few extensions also end product names (Node.js, Tailwind.css, Bun.sh) or domains (docs.rs); a capitalized name
# on any other extension (Readme.md, Setup.py) is still a file.
_PRODUCT_STEM = re.compile(r"^[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)*$")
_PRODUCT_EXTENSIONS = frozenset({"js", "css", "sh", "rs"})
_DOMAIN_STEM = re.compile(r"^[a-z]+$")
_DOMAIN_EXTENSIONS = frozenset({"rs", "sh"})
_ENV_VAR = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")
_HEX = re.compile(r"^[0-9a-f]{7,64}$")
_HEX_RUN = re.compile(r"\d+|[a-f]+")
_UUID = re.compile(r"^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$")
# Email addresses and Matrix user IDs are aliases of a person, to merge rather than delete, whatever their domain
# ends with (`ann@example.rs`, `@ann:example.sh`).
_IDENTITY = re.compile(r"^[^\s@/]+@[^\s@/]+\.[^\s@/]+$|^@[^\s:/]+:[^\s/]+$")
# Room IDs, event IDs, room aliases, and links to rooms.
_MATRIX = re.compile(r"^(?:![A-Za-z0-9]+:\S+|\$[\w+/-]{20,}|#[^\s:#]+:\S+|matrix:(?:r|roomid)/|matrix\.to/#/)")


def _looks_like_junk(name: str) -> bool:
    """Whether a name is a path, code or data file, URL, env var, hash, or Matrix room or event."""
    if _IDENTITY.match(name):
        return False
    if any(pattern.search(name) for pattern in (_URL, _ABSOLUTE_PATH, _RELATIVE_PATH, _DEEP_PATH, _ENV_VAR, _MATRIX)):
        return True
    if _looks_like_hash(name):
        return True
    file = _FILE.search(name)
    if file is None:
        return False
    stem, extension = file["stem"], file["extension"].lower()
    product = extension in _PRODUCT_EXTENSIONS and _PRODUCT_STEM.match(stem) is not None
    domain = extension in _DOMAIN_EXTENSIONS and _DOMAIN_STEM.match(stem) is not None
    return not (product or domain)


def _looks_like_hash(name: str) -> bool:
    """Whether a name is a UUID or lowercase hex with letters and digits mixed, unlike `feb2026` or `face2face`."""
    if _UUID.match(name):
        return True
    if not _HEX.match(name):
        return False
    digits = sum(character.isdigit() for character in name)
    return digits >= 2 and len(name) - digits >= 2 and len(_HEX_RUN.findall(name)) >= 3


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
    again finishes it. Names in ``duplicates`` that match no entity are reported as unmatched.
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
    duplicate_entities = [*same_name, *named]
    order = {entity.uuid: index for index, entity in enumerate(duplicate_entities)}
    group = {kept.uuid, *order}
    facts = await _records(
        driver,
        _FACTS_TOUCHING_WITH_PROPERTIES if apply else _FACTS_TOUCHING,
        uuids=list(order),
    )
    merged = _merged(duplicate_entities, kept.uuid, facts, await _records(driver, _MENTIONS_OF, uuids=[*group]))
    unmatched = await _unmatched(driver, duplicates)
    moving = [fact for fact in facts if not _between(fact, group)]
    if apply:
        for fact in moving:
            await _move_fact(driver, fact, group, kept.uuid)
        await driver.execute_query(_MOVE_MENTIONS, uuids=list(order), keep=kept.uuid)
        await driver.execute_query(_DELETE, uuids=list(order))
    return MergeReport(
        keep=kept,
        merged=merged,
        facts_moved=len(moving),
        self_loops_dropped=len(facts) - len(moving),
        unmatched=unmatched,
        applied=apply,
    )


def _between(fact: dict[str, Any], group: set[str]) -> bool:
    """Whether both ends of a fact are in ``group``, so merging the group would make it a self-loop."""
    return fact["source"] in group and fact["target"] in group


def _merged(
    duplicates: list[EntityCount],
    keep: str,
    facts: list[dict[str, Any]],
    mentions: list[dict[str, Any]],
) -> list[_MergedEntity]:
    """Count what each duplicate moves and drops, giving a fact or mention two duplicates share to the first."""
    order = {entity.uuid: index for index, entity in enumerate(duplicates)}
    group = {keep, *order}
    moved_facts = dict.fromkeys(order, 0)
    dropped_facts = dict.fromkeys(order, 0)
    for fact in facts:
        owner = min((uuid for uuid in (fact["source"], fact["target"]) if uuid in order), key=order.__getitem__)
        (dropped_facts if _between(fact, group) else moved_facts)[owner] += 1
    # The merge keeps one mention per episode: the kept entity's own, or else the first duplicate's.
    mentioned = {mention["episode"] for mention in mentions if mention["entity"] == keep}
    moved_mentions = dict.fromkeys(order, 0)
    dropped_mentions = dict.fromkeys(order, 0)
    for mention in sorted(
        (mention for mention in mentions if mention["entity"] in order),
        key=lambda mention: order[mention["entity"]],
    ):
        if mention["episode"] in mentioned:
            dropped_mentions[mention["entity"]] += 1
        else:
            moved_mentions[mention["entity"]] += 1
            mentioned.add(mention["episode"])
    return [
        _MergedEntity(
            name=entity.name,
            uuid=entity.uuid,
            facts_moved=moved_facts[entity.uuid],
            self_loops_dropped=dropped_facts[entity.uuid],
            mentions_moved=moved_mentions[entity.uuid],
            mentions_dropped=dropped_mentions[entity.uuid],
        )
        for entity in duplicates
    ]


async def _unmatched(driver: GraphDriver, names: Sequence[str]) -> list[str]:
    """Return the names no entity has, ignoring case, in the order given."""
    missing = {record["wanted"] for record in await _records(driver, _UNMATCHED, names=list(names))}
    return [name for name in dict.fromkeys(names) if name in missing]


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
    await driver.execute_query(
        _MOVE_FACT,
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

    An entity goes with its facts and mentions. Facts it shared with other entities are gone from those too. Names
    that match no entity are reported as unmatched.
    """
    deleted = await _entities(driver, _NAMED + _COUNTED, names=list(names))
    if junk:
        listed = {entity.uuid for entity in deleted}
        everything = await _records(driver, "MATCH (entity:Entity) RETURN entity.name AS name, entity.uuid AS uuid")
        uuids = [row["uuid"] for row in everything if row["uuid"] not in listed and _looks_like_junk(row["name"])]
        deleted += await _entities(driver, _WITH_UUID + _COUNTED, uuids=uuids)
    unmatched = await _unmatched(driver, names)
    if apply and deleted:
        await driver.execute_query(_DELETE, uuids=[entity.uuid for entity in deleted])
    return DeleteReport(deleted=deleted, unmatched=unmatched, applied=apply)
