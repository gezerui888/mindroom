"""Operator maintenance of one scope's graph, against the real embedded FalkorDB server and Graphiti's own writes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio
from falkordb.asyncio import FalkorDB
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk

from mindroom.knowledge_graph.clients import MindRoomFalkorDriver
from mindroom.knowledge_graph.embedded_server import start_embedded_server
from mindroom.knowledge_graph.maintenance import (
    EntityCount,
    MergeInputError,
    _looks_like_junk,
    _MergedEntity,
    delete_entities,
    list_entities,
    merge_entities,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from graphiti_core.driver.driver import GraphDriver

GRAPH = "kg_" + "c" * 24
JAN = datetime(2026, 1, 5, tzinfo=UTC)


@pytest_asyncio.fixture
async def graph(tmp_path: Path) -> AsyncIterator[GraphDriver]:
    """A driver for one scope's graph on a fresh embedded server, addressed as the store addresses it."""
    server = await start_embedded_server(tmp_path)
    driver = MindRoomFalkorDriver(falkor_db=FalkorDB(unix_socket_path=server.socket_path), database=GRAPH)
    try:
        yield driver
    finally:
        await driver.close()
        await server.stop()


def _entity(name: str) -> EntityNode:
    return EntityNode(name=name, group_id=GRAPH, summary=f"About {name}.", name_embedding=[0.5, 0.5, 0.5, 0.5])


def _fact(source: EntityNode, target: EntityNode, fact: str, embedding: list[float]) -> EntityEdge:
    return EntityEdge(
        source_node_uuid=source.uuid,
        target_node_uuid=target.uuid,
        name="RELATES",
        fact=fact,
        group_id=GRAPH,
        episodes=["episode-1"],
        created_at=JAN,
        valid_at=JAN,
        fact_embedding=embedding,
    )


async def _ingest(
    graph: GraphDriver,
    entities: list[EntityNode],
    facts: list[EntityEdge],
    mentioned: list[EntityNode],
) -> EpisodicNode:
    """Save as Graphiti's ingestion does, in one bulk write: an episode mentioning ``mentioned``, entities, facts."""
    episode = EpisodicNode(
        name="conversation",
        group_id=GRAPH,
        source=EpisodeType.message,
        source_description="conversation: room !r:x",
        content="person @bas (x): hi",
        valid_at=JAN,
    )
    mentions = [
        EpisodicEdge(source_node_uuid=episode.uuid, target_node_uuid=node.uuid, group_id=GRAPH, created_at=JAN)
        for node in mentioned
    ]
    # Every embedding is already set, so Graphiti needs no embedder.
    await add_nodes_and_edges_bulk(graph, [episode], mentions, entities, facts, None)
    return episode


async def _rows(graph: GraphDriver, cypher: str, **params: object) -> list[dict[str, Any]]:
    records, _, _ = await graph.execute_query(cypher, **params)
    return records


async def _stored_fact(graph: GraphDriver, uuid: str) -> list[dict[str, Any]]:
    return await _rows(
        graph,
        "MATCH (source:Entity)-[fact:RELATES_TO {uuid: $uuid}]->(target:Entity) "
        "RETURN source.name AS source, target.name AS target, properties(fact) AS properties, "
        "typeOf(fact.fact_embedding) AS embedding_type",
        uuid=uuid,
    )


async def _mentions(graph: GraphDriver) -> list[tuple[str, str]]:
    # The edge is returned because FalkorDB reports parallel edges between two nodes once when a query does not use it.
    rows = await _rows(
        graph,
        "MATCH (episode:Episodic)-[mention:MENTIONS]->(entity:Entity) "
        "RETURN episode.uuid AS episode, entity.name AS entity, mention.uuid AS mention",
    )
    return sorted((row["episode"], row["entity"]) for row in rows)


async def _names(graph: GraphDriver, query: str = "") -> set[str]:
    return {entity.name for entity in await list_entities(graph, query)}


@pytest.mark.asyncio
async def test_merge_moves_facts_and_mentions_and_drops_the_duplicate(graph: GraphDriver) -> None:
    """The duplicate's facts move to the kept entity whole, its mentions follow, and a fact between the two goes."""
    kept, bas, acme, seattle = _entity("Bas Nijholt"), _entity("Bas"), _entity("Acme"), _entity("Seattle")
    f1 = _fact(kept, acme, "Bas Nijholt works at Acme", [1.0, 0.0, 0.0, 0.0])
    f2 = _fact(bas, seattle, "Bas lives in Seattle", [0.1, 0.2, 0.3, 0.4])
    f3 = _fact(bas, kept, "Bas is short for Bas Nijholt", [0.0, 1.0, 0.0, 0.0])
    episode = await _ingest(graph, [kept, bas, acme, seattle], [f1, f2, f3], [bas])

    report = await merge_entities(graph, "Bas Nijholt", ["Bas"], apply=False)

    assert (report.facts_moved, report.self_loops_dropped, report.applied) == (1, 1, False)
    assert report.keep == EntityCount(name="Bas Nijholt", uuid=kept.uuid, facts=2, mentions=0)
    assert report.merged == [
        _MergedEntity(
            name="Bas",
            uuid=bas.uuid,
            facts_moved=1,
            self_loops_dropped=1,
            mentions_moved=1,
            mentions_dropped=0,
        ),
    ]
    assert report.unmatched == []
    assert await _names(graph, "bas") == {"Bas Nijholt", "Bas"}  # dry run changed nothing
    assert len(await _stored_fact(graph, f3.uuid)) == 1

    applied = await merge_entities(graph, "Bas Nijholt", ["Bas"], apply=True)

    assert (applied.facts_moved, applied.self_loops_dropped, applied.applied) == (1, 1, True)
    assert await _names(graph, "bas") == {"Bas Nijholt"}
    [moved] = await _stored_fact(graph, f2.uuid)
    assert (moved["source"], moved["target"]) == ("Bas Nijholt", "Seattle")
    assert moved["properties"]["fact"] == "Bas lives in Seattle"
    assert moved["properties"]["episodes"] == ["episode-1"]
    assert moved["embedding_type"] == "Vectorf32"
    assert moved["properties"]["fact_embedding"] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    # Graphiti's bulk write stores these endpoint copies; the rewrite does not invent the other pair.
    assert moved["properties"]["source_node_uuid"] == kept.uuid
    assert moved["properties"]["target_node_uuid"] == seattle.uuid
    assert "source_uuid" not in moved["properties"]
    # Graphiti reads the moved fact back from its new endpoint.
    read_back = await EntityEdge.get_by_uuid(graph, f2.uuid)
    assert (read_back.source_node_uuid, read_back.target_node_uuid) == (kept.uuid, seattle.uuid)
    assert await _mentions(graph) == [(episode.uuid, "Bas Nijholt")]
    assert await _stored_fact(graph, f3.uuid) == []
    assert len(await _stored_fact(graph, f1.uuid)) == 1


@pytest.mark.asyncio
async def test_merge_rejects_keep_in_duplicates_or_missing(graph: GraphDriver) -> None:
    """Merging an entity into itself, or into an entity that does not exist, fails before anything changes."""
    kept, bas, seattle = _entity("Bas Nijholt"), _entity("Bas"), _entity("Seattle")
    lives = _fact(bas, seattle, "Bas lives in Seattle", [0.1, 0.2, 0.3, 0.4])
    await _ingest(graph, [kept, bas, seattle], [lives], [bas])

    with pytest.raises(MergeInputError, match="both the entity to keep and a duplicate"):
        await merge_entities(graph, "Bas Nijholt", ["Bas", "bas nijholt"], apply=True)
    with pytest.raises(MergeInputError, match="No entity is named"):
        await merge_entities(graph, "Bas N.", ["Bas"], apply=True)

    assert await _names(graph) == {"Bas Nijholt", "Bas", "Seattle"}
    assert len(await _rows(graph, "MATCH ()-[fact:RELATES_TO]->() RETURN fact.uuid AS uuid")) == 1


@pytest.mark.asyncio
async def test_merging_several_duplicates_drops_the_facts_between_them(graph: GraphDriver) -> None:
    """Facts among the merged entities become no self-loops, and an episode mentioning several keeps one mention.

    Each fact and mention is reported under one duplicate, the first in merge order, so the dry run's numbers add up to
    what the merge does.
    """
    kept, same_name = _entity("Bas Nijholt"), _entity("Bas Nijholt")
    bas, email, acme, fastmail = _entity("Bas"), _entity("bas@nijho.lt"), _entity("Acme"), _entity("Fastmail")
    works = _fact(kept, acme, "Bas Nijholt works at Acme", [1.0, 0.0, 0.0, 0.0])
    # Parallel facts between two entities, which FalkorDB reports once unless a query reads the edges.
    uses = _fact(bas, email, "Bas uses bas@nijho.lt", [0.0, 1.0, 0.0, 0.0])
    signs = _fact(bas, email, "Bas signs mail as bas@nijho.lt", [0.0, 1.0, 1.0, 0.0])
    since = _fact(email, fastmail, "bas@nijho.lt has been on Fastmail since 2019", [0.0, 0.0, 1.0, 1.0])
    entities = [kept, same_name, bas, email, acme, fastmail]
    episode = await _ingest(graph, entities, [works, uses, signs, since], [bas, email])
    # An episode that already mentions the kept entity keeps that one mention.
    other_episode = await _ingest(graph, [], [], [kept, bas])
    # A fact Graphiti saved one at a time, which stores the other pair of endpoint copies.
    hosted = _fact(email, fastmail, "bas@nijho.lt is hosted by Fastmail", [0.0, 0.0, 1.0, 0.0])
    await hosted.save(graph)

    dry_run = await merge_entities(graph, "Bas Nijholt", ["Bas", "BAS@nijho.lt"], apply=False)
    report = await merge_entities(graph, "Bas Nijholt", ["Bas", "BAS@nijho.lt"], apply=True)

    # The kept entity is the one with most facts; another entity with its name is merged into it too.
    assert report.keep == EntityCount(name="Bas Nijholt", uuid=kept.uuid, facts=1, mentions=1)
    assert report.merged == [
        # Facts moved, self-loops dropped, mentions moved, mentions dropped. What two duplicates share counts
        # under the first: the facts between them, and the episode that mentions both.
        _MergedEntity("Bas Nijholt", same_name.uuid, 0, 0, 0, 0),
        _MergedEntity("bas@nijho.lt", email.uuid, 2, 2, 1, 0),
        _MergedEntity("Bas", bas.uuid, 0, 0, 0, 2),
    ]
    assert dry_run.merged == report.merged
    assert (dry_run.facts_moved, dry_run.self_loops_dropped, dry_run.applied) == (2, 2, False)
    assert (report.facts_moved, report.self_loops_dropped, report.applied) == (2, 2, True)
    # The kept entity now has what the report said would move, and nothing that it said would be dropped.
    assert await list_entities(graph, "bas nijholt") == [
        EntityCount(name="Bas Nijholt", uuid=kept.uuid, facts=1 + 2, mentions=1 + 1),
    ]
    assert await _names(graph) == {"Bas Nijholt", "Acme", "Fastmail"}
    assert await _rows(graph, "MATCH (entity:Entity)-[fact:RELATES_TO]->(entity) RETURN fact.uuid AS uuid") == []
    assert await _stored_fact(graph, uses.uuid) == await _stored_fact(graph, signs.uuid) == []
    assert len(await _stored_fact(graph, since.uuid)) == 1
    [moved] = await _stored_fact(graph, hosted.uuid)
    assert (moved["source"], moved["target"]) == ("Bas Nijholt", "Fastmail")
    assert (moved["properties"]["source_uuid"], moved["properties"]["target_uuid"]) == (kept.uuid, fastmail.uuid)
    assert "source_node_uuid" not in moved["properties"]
    assert moved["embedding_type"] == "Vectorf32"
    assert await _mentions(graph) == sorted([(episode.uuid, "Bas Nijholt"), (other_episode.uuid, "Bas Nijholt")])


@pytest.mark.asyncio
async def test_names_that_match_nothing_are_reported(graph: GraphDriver) -> None:
    """A mistyped name is listed as unmatched, so an operator does not merge or delete half of what they meant."""
    kept, bas, seattle = _entity("Bas Nijholt"), _entity("Bas"), _entity("Seattle")
    await _ingest(graph, [kept, bas, seattle], [], [bas])

    merge = await merge_entities(graph, "Bas Nijholt", ["Bsa", "bas", "Bas N."], apply=False)
    delete = await delete_entities(graph, ["Seatle", "SEATTLE"], apply=False)

    assert [entity.name for entity in merge.merged] == ["Bas"]
    assert merge.unmatched == ["Bsa", "Bas N."]
    assert [entity.name for entity in delete.deleted] == ["Seattle"]
    assert delete.unmatched == ["Seatle"]


@pytest.mark.parametrize(
    ("name", "junk"),
    [
        ("/home/basnijholt/.mindroom-chat/config.yaml", True),
        ("compaction.py", True),
        ("memory/2026-04-05.md", True),
        ("https://github.com/x/y/pull/1", True),
        ("capacitor://localhost/!r:x/$e", True),
        ("MINDROOM_API_KEY", True),
        ("3442fa792", True),
        ("!l3DXzqtF5lrDRsNeOO:mindroom.chat", True),
        ("Bas Nijholt", False),
        ("MindRoom", False),
        ("OpenAI", False),
        ("ISSUE-267", False),
        ("Lake Washington PT", False),
        # Aliases of a person are merged into the person, never deleted as junk.
        ("bas@nijho.lt", False),
        ("@basnijholt:mindroom.chat", False),
        ("Node.js", False),
        # Product names that end like a file, words with a number, and slashed names that are no paths.
        ("Bun.sh", False),
        ("docs.rs", False),
        ("Tailwind.css", False),
        ("feb2026", False),
        ("dec2025", False),
        ("face2face", False),
        ("abc1234", False),
        ("1/2/2026", False),
        ("A/B/C", False),
        ("TCP/IP", False),
        ("e5eaa51", True),
        ("MEMORY.md", True),
        ("src/mindroom", True),
        ("index.js", True),
        ("vite.config.js", True),
        ("#general:mindroom.chat", True),
        ("matrix:r/general:mindroom.chat", True),
        ("matrix.to/#/!room:mindroom.chat", True),
        ("#MeToo", False),
        # Only a few extensions end product names; a capitalized file name with another is still a file.
        ("Readme.md", True),
        ("Config.yaml", True),
        ("Setup.py", True),
        ("App.tsx", True),
        ("HelloWorld.java", True),
        ("MyComponent.vue", True),
        ("Notes.txt", True),
        # Capitalized paths need two slashes and segments longer than a letter.
        ("Users/Ann/Desktop", True),
        ("Projects/Atlas/Docs", True),
        ("AC/DC", False),
        # Aliases on domains that end like files are still aliases.
        ("ann@example.rs", False),
        ("ann@example.sh", False),
        ("ann@example.md", False),
        ("ann@example.py", False),
        ("ann@example.js", False),
        ("@ann:example.rs", False),
        ("@ann:example.sh", False),
        ("@ann:example.md", False),
        ("@ann:example.py", False),
        ("@ann:example.js", False),
    ],
)
def test_looks_like_junk(name: str, junk: bool) -> None:
    """Paths, code and data files, URLs, env vars, hashes, and Matrix room links are junk; names of things are not."""
    assert _looks_like_junk(name) is junk


@pytest.mark.asyncio
async def test_delete_junk_removes_nodes_and_their_facts(graph: GraphDriver) -> None:
    """Deleting removes the junk entities, named ones too, with their facts and mentions; everything else stays."""
    source_file, config_file = _entity("compaction.py"), _entity("/home/basnijholt/.mindroom-chat/config.yaml")
    mindroom, acme, seattle = _entity("MindRoom"), _entity("Acme"), _entity("Seattle")
    in_mindroom = _fact(source_file, mindroom, "compaction.py is part of MindRoom", [1.0, 0.0, 0.0, 0.0])
    configures = _fact(config_file, mindroom, "config.yaml configures MindRoom", [0.0, 1.0, 0.0, 0.0])
    runs_on = _fact(mindroom, acme, "MindRoom runs on Acme servers", [0.0, 0.0, 1.0, 0.0])
    episode = await _ingest(
        graph,
        [source_file, config_file, mindroom, acme, seattle],
        [in_mindroom, configures, runs_on],
        [source_file, mindroom],
    )

    report = await delete_entities(graph, junk=True, apply=False)

    assert sorted(report.deleted, key=lambda entity: entity.name) == [
        EntityCount(name="/home/basnijholt/.mindroom-chat/config.yaml", uuid=config_file.uuid, facts=1, mentions=0),
        EntityCount(name="compaction.py", uuid=source_file.uuid, facts=1, mentions=1),
    ]
    assert report.applied is False
    assert len(await _names(graph)) == 5  # dry run changed nothing

    applied = await delete_entities(graph, ["seattle"], junk=True, apply=True)

    assert {entity.name for entity in applied.deleted} == {
        "Seattle",
        "compaction.py",
        "/home/basnijholt/.mindroom-chat/config.yaml",
    }
    assert applied.applied is True
    assert await _names(graph) == {"MindRoom", "Acme"}
    assert [row["uuid"] for row in await _rows(graph, "MATCH ()-[fact:RELATES_TO]->() RETURN fact.uuid AS uuid")] == [
        runs_on.uuid,
    ]
    assert await _mentions(graph) == [(episode.uuid, "MindRoom")]


@pytest.mark.asyncio
async def test_junk_deletion_keeps_aliases_for_merging(graph: GraphDriver) -> None:
    """An email address or Matrix ID on a domain that ends like a file keeps its facts for merging into the person."""
    person, email, matrix_id = _entity("Ann Lee"), _entity("ann@example.rs"), _entity("@ann:example.sh")
    source_file = _entity("compaction.py")
    writes = _fact(email, person, "ann@example.rs belongs to Ann Lee", [1.0, 0.0, 0.0, 0.0])
    chats = _fact(matrix_id, person, "@ann:example.sh is Ann Lee on Matrix", [0.0, 1.0, 0.0, 0.0])
    await _ingest(graph, [person, email, matrix_id, source_file], [writes, chats], [email, matrix_id])

    report = await delete_entities(graph, junk=True, apply=True)

    assert [entity.name for entity in report.deleted] == ["compaction.py"]
    assert await _names(graph) == {"Ann Lee", "ann@example.rs", "@ann:example.sh"}
    assert len(await _stored_fact(graph, writes.uuid)) == len(await _stored_fact(graph, chats.uuid)) == 1
