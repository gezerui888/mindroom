---
icon: lucide/network
---

# Knowledge Graph

A knowledge graph gives an agent a model of the people, projects, documents, and events its conversations and tools reveal, with when each fact held and where it came from.
MindRoom builds it with [Graphiti](https://github.com/getzep/graphiti) in the background from the agent's finished conversations, and the agent looks facts up with the `knowledge_graph` tool.
To have the agent also explore its data sources on a schedule, add the [`exploration`](scheduling.md#exploration) automation.
[Memory](memory.md) keeps what is worth remembering as text and a [knowledge base](knowledge.md) searches your documents; the graph links the entities that conversations and tool results reveal, with dated facts, so an agent can answer who owns what, how things connect, and what changed.

## Enable It

```yaml
models:
  luna:
    provider: openai
    id: gpt-6-luna

knowledge_graph:
  model: luna          # extracts entities and facts

agents:
  mind:
    display_name: Mind
    knowledge_graph: true
```

`agents.<name>.knowledge_graph` (bool, default from `defaults.knowledge_graph`, which defaults to `false`) turns the graph on for one agent and gives it the `knowledge_graph` tool.
MindRoom installs the optional `knowledge_graph` extra the first time a graph is used; its embedded database ships for macOS and for Linux with glibc 2.39 or newer, and elsewhere it must build from source.
Extraction needs a model with reliable structured output; small local models often fail it; a failed extraction is retried later, and a part of a conversation is skipped once its extraction output was unusable five times or it kept failing for three days.

## Settings

| Field | Default | Description |
|---|---|---|
| `knowledge_graph.model` | none | A key of `models` that extracts entities and facts; required once any agent enables the graph |
| `knowledge_graph.ingest.interval_seconds` | `600` (min 30) | Time between ingestion passes |
| `knowledge_graph.ingest.idle_seconds` | `120` (min 1) | How long a conversation must be quiet before its new turns are added |
| `knowledge_graph.ingest.max_sessions_per_cycle` | `10` (min 1) | Conversations added per pass, across all graphs; the rest wait for the next |
| `knowledge_graph.ingest.max_episode_chars` | `24000` (1000 to 60000) | Largest piece of a conversation sent for extraction at once; a single longer message is cut to this length |
| `knowledge_graph.ingest.max_tool_result_chars` | `4000` (min 100, at most half of `max_episode_chars`) | Characters kept from each tool result |
| `knowledge_graph.ingest.backfill_days` | `7` (min 0) | A new graph starts from conversations active within this many days |

With the defaults, new turns reach the graph 2 to 12 minutes after a conversation goes quiet, plus extraction time, and later when many conversations are waiting.
Facts are embedded with [`memory.embedder`](memory.md#embedder).
The graph database runs inside MindRoom and keeps every graph in memory, so MindRoom's memory use grows with the graphs.
Changing the embedder starts a new, empty graph that fills again from conversations active within `backfill_days`; the old graph is no longer read but keeps its memory until you stop MindRoom and delete `knowledge_graph/` in the storage directory.

## What Goes Into the Graph

- Every finished turn of the agent's Matrix conversations: what people said, what the agent answered, and the tool calls it made with their results.
- From [`exploration`](scheduling.md#exploration) threads, tasks another agent delegated to it, and an automation's turn inside an ordinary conversation, only the tool calls and their results, never the agent's own summary.

Not added: threads started by other automations such as `dreaming`, graph lookups, including those through a shell, the `reasoning` tool's thoughts, answers from subagents, workflows, and Claude Code, results collected from background jobs, conversations through the OpenAI-compatible API, and team conversations.
Credential-like values are removed on a best-effort basis before anything is sent for extraction.
A turn waiting for tool approval holds back the later turns of its conversation until it finishes or its approval expires.

Each fact keeps its sources: `conversation` means someone said it, `tool_results` means it came from a tool call, its arguments or what it returned.
When later evidence contradicts a fact, the fact is marked superseded with the date it stopped holding instead of being deleted.
Facts are extracted by a model, so they can be wrong; check a fact's source before relying on it.
A message redacted before its conversation is added stays out of the graph, and so does the turn that read it.
Redacting a message later does not remove facts already extracted from it; to forget them, stop MindRoom and delete `knowledge_graph/` in the storage directory, and the graph rebuilds from the remaining conversations of the last `backfill_days`, without redacted messages.

## The `knowledge_graph` Tool

```python
search_graph("who owns Project Atlas", limit=10)   # limit 1 to 50
graph_entity("Project Atlas")
```

The agent decides when to look facts up; ask it to search the graph, or to cite each fact's source and date, when you want to be sure.
Each fact comes back with its validity, for example `valid 2026-01-05 until 2026-03-09 (superseded)`, and its sources with dates.
An agent reads only its own graph; a [private agent](configuration/agents.md#private-instances) has one graph per requester.

## Cleaning Up a Graph

Extraction can store one person under several names, such as a full name, a first name, an email address, and a Matrix ID, and it has turned file paths, URLs, and room links into entities.
The `mindroom knowledge-graph` commands merge duplicates into one entity and delete junk entities in a shared agent's graph.
They work through the running MindRoom's API and need `MINDROOM_API_KEY` both in MindRoom's environment and in the environment the command reads (`--config`).
Only shared agents can be cleaned up, because a private agent has one graph per requester.
Every command is a dry run that changes nothing unless you add `--apply`.

1. Snapshot the graph first: stop MindRoom, copy `knowledge_graph/` from the storage directory, and start MindRoom again.
    To undo a change, stop MindRoom and put the copy back.
2. Find the names to merge or delete:

    ```bash
    mindroom knowledge-graph entities mind ann
    ```

3. Preview a merge, read the report, then apply it:

    ```bash
    mindroom knowledge-graph merge mind "Ann Lee" Ann ann@example.org @ann:example.org
    mindroom knowledge-graph merge mind "Ann Lee" Ann ann@example.org @ann:example.org --apply
    ```

4. Preview deleting junk, read every name in the list, then apply it:

    ```bash
    mindroom knowledge-graph delete mind --junk
    mindroom knowledge-graph delete mind --junk --apply
    ```

A merge matches names exactly, ignoring case, and also merges any other entity with exactly the kept name.
Each duplicate's facts move to the kept entity with their dates and sources, the episodes that mentioned a duplicate mention the kept entity instead, and the duplicates are deleted.
Facts between the merged entities are dropped rather than left pointing from the kept entity to itself.
The kept entity keeps its own name and summary, and fact texts still read as extracted, for example "Ann moved to Lisbon".
The report lists, for each duplicate, the facts and mentions that move and those dropped, and any name that matched no entity; a misspelled name is skipped, so check that list before applying.

A delete removes each entity with its facts and mentions, including facts it shared with other entities.
`--junk` adds every entity whose name looks like a file path, a file with a code or data extension, a URL, an environment variable, a commit hash or UUID, or a Matrix room ID, alias, or link.
The patterns are guesses, and a real entity can land in the list; when it does, delete the others by name instead, for example `mindroom knowledge-graph delete mind compaction.py config.yaml --apply`.
Email addresses and Matrix user IDs are never treated as junk; merge them into the person they belong to.

While a command runs, ingestion pauses between episodes.
A command first waits for the episode being added to finish, which can take a few minutes, so it gives up after `--timeout` seconds (600 by default); the change may still finish after that, so run the dry run again to see the graph.
The commands call operator routes that require `MINDROOM_API_KEY` as a bearer token: `GET /api/knowledge-graph/<agent>/entities?query=&limit=`, `POST /api/knowledge-graph/<agent>/merge` with `{"keep": ..., "duplicates": [...], "apply": false}`, and `POST /api/knowledge-graph/<agent>/delete` with `{"names": [...], "junk": false, "apply": false}`.

## Privacy

A shared agent's graph holds what it learned from every requester's conversations, including tool results, and anyone who can talk to that agent can query it.
Use a private agent when requesters must not see each other's facts.

## Cost and Health

Extraction is reported as `knowledge_graph` in [usage tracking](usage.md#token-usage) and makes about 15 model calls per piece of a conversation.
Each lookup makes one embedding request.
When ingestion fails, `/api/health` includes a `knowledge_graph` block naming the kind of failure until a later pass succeeds with no failed episode still waiting for its retry, and the MindRoom log has the details.
