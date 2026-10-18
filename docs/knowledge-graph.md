---
icon: lucide/network
---

# Knowledge Graph

A knowledge graph gives an agent a model of the people, projects, documents, and events its conversations, tools, and files reveal, with when each fact held and where it came from.
MindRoom builds it with [Graphiti](https://github.com/getzep/graphiti) in the background from the agent's finished conversations and, optionally, the files of its [knowledge bases](#documents), and the agent looks facts up with the `knowledge_graph` tool.
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
| `knowledge_graph.ingest.max_sessions_per_cycle` | `10` (min 1) | Conversations and changed [files](#documents) added per pass, across all graphs; the rest wait for the next |
| `knowledge_graph.ingest.max_episode_chars` | `24000` (1000 to 60000) | Largest piece of a conversation or file sent for extraction at once; a single longer message or line is cut to this length |
| `knowledge_graph.ingest.max_tool_result_chars` | `4000` (min 100, at most half of `max_episode_chars`) | Characters kept from each tool result |
| `knowledge_graph.ingest.backfill_days` | `7` (min 0) | A new graph starts from conversations active within this many days |

With the defaults, new turns reach the graph 2 to 12 minutes after a conversation goes quiet, plus extraction time, and later when many conversations are waiting.
Facts are embedded with [`memory.embedder`](memory.md#embedder).
The graph database runs inside MindRoom and keeps every graph in memory, so MindRoom's memory use grows with the graphs.
Changing the embedder starts a new, empty graph that fills again from conversations active within `backfill_days`; the old graph is no longer read but keeps its memory until you stop MindRoom and delete `knowledge_graph/` in the storage directory.

## What Goes Into the Graph

- Every finished turn of the agent's Matrix conversations: what people said, what the agent answered, and the tool calls it made with their results.
- From [`exploration`](scheduling.md#exploration) threads, tasks another agent delegated to it, and an automation's turn inside an ordinary conversation, only the tool calls and their results, never the agent's own summary.
- The files of knowledge bases flagged `knowledge_graph: true` that the agent uses; see [Documents](#documents).

Not added: threads started by other automations such as `dreaming`, graph lookups, including those through a shell, the `reasoning` tool's thoughts, answers from subagents, workflows, and Claude Code, results collected from background jobs, conversations through the OpenAI-compatible API, and team conversations.
Credential-like values are removed on a best-effort basis before anything is sent for extraction.
A turn waiting for tool approval holds back the later turns of its conversation until it finishes or its approval expires.

Each fact keeps its sources: `conversation` means someone said it, `tool_results` means it came from a tool call, its arguments or what it returned, and `document: <base>/<path>` means it was read from that knowledge-base file.
When later evidence contradicts a fact, the fact is marked superseded with the date it stopped holding instead of being deleted.
Facts are extracted by a model, so they can be wrong; check a fact's source before relying on it.
A message redacted before its conversation is added stays out of the graph, and so does the turn that read it.
Redacting a message later does not remove facts already extracted from it; to forget them, stop MindRoom and delete `knowledge_graph/` in the storage directory, and the graph rebuilds from the remaining conversations of the last `backfill_days`, without redacted messages.

## Documents

Flag a [knowledge base](knowledge.md) with `knowledge_graph: true` to also extract its files into the graph of every graph-enabled agent that lists the base in `knowledge_bases`:

```yaml
knowledge_bases:
  facts:
    path: ./memory/facts
    knowledge_graph: true

agents:
  mind:
    display_name: Mind
    knowledge_graph: true
    knowledge_bases: [facts]
```

MindRoom reads each file itself, so a fact's evidence is the file's own text and no model spends tokens reading the files in a conversation first.
Files are chosen by the base's own rules: include and exclude patterns, extensions, hidden files, and for a Git-backed base only tracked files.
Every pass hashes each file of the flagged bases; a new or changed file becomes `document` episodes: its text with credential-like values removed, split between lines into pieces of at most `max_episode_chars`, dated by the file's modification time.
A file whose content is unchanged is not extracted again, even when a Git checkout or copy gives it a new modification time.
Each changed file counts as one of the pass's `max_sessions_per_cycle`, after the conversations, and adds at most twenty pieces per pass.
A failed piece is retried and skipped like a part of a conversation, and a file edited while a piece waits for its retry starts over with the new version.
Deleting a file, or a line of it, does not remove the facts it already added; a new version adds its facts, and Graphiti marks the facts it contradicts superseded.
A file that is not UTF-8 text adds nothing.
The flag exists only on `knowledge_bases` entries; an agent's `private.knowledge` cannot feed the graph.

Each changed file costs about 15 extraction calls per piece, so a 100,000-character file costs about five pieces' worth every time its content changes, in every graph that reads it: a [private agent](configuration/agents.md#private-instances) extracts it once per requester.
An unchanged file costs only reading and hashing it on each pass.
Flag curated files that state facts, such as people, projects, and decisions, rather than daily notes or logs that change every day.

## The `knowledge_graph` Tool

```python
search_graph("who owns Project Atlas", limit=10)   # limit 1 to 50
graph_entity("Project Atlas")
```

The agent decides when to look facts up; ask it to search the graph, or to cite each fact's source and date, when you want to be sure.
Each fact comes back with its validity, for example `valid 2026-01-05 until 2026-03-09 (superseded)`, and its sources with dates.
An agent reads only its own graph; a [private agent](configuration/agents.md#private-instances) has one graph per requester.

## Privacy

A shared agent's graph holds what it learned from every requester's conversations, including tool results, and anyone who can talk to that agent can query it.
Use a private agent when requesters must not see each other's facts.

## Cost and Health

Extraction is reported as `knowledge_graph` in [usage tracking](usage.md#token-usage), under `system:internal` for [files](#documents), and makes about 15 model calls per piece of a conversation or file.
Each lookup makes one embedding request.
When ingestion fails, `/api/health` includes a `knowledge_graph` block naming the kind of failure until a later pass succeeds with no failed episode still waiting for its retry, and the MindRoom log has the details.
