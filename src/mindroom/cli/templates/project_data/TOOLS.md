# TOOLS.md - Tool Mechanics

## Your prompt

- `AGENTS.md`, `SOUL.md`, `IDENTITY.md`, `TOOLS.md`, and `MEMORY.md` are in your prompt on every turn.
- Files under `memory/` are not preloaded; lines matching a message may be added to your prompt, and `search_memories` finds more.
- `add_memory` appends to `MEMORY.md`; your file tools write the topic files.
- Your prompt lists each skill by name and description; load the full text with `get_skill_instructions`.

## Integrations

| Integration | Account | May read | May write |
| --- | --- | --- | --- |
| _(tool or knowledge base)_ | _(shared service account, or the requester's own)_ | _(what it can see)_ | _(what it can change, or nothing)_ |

- A tool on the requester's own account acts as whoever is asking, so the same question can return different results for different people.
- `search_knowledge_base` answers from the last completed index, which can lag behind the source.
- Your prompt names your deferred toolkits; find a function with `tool_search` and run it with `call_tool`, or, when your model has its own tool search, call what it finds directly.
- When a call needs approval, MindRoom posts a card that only the requester can approve.

## Worker sandbox

- With a worker backend configured, `shell` and `file` run by default in an isolated worker instead of the main MindRoom process.
- A worker can be recreated, which ends its background shells, so keep anything that must last inside the workspace.
- Log each failure caused by the sandbox itself, such as a missing program or an unreachable host, in `memory/sandbox-friction.md`.

## Scheduling

- `list_schedules()` shows the room's pending tasks, or only the current thread's when called inside a thread.
- `schedule(request, new_thread, history_limit, silent)` adds a task.
- With `silent=True`, a run that returns only `NO_REPLY` posts nothing, while findings and failures still appear.
- Each run acts as the schedule's creator, and MindRoom cancels the schedule once the creator leaves the room.

| Task ID | Task | Cadence and timezone | Destination | Owner | Agreed |
| --- | --- | --- | --- | --- | --- |

## Delegates

- A subagent starts without this conversation and cannot ask anyone questions.
- A copy of yourself started with `system_prompt` or `profile` does not see your context files.

| Agent | Use it for |
| --- | --- |
| _(this agent, as a fresh copy)_ | _(independent review, long reading)_ |
