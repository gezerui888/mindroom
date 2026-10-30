---
icon: lucide/layout-template
---

# Agent Templates

`mindroom config new-agent` creates an agent's starter files from a built-in template and prints the config block to add under `agents:`.
It never edits `config.yaml` and never overwrites a file that already exists.

| Template | For | Files go to | Each person gets |
| --- | --- | --- | --- |
| `project` | One shared agent for a project or team, used by several people in one room | The agent's workspace, `<storage>/agents/<name>/workspace/` | The same workspace as everyone else |
| `work` | A private work assistant for each person | A template directory, `<config dir>/templates/<name>/` | A private copy, made the first time they talk to the agent |

## Setup

```bash
mindroom config new-agent apollo --template project
mindroom config new-agent assistant --template work
```

1. Run the command with the agent name you want; names use letters, digits, and underscores.
2. Paste the printed block into the `agents:` section of `config.yaml` and change what you need, such as `display_name`, `model`, or `tools`.
3. Run `mindroom config validate`.
4. Talk to the agent: a `project` agent joins the room named after it, which MindRoom creates, and a `work` agent answers direct messages and invites.

On its first conversation the agent works through `BOOTSTRAP.md` and records the answers in its own files.
A `project` agent asks the room for its operators, scope, and sources, then deletes `BOOTSTRAP.md`.
A `work` agent asks the person about their role and priorities and fills in `USER.md`; its `BOOTSTRAP.md` stays, because MindRoom restores missing template files before every reply, and the agent ignores it once `USER.md` is filled in.

The `work` block admits only MindRoom's administrators until you uncomment its `access.users` line, for example `"@*:example.org"` for everyone on your homeserver.
The `project` block admits members of the agent's room, the [default](../authorization.md#responder-access) when `access` is omitted.

## Files

| File | Role | When it reaches the agent |
| --- | --- | --- |
| `AGENTS.md` | Operating rules | Every turn, as a context file |
| `SOUL.md` | Purpose and tone | Every turn, as a context file |
| `IDENTITY.md` | Name, scope, and, for `project`, operators and source precedence | Every turn, as a context file |
| `TOOLS.md` | Tool notes, connected accounts, and agreed schedules | Every turn, as a context file |
| `USER.md` (`work` only) | The person's profile | Every turn, as a context file |
| `MEMORY.md` | Curated long-term memory | Every turn, through [file memory](../memory.md) |
| `memory/` | `project`: source register, people, decisions, questions, tracker, and sandbox friction log; `work`: daily notes the agent creates | When the agent reads or searches them |
| `BOOTSTRAP.md` | First-run checklist | Read while setup is unfinished |
| `skills/` | `project`: `status-report` and `meeting-notes-to-updates`; `work`: `meeting-prep` and `weekly-review` | Name and description every turn, full text when used ([Skills](../skills.md)) |

`MEMORY.md` is not listed in `context_files`, because file memory already loads it and listing it would load it twice.

## Updating and Rerunning

Running the command again restores files that are missing and keeps every existing file.
A `project` agent that finds a restored `BOOTSTRAP.md` after setup deletes it without starting over.
For `work`, edits to `templates/<name>/` reach existing people only as new files, because MindRoom never overwrites a person's copy; see [Private Instances](agents.md#private-instances).

## Notes

- A `project` agent has one workspace, so its direct chats write to the same files as its room.
- The `file` tool stays inside the workspace, but `shell` does not; with a [worker backend](../deployment/sandbox-proxy.md), list `shell` in `worker_tools` when people you do not fully trust can reach the agent.
- The `project` command must run where the storage root lives, because it writes the workspace directly.
