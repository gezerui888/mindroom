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

On its first conversation the agent works through `BOOTSTRAP.md`: a `project` agent asks the room for its operators, scope, and sources, and a `work` agent asks the person about their role and priorities.
It records the answers in its own files and then deletes `BOOTSTRAP.md`.

The `work` block admits only MindRoom's administrators until you uncomment its `access.users` line, for example `"@*:example.org"` for everyone on your homeserver.
The `project` block admits members of the agent's room, the [default](https://docs.mindroom.chat/authorization/#responder-access) when `access` is omitted.

## Files

| File | Role | When it reaches the agent |
| --- | --- | --- |
| `AGENTS.md` | Operating rules | Every turn, as a context file |
| `SOUL.md` | Purpose and tone | Every turn, as a context file |
| `IDENTITY.md` | Name, scope, and, for `project`, operators and source precedence | Every turn, as a context file |
| `TOOLS.md` | Tool notes, connected accounts, and agreed schedules | Every turn, as a context file |
| `USER.md` (`work` only) | The person's profile | Every turn, as a context file |
| `MEMORY.md` | Curated long-term memory | Every turn, through [file memory](https://docs.mindroom.chat/memory/) |
| `memory/` | `project`: source register, people, decisions, questions, tracker, and sandbox friction log; `work`: daily notes the agent creates | When the agent reads or searches them |
| `BOOTSTRAP.md` | First-run checklist | Read while setup is unfinished |
| `skills/` | `project`: `status-report` and `meeting-notes-to-updates`; `work`: `meeting-prep` and `weekly-review` | Name and description every turn, full text when used ([Skills](https://docs.mindroom.chat/skills/)) |

Edits to context files apply from the agent's next reply.
`MEMORY.md` is not listed in `context_files`, because file memory already loads it and listing it would load it twice.

## Updating and Rerunning

Running the command again restores files that are missing and keeps every existing file.
A restored `BOOTSTRAP.md` is harmless: the agent sees that setup already happened, deletes it, and carries on.
For `work`, MindRoom copies template files that a person's private copy is missing on later runs too, so edits to `templates/<name>/` reach existing people only as new files.
See [Private Instances](https://docs.mindroom.chat/configuration/agents/#private-instances) for where each person's copy lives.

## Notes

- A `project` agent has one workspace, so its direct chats write to the same files as its room.
- `file_access: workspace` confines the `file` tool but not `shell`; when people you do not fully trust can reach the agent, run `shell` in a worker with `worker_tools`.
- Use [personal agents](https://docs.mindroom.chat/personal-agents/) instead of `work` when each person needs an agent of their own that others can mention or add to a team.
- The `project` command must run where the storage root lives, because it writes the workspace directly.
