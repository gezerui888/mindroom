---
icon: lucide/user-pen
---

# Renaming an Agent's Matrix Account

`mindroom agents rename-account` moves MindRoom's state over to a new Matrix ID after your homeserver renamed an agent's account in place.
The agent's name in `config.yaml` does not change.
Only its Matrix ID does, for example from `@mindroom_openclaw:example.org` to `@mindroom_harry:example.org`.

MindRoom keeps credentials, workers, journal rows, and other state under the Matrix ID.
After a server-side rename it would look for all of them under the new ID and find nothing.
The command moves them, rewrites what holds the old ID, and tells you what it left alone.

## When to use it

Run it after the homeserver renamed the account, and before MindRoom starts for the first time under the new ID.

- The MindRoom fork of Tuwunel renames a local user in place with the admin command `query user-id-rewrite` ([mindroom-tuwunel#109](https://github.com/mindroom-ai/mindroom-tuwunel/pull/109)). Its operator guide is `docs/user-id-rewrite.md` in that repository. The account keeps its password, rooms, and history.
- Synapse has no equivalent admin command, so this page does not apply to a Synapse server.
- The new ID must be on the same server as the old one. A rename in place keeps the server, and the command refuses another one.

If MindRoom already started under the new ID, it created fresh state there.
Add `--discard-new-principal` to drop that state when it holds no replies or credentials.
[New-ID state](#new-id-state) explains what it deletes and when it still refuses.

The command supports the SQLite event journal and the `static_runner` or `docker` worker backend.
Any other setup is refused with `not supported yet`.

## Procedure

1. Rename the account on the homeserver and confirm the new ID can log in with the agent's old password.
2. Stop MindRoom.
3. Take a snapshot or copy of the storage directory, the config directory, and the homeserver's database.
   The command keeps its own backups, but a snapshot is the only rollback that also covers the homeserver.
4. Run a dry run with the same `--config` and `--storage-path` the service uses.
   It prints the plan and changes nothing.

   ```bash
   mindroom agents rename-account harry @mindroom_harry:example.org \
     --config config.yaml --storage-path mindroom_data
   ```

5. Read the plan, then apply it.
   The command asks `Apply this plan?` unless you pass `--yes`.

   ```bash
   mindroom agents rename-account harry @mindroom_harry:example.org \
     --config config.yaml --storage-path mindroom_data --apply
   ```

6. Start MindRoom.
   The agent logs in with its stored password on a new device and syncs under the new ID.

A dry run neither contacts the homeserver nor checks that MindRoom is stopped.
`--apply` takes the locks a running MindRoom holds, so it refuses to run while MindRoom is up and blocks MindRoom from starting until it exits.
It then checks the login (see below) and plans again under the locks.
Some event journal collisions only show up in that second plan.
If one does, the command changes nothing and refuses.

Running the same command again after a successful apply prints `Already adopted; nothing to do.` and exits 0.

## What changes

| Where | Change |
| --- | --- |
| `matrix_state.yaml`, account `agent_<name>` | The `username` becomes the new localpart. The old `device_id` and `access_token` are cleared so the next start logs in on a fresh device. The password stays. |
| Event journal, `tracking/event_journal.db` | In one transaction: the principal `<name>@<old ID>` becomes `<name>@<new ID>`, the old ID is replaced in every other text column, the principal's sync consumer and approval grants are deleted, and leftover old-ID `room_member_joins` rows are deleted. |
| `private_oauth/` | The directory of the old ID moves to the directory of the new ID. The worker key inside each OAuth credential store is rewritten. |
| `workers/` | Each worker whose key names the old ID as requester moves to the directory of its new key. Its `metadata/worker.json` and OAuth stores are rewritten. Docker control records and config projections of the old key are removed. |
| `private_instances/` and `tracking/private_instances/` | Private scopes owned by the old ID move to the scope of the new ID, and their owner record is rewritten. |
| `tracking/agents/*/personal_rooms/*.json` | The old ID is replaced in every record. A record owned by the old ID moves to the file name of the new ID. |
| `control_state/matrix_message_sends/` | Files holding send intents of the old ID are removed into the backup, because MindRoom refuses to replay an intent as another sender. |
| `config.yaml` and the YAML files it includes | Every string value equal to the old ID, such as entries in user lists, `bot_accounts`, or `authorization.aliases`, becomes the new ID. |

IDs are matched as whole user IDs the way Tuwunel matches them.
`@harry:example.org` is found in `ask @harry:example.org.` and in `@harry:example.org:DEVICEID`.
It is not found in `@harry:example.org.uk` or `@harry:example.org:8448`, which name other users.

The config edit is made on the text, so comments, quoting, and indentation survive.
The command parses the edited files again and refuses unless they differ from the originals only in those values.
It never edits mapping keys, `.env`, or other files.

Anything that went wrong while rewriting the journal rolls the whole transaction back.
A row that would collide with a row already holding the new ID, in any table but `room_member_joins`, makes the command refuse and name the table and column.

The old Docker worker container is removed last, after every other step succeeded.
If that fails, the command prints a warning and you remove the container by hand.

### What stays

These stay on purpose, and the plan lists them as `report:` lines.

- `encryption_keys/<old ID>/` stays. Only that crypto store decrypts the old encrypted history, and the new device starts its own store. It cannot read encrypted history from before the rename.
- A desktop bridge pinned to the old ID stays pinned. Pair it again.
- Records of past tool jobs and scheduled runs outside the event journal keep the old ID as history.
- A worker directory that looks like it belonged to the old ID, but whose worker key cannot be found, stays in place and is reported with its path.

### New-ID state

When MindRoom ran after the server rename, it already wrote state under the new ID.
Without `--discard-new-principal` the command refuses and names it, either the journal tables with their row counts or the path that already exists.

With `--discard-new-principal`:

- The journal rows under `<name>@<new ID>` are deleted first.
- A destination path that holds only OAuth stores without credentials, lock files, and empty directories is moved into the backup directory, then the old path moves in.

It still refuses, and changes nothing, when the new principal has rows in `reply_messages` or `matrix_delivery_outbox`, because those are replies MindRoom already sent under the new ID.
It also refuses when a destination holds credentials or any other file.
Handle those by hand, because deleting them could lose track of messages already in your rooms.
When the new-ID state holds nothing you need, move that path away yourself and run the command again.

## Checks before it changes anything

Every refusal exits 1 with the reason, and nothing has changed yet.

- The config exists and loads, `AGENT` is an agent in it, and `matrix_state.yaml` holds the account `agent_<name>`.
- The new ID is a valid Matrix user ID on the old ID's server, and is not another stored account.
- The journal backend and worker backend are supported.
- MindRoom is stopped: `Stop MindRoom first: another process holds <lock file>`.
- The login check passes (`--apply` only). The command logs in at `/_matrix/client/v3/login` on the configured homeserver as the new ID with the stored password, then logs out. A rejected login means the server rename has not happened, and the command refuses with `The server does not accept <new ID> with the stored password`.
- Event journal rows and paths under the new ID are dealt with as described in [New-ID state](#new-id-state).

The login check uses plain HTTP and creates a device named `mindroom rename check`.
If logging that device out fails, the command warns and you remove the device from the account by hand.
Pass `--skip-login-check` when the homeserver is unreachable from where you run the command, or when `matrix_state.yaml` holds no password for the agent.

## What it reports

With `--apply` the command prints, in order:

1. The plan: one line per step, then the `report:` lines.
2. `Adopted: agent <name> now uses <new ID>.`
3. `Backups:` and the path of the backup directory.
4. A `Warning:` line for each problem that does not undo the rename, such as an old Docker worker that could not be removed.
5. A leftover scan of the storage directory.

The leftover scan looks for the old ID in file and directory names, in JSON and YAML files, and in every text cell of the event journal.
It skips `encryption_keys`, the backup directory, and the package and cache directories of workers.
It lists what it found in two groups:

- `Reported above as staying` holds places the plan said it would leave, such as the desktop bridge pin.
- `Not reported by the plan, or not scanned` holds everything else, including files too large or unreadable to scan.

Read the second group with care.
It is where state the command did not know about shows up.

| Exit code | Meaning |
| --- | --- |
| `0` | Applied, nothing to do, or a dry run. |
| `1` | Refused, failed with every step undone, or you declined the confirmation. |
| `2` | A step failed and undoing the earlier steps failed too. |

## When a step fails

If a step fails, the command undoes the completed steps in reverse order.
It prints `Not applied:` with the failing step and exits 1.
The disk is back as it was, apart from the backup directory.

If an undo fails as well, the command prints `Undo failed:` and exits 2.
Do not start MindRoom.
Restore the remaining steps by hand from the manifest it names, as described below, or restore your snapshot.

## Restoring from `agent-rename-backups`

Every apply writes a backup directory, even when it succeeds:

```text
<storage>/agent-rename-backups/<UTC timestamp>-rename-account/
```

For example `mindroom_data/agent-rename-backups/20261019T214558Z-rename-account/`.
The command never deletes it.
It can hold copies of credentials and the config, so it is readable by its owner only.
Delete it once you are sure of the result.

`manifest.json` lists every step in the order it ran.
Each entry has a `type`, a `description`, a `source`, a `destination`, and a `backup` directory.
The `backup` directories are numbered `000`, `001`, and so on, and hold the originals.
Plain moves keep nothing there, because a move is only a rename.

To restore by hand, stop MindRoom and walk the manifest from its last entry to its first:

| `type` | Restore |
| --- | --- |
| `MovePath` | Move `destination` back to `source`. |
| `SqliteEdit` | Delete `<source>-wal` and `<source>-shm` if they exist, then copy the database from `backup` over `source`. |
| `FileRewrite` | Copy the file from `backup` over `source`. |
| `DropPath` | Move the item from `backup` back to `source`. |

A `backup` directory holds the file under its original name.

Two things the backup does not cover.
A removed Docker worker container is not restored, and MindRoom creates a new one when the agent needs it.
The homeserver rename is not undone either, so after restoring the files MindRoom is back on the old ID while the server still knows the new one.
Restore the homeserver's snapshot too, or run the command again once the files and the server agree.

See also [Data Storage & Journal](storage.md) for the files this command touches and [CLI Reference](../cli.md#agents-rename-account) for its options.
