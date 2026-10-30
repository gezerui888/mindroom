# TOOLS.md

Mechanics of this setup.

## Connected accounts

On this private agent every account connection belongs to the person, so anything done through one is done as them.
List each service here when the person connects it, and remove it when they disconnect.

| Service | What it reaches |
| --- | --- |

## Workers and the workspace

Tools such as `shell` and `file` run in a worker that MindRoom may stop or recreate.
Only the workspace is durable: in a worker, the working directory, `~`, and relative paths all point into it.
Anything outside it, such as `/tmp`, the worker's own Python environment, and running or background shells, can vanish with the worker.
Install extra command-line tools under the workspace, for example in `~/.local/bin`.
Put them on `PATH` with an `export` line in `.mindroom/worker-env.sh`, which runs before each worker shell or Python call.

## Skills

Workspace skills live in `skills/<name>/SKILL.md` and load on the next run.
Platform-wide skills are read-only from here, and a workspace skill with the same name would replace the shared one for this person, so a personal variant takes a distinct name.

## Replies

Answer in the current thread as usual, because the reply is delivered there.
Reserve the Matrix message tool for Matrix actions the person requested, such as posting in another room, reacting, or editing a message.

## Agreed reminders

`list_schedules` shows what is live, and this table records what the person agreed to.

| Task ID | Covers | Cadence | Silent |
| --- | --- | --- | --- |
