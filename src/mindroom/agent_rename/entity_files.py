"""Plan the file changes for renaming an agent's config key.

Stores are named or keyed after the agent's name: worker keys end with it, primary-runtime OAuth stores sit in
directories hashed from it, and each store's binding records it. After the rename MindRoom looks for all of them
under the new name, so each moves there and its contents follow. Names match exactly, never as substrings,
so renaming ``code`` leaves ``code_team`` alone.
Building the steps only reads; every refusal happens here, before anything is written.
"""

from __future__ import annotations

import os
from functools import partial
from typing import TYPE_CHECKING

from mindroom.agent_rename.steps import SqliteEdit
from mindroom.agent_rename.workers import move_steps, oauth_worker_keys, rewrite_oauth_worker_keys
from mindroom.credentials.manager import PRIMARY_RUNTIME_SCOPED_CREDENTIALS_DIRNAME, scoped_credentials_dir_part
from mindroom.tool_system.worker_routing import normalize_worker_key_part

if TYPE_CHECKING:
    import sqlite3

    from mindroom.agent_rename.steps import Step
    from mindroom.constants import RuntimePaths

_AGENT_SCOPES = frozenset({"shared", "unscoped"})


def rewrite_worker_key_agent(worker_key: str, *, old_name: str, new_name: str) -> str | None:
    """Return ``worker_key`` with its agent changed from ``old_name`` to ``new_name``.

    Agent keys read ``v1:<tenant>:shared:<agent>``, ``v1:<tenant>:unscoped:<agent>`` and
    ``v1:<tenant>:user_agent:<requester>:<agent>``; requesters contain ``:``, so the agent is the last component.
    The agent is spelled by ``normalize_worker_key_part``, like ``resolve_worker_key`` does.
    Keys of any other agent or scope give ``None``.
    """
    parts = worker_key.split(":")
    if len(parts) < 4 or parts[0] != "v1" or parts[-1] != normalize_worker_key_part(old_name):
        return None
    scope = parts[2]
    if not ((scope in _AGENT_SCOPES and len(parts) == 4) or (scope == "user_agent" and len(parts) >= 5)):
        return None
    return ":".join([*parts[:-1], normalize_worker_key_part(new_name)])


def _rename_oauth_binding(connection: sqlite3.Connection, *, old_name: str, new_name: str) -> None:
    rewrite_oauth_worker_keys(
        connection,
        rewrite=partial(rewrite_worker_key_agent, old_name=old_name, new_name=new_name),
    )
    # The binding holds the agent's own name (oauth/credential_store.py's _scope_binding), not the key's spelling.
    connection.execute(
        "UPDATE oauth_credential_state SET routing_agent_name = ? WHERE routing_agent_name = ?",
        (new_name, old_name),
    )


def entity_credential_steps(runtime_paths: RuntimePaths, *, old_name: str, new_name: str) -> list[Step]:
    """Return the steps that move the agent's primary-runtime OAuth directories to the new name and rebind their stores.

    Covers ``private_oauth/_agents/<part(OLD)>`` (agent-scoped credentials) and every requester's
    ``private_oauth/<requester>/<part(OLD)>`` (requester-agent credentials), with ``scoped_credentials_dir_part``.
    Each moved OAuth store gets its worker key and routing agent renamed, so it opens for the new agent.
    Refuses when a destination already exists.
    """
    oauth_root = runtime_paths.storage_root / PRIMARY_RUNTIME_SCOPED_CREDENTIALS_DIRNAME
    if not oauth_root.is_dir():
        return []
    old_part, new_part = scoped_credentials_dir_part(old_name), scoped_credentials_dir_part(new_name)
    steps: list[Step] = []
    for scope in sorted(oauth_root.iterdir()):
        source, destination = scope / old_part, scope / new_part
        if not os.path.lexists(source):
            continue
        steps += move_steps(source, destination, f"the OAuth credentials of {old_name}", discard_new_state=False)
        for store in sorted(source.glob("*_credentials.sqlite3")):
            if not oauth_worker_keys(store):
                continue
            database = destination / store.name
            steps.append(
                SqliteEdit(
                    database,
                    partial(_rename_oauth_binding, old_name=old_name, new_name=new_name),
                    f"Rename agent {old_name} to {new_name} in the binding of the OAuth store {database}",
                ),
            )
    return steps
