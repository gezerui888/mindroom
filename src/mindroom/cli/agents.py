"""`mindroom agents` commands that change an agent's identity on disk."""

from __future__ import annotations

import shlex
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.markup import escape

from mindroom.constants import matrix_state_file, runtime_matrix_homeserver, runtime_matrix_ssl_verify

from .config import activate_cli_runtime, console, format_validation_errors, load_config_quiet

if TYPE_CHECKING:
    from collections.abc import Iterator

    import httpx

    from mindroom.agent_rename.steps import ApplyResult, RenamePlan
    from mindroom.config.main import Config
    from mindroom.constants import RuntimePaths
    from mindroom.matrix.state import MatrixAccount

_LOGIN_CHECK_DEVICE_NAME = "mindroom rename check"
_LOGIN_CHECK_TIMEOUT_SECONDS = 30.0
_REJECTED_LOGIN_STATUSES = frozenset({401, 403})
_UNDO_FAILED_EXIT_CODE = 2
# The .env variable naming the agent that serves the Connections portal.
_CONNECTIONS_AGENT_VARIABLE = "MINDROOM_CONNECTIONS_AGENT"


@dataclass(frozen=True)
class _Adoption:
    """One agent's stored account and the ID the server renamed it to."""

    agent: str
    old_user_id: str
    new_user_id: str
    stored_user_id: str
    password: str | None


@dataclass(frozen=True)
class _AgentRename:
    """One agent's config key and the key it is renamed to, with the Matrix ID it keeps."""

    old: str
    new: str
    # The stored account's ID, or the ID MindRoom generates for the agent when nothing is stored.
    user_id: str
    has_account: bool


def rename_account(
    agent: str = typer.Argument(..., help="The agent's name in the config."),
    new_user_id: str = typer.Argument(..., help="The Matrix ID the server renamed the agent's account to."),
    from_user_id: str | None = typer.Option(
        None,
        "--from",
        metavar="OLD_USER_ID",
        help=(
            "The Matrix ID the server renamed the account from, when matrix_state.yaml already holds the new one "
            "because its username was changed by hand."
        ),
    ),
    apply: bool = typer.Option(False, "--apply", help="Make the changes; without it the plan is only printed."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Apply without asking for confirmation."),
    discard_new_principal: bool = typer.Option(
        False,
        "--discard-new-principal",
        help=(
            "Delete what MindRoom created under the new ID before its state was moved: "
            "the new principal's event journal rows, only while the old principal still has rows to move, "
            "and new-ID state that holds no credentials."
        ),
    ),
    skip_login_check: bool = typer.Option(
        False,
        "--skip-login-check",
        help="Apply without first logging in as the new ID with the stored password.",
    ),
    config_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--config",
        "-c",
        help="Use this config file path.",
    ),
    storage_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--storage-path",
        "-s",
        help="Base directory for persistent MindRoom data.",
    ),
) -> None:
    """Adopt an agent's Matrix account after the server renamed it.

    Run it with MindRoom stopped, after the homeserver renamed the account in place
    (for example with Tuwunel's `query user-id-rewrite`). It moves credentials, workers,
    and other state kept under the old Matrix ID to the new one, rewrites the event journal,
    and replaces the old ID in the config. The agent's name in the config stays.
    """
    runtime_paths = activate_cli_runtime(path=config_path, storage_path=storage_path)
    config = _load_config_or_exit(runtime_paths)
    with _exit_on_rename_errors():
        _rename_account(
            runtime_paths,
            config,
            agent=agent,
            new_user_id=new_user_id,
            from_user_id=from_user_id,
            apply=apply,
            yes=yes,
            discard_new_principal=discard_new_principal,
            skip_login_check=skip_login_check,
        )


def rename(
    old: str = typer.Argument(..., help="The agent's name in the config."),
    new: str = typer.Argument(..., help="The agent's new name."),
    apply: bool = typer.Option(False, "--apply", help="Make the changes; without it the plan is only printed."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Apply without asking for confirmation."),
    config_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--config",
        "-c",
        help="Use this config file path.",
    ),
    storage_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--storage-path",
        "-s",
        help="Base directory for persistent MindRoom data.",
    ),
) -> None:
    """Rename an agent in the config and carry its state over to the new name.

    Run it with MindRoom stopped. It moves the agent's state, history, memory, team history,
    workers, credentials and tracking records to the new name, renames it in the event journal
    and matrix_state.yaml, and renames it in the config last. Its Matrix account and device stay.
    """
    runtime_paths = activate_cli_runtime(path=config_path, storage_path=storage_path)
    config = _load_config_or_exit(runtime_paths)
    with _exit_on_rename_errors():
        _rename(runtime_paths, config, old=old, new=new, apply=apply, yes=yes)


@contextmanager
def _exit_on_rename_errors() -> Iterator[None]:
    """Print a refused, failed, or half-undone rename and exit with its code: 1, or 2 when the disk needs a restore."""
    from mindroom.agent_rename.steps import (  # noqa: PLC0415 - keeps the rename engine out of CLI import time
        RenameApplyError,
        RenameRefusedError,
        RenameUndoError,
    )

    try:
        yield
    except RenameRefusedError as exc:
        _say(str(exc), label="[red]Refused:[/red] ")
        raise typer.Exit(1) from None
    except RenameApplyError as exc:
        _say(str(exc), label="[red]Not applied:[/red] ")
        raise typer.Exit(1) from None
    except RenameUndoError as exc:
        _say(str(exc), label="[red]Undo failed:[/red] ")
        _say(f"Do not start MindRoom before restoring the remaining steps listed in {exc.manifest} by hand.")
        raise typer.Exit(_UNDO_FAILED_EXIT_CODE) from None


def _say(text: str, *, label: str = "") -> None:
    """Print ``text`` literally after an optional styled ``label``.

    Lines are never wrapped, so paths stay copyable, and ``:name:`` emoji codes stay text:
    Matrix IDs followed by a colon, like ``@old:x:``, contain them.
    """
    console.print(f"{label}{escape(text)}", soft_wrap=True, emoji=False)


def _load_config_or_exit(runtime_paths: RuntimePaths) -> Config:
    """Load the config, or exit with its errors.

    Unlike other commands, a missing config is not seeded from a template: that would be a write,
    and without a config there is no agent to rename.
    """
    from mindroom.config.main import CONFIG_LOAD_USER_ERROR_TYPES  # noqa: PLC0415

    config_path = runtime_paths.config_path
    if not config_path.exists():
        _say(f"No config at {config_path}.", label="[red]Error:[/red] ")
        raise typer.Exit(1)
    try:
        return load_config_quiet(runtime_paths, tolerate_plugin_load_errors=True)
    except CONFIG_LOAD_USER_ERROR_TYPES as exc:
        format_validation_errors(exc, config_path)
        raise typer.Exit(1) from None


def _rename_account(
    runtime_paths: RuntimePaths,
    config: Config,
    *,
    agent: str,
    new_user_id: str,
    from_user_id: str | None,
    apply: bool,
    yes: bool,
    discard_new_principal: bool,
    skip_login_check: bool,
) -> None:
    from mindroom.agent_rename.guard import require_supported_backends, stopped_runtime  # noqa: PLC0415
    from mindroom.credentials.manager import scoped_credentials_dir_part  # noqa: PLC0415
    from mindroom.desktop.native_config import native_config_path  # noqa: PLC0415

    adoption = _adoption(runtime_paths, config, agent=agent, new_user_id=new_user_id, from_user_id=from_user_id)
    if adoption is None:
        _say("Already adopted; nothing to do.")
        return
    require_supported_backends(config, runtime_paths)
    if not apply:
        # Taking MindRoom's locks would create lock files, so a dry run plans without them.
        plan = _plan(runtime_paths, config, adoption, discard_new_principal=discard_new_principal)
        _print_plan(_adoption_header(adoption), plan)
        if not plan.steps:
            _say(f"Nothing left to move from {adoption.old_user_id}; nothing was changed.")
            return
        _say("Dry run: nothing was changed. Rerun with --apply to make these changes.")
        _say(
            "--apply plans again under MindRoom's locks. Event journal collisions in tables other than "
            "room_member_joins only show up when --apply rewrites the journal; "
            "if one does, that rewrite rolls back and the command refuses.",
        )
        return
    with stopped_runtime(runtime_paths):
        if not skip_login_check:
            _check_login(runtime_paths, adoption)
        plan = _plan(runtime_paths, config, adoption, discard_new_principal=discard_new_principal)
        _print_plan(_adoption_header(adoption), plan)
        if not plan.steps:
            # Only with --from: otherwise the stored username still needs its step.
            _say(f"Nothing left to move from {adoption.old_user_id}; nothing was changed.")
            return
        _confirm(yes=yes)
        result = _apply(plan, kind="rename-account")
        _say(f"agent {agent} now uses {adoption.new_user_id}.", label="[green]Adopted:[/green] ")
        _print_applied(result)
        storage = runtime_paths.storage_root
        old_user_id = adoption.old_user_id
        # The plan reports a desktop bridge pinned to the old ID as staying.
        # It reports the old crypto store too, but the scan never enters encryption_keys.
        _print_leftovers(
            storage,
            old_user_id,
            needles=(old_user_id, scoped_credentials_dir_part(old_user_id)),
            expected=frozenset({native_config_path(storage)}),
        )


def _stored_accounts(runtime_paths: RuntimePaths) -> dict[str, MatrixAccount]:
    """Read the stored Matrix accounts; ``MatrixState.load`` can rewrite a legacy file, which a dry run must not."""
    import yaml  # noqa: PLC0415

    from mindroom import yaml_io  # noqa: PLC0415
    from mindroom.agent_rename.steps import RenameRefusedError  # noqa: PLC0415
    from mindroom.matrix.state import MatrixState  # noqa: PLC0415

    path = matrix_state_file(runtime_paths)
    try:
        data = yaml_io.safe_load(path.read_bytes()) if path.exists() else None
        return MatrixState.model_validate(data or {}).accounts
    except (OSError, ValueError, yaml.YAMLError) as exc:
        msg = f"Cannot read {path}: {exc}"
        raise RenameRefusedError(msg) from exc


def _adoption(
    runtime_paths: RuntimePaths,
    config: Config,
    *,
    agent: str,
    new_user_id: str,
    from_user_id: str | None,
) -> _Adoption | None:
    """Check the request and return it, or ``None`` when nothing is left under another ID.

    The stored username alone does not prove an adoption: changing it by hand leaves the journal and files
    under the old ID. So when it already names the new ID, the journal decides, and ``--from`` names the old ID.
    """
    from mindroom.agent_rename.steps import RenameRefusedError  # noqa: PLC0415
    from mindroom.matrix.identity import MatrixID, managed_account_key  # noqa: PLC0415

    state_path = matrix_state_file(runtime_paths)
    accounts = _stored_accounts(runtime_paths)
    domain = config.get_domain(runtime_paths)
    user_ids = {
        key: MatrixID.from_username(account.username, account.domain or domain).full_id
        for key, account in accounts.items()
    }
    account_key = managed_account_key(agent)
    stored_user_id = user_ids.get(account_key)
    # Checked before anything else that can refuse, so a rerun after a successful apply is a no-op.
    if from_user_id is None and stored_user_id == new_user_id:
        _refuse_journal_under_other_ids(runtime_paths, agent=agent, new_user_id=new_user_id)
        return None
    if agent not in config.agents:
        msg = f"{agent} is not an agent in {runtime_paths.config_path}."
        raise RenameRefusedError(msg)
    if stored_user_id is None:
        msg = f"{state_path} holds no account {account_key}, so there is no Matrix account to adopt."
        raise RenameRefusedError(msg)
    old_user_id = _old_user_id(
        state_path,
        account_key,
        stored_user_id=stored_user_id,
        new_user_id=new_user_id,
        from_user_id=from_user_id,
    )
    old_domain = MatrixID.parse(old_user_id).domain
    if MatrixID.parse(new_user_id).domain != old_domain:
        msg = f"{new_user_id} is not on {old_domain}, the server of {old_user_id}; a rename in place keeps the server."
        raise RenameRefusedError(msg)
    owner = _other_owner(user_ids, account_key, new_user_id)
    if owner is not None:
        msg = f"{new_user_id} is already the stored account {owner} in {state_path}."
        raise RenameRefusedError(msg)
    owner = _other_owner(user_ids, account_key, old_user_id)
    if owner is not None:
        msg = f"--from {old_user_id} is the stored account {owner} in {state_path}, not a former ID of {account_key}."
        raise RenameRefusedError(msg)
    if old_user_id != stored_user_id:
        _require_former_user_id(runtime_paths, agent=agent, from_user_id=old_user_id, new_user_id=new_user_id)
    return _Adoption(agent, old_user_id, new_user_id, stored_user_id, accounts[account_key].password)


def _old_user_id(
    state_path: Path,
    account_key: str,
    *,
    stored_user_id: str,
    new_user_id: str,
    from_user_id: str | None,
) -> str:
    """Check both IDs and return the one to adopt from: ``--from`` when given, otherwise the stored account's."""
    from mindroom.agent_rename.steps import RenameRefusedError  # noqa: PLC0415
    from mindroom.matrix.identity import parse_historical_matrix_user_id  # noqa: PLC0415

    for user_id in (new_user_id, from_user_id):
        try:
            if user_id is not None:
                parse_historical_matrix_user_id(user_id)
        except ValueError as exc:
            msg = f"{user_id} is not a Matrix user ID ({exc})."
            raise RenameRefusedError(msg) from exc
    if from_user_id is None:
        return stored_user_id
    if from_user_id == new_user_id:
        msg = f"--from {from_user_id} is the new ID itself; pass the ID the server renamed the account from."
        raise RenameRefusedError(msg)
    if stored_user_id not in (from_user_id, new_user_id):
        msg = (
            f"--from {from_user_id} does not match the stored account {stored_user_id} of {account_key} "
            f"in {state_path}; --from is for a stored account that already holds {new_user_id}."
        )
        raise RenameRefusedError(msg)
    return from_user_id


def _require_former_user_id(runtime_paths: RuntimePaths, *, agent: str, from_user_id: str, new_user_id: str) -> None:
    """Refuse a ``--from`` that nothing shows to be a former Matrix ID of ``agent``.

    ``--from`` moves whatever is stored under that ID, so a wrong one, such as a person's, would hand their
    OAuth credentials and journal mentions to the agent. Accepted are an ID the journal keys the agent's rows by,
    and the ID MindRoom generates for the agent on the new ID's server, which lets a killed run resume.
    """
    from mindroom.agent_rename.journal import other_principal_user_ids  # noqa: PLC0415
    from mindroom.agent_rename.steps import RenameRefusedError  # noqa: PLC0415
    from mindroom.event_journal.opening import event_journal_sqlite_path  # noqa: PLC0415
    from mindroom.matrix.identity import MatrixID  # noqa: PLC0415
    from mindroom.matrix_identifiers import agent_username_localpart  # noqa: PLC0415

    database = event_journal_sqlite_path(runtime_paths.storage_root)
    if from_user_id in other_principal_user_ids(database, agent=agent, user_id=new_user_id):
        return
    generated = MatrixID.from_username(
        agent_username_localpart(agent, runtime_paths),
        MatrixID.parse(new_user_id).domain,
    ).full_id
    if from_user_id == generated:
        return
    msg = (
        f"--from {from_user_id} must be a former Matrix ID of agent {agent}, and nothing shows that it is: "
        f"the event journal {database} has no rows for {agent}@{from_user_id}, "
        f"and it is not {generated}, the ID MindRoom generates for {agent}. "
        "If an earlier run with this --from was killed after its journal step, restore from its backup manifest "
        "before running again."
    )
    raise RenameRefusedError(msg)


def _other_owner(user_ids: dict[str, str], account_key: str, user_id: str) -> str | None:
    """Return the stored account other than ``account_key`` that has ``user_id``, if any."""
    return next((key for key, stored in user_ids.items() if key != account_key and stored == user_id), None)


def _refuse_journal_under_other_ids(runtime_paths: RuntimePaths, *, agent: str, new_user_id: str) -> None:
    """Refuse when the journal still keys ``agent``'s rows by an ID other than the stored ``new_user_id``.

    That is what changing the stored username by hand leaves behind: MindRoom then runs as the new ID,
    and its sync fails on the old rows with ``IngestionBatchIntegrityError``.
    """
    from mindroom.agent_rename.journal import other_principal_user_ids  # noqa: PLC0415
    from mindroom.agent_rename.steps import RenameRefusedError  # noqa: PLC0415
    from mindroom.event_journal.opening import event_journal_sqlite_path  # noqa: PLC0415
    from mindroom.matrix.identity import MatrixID, try_parse_historical_matrix_user_id  # noqa: PLC0415

    database = event_journal_sqlite_path(runtime_paths.storage_root)
    server = MatrixID.parse(new_user_id).domain
    # Only an ID on the new ID's server can have been renamed in place, and --from refuses any other.
    others = {
        user_id: counts
        for user_id, counts in other_principal_user_ids(database, agent=agent, user_id=new_user_id).items()
        if (parsed := try_parse_historical_matrix_user_id(user_id)) is not None
        and MatrixID.parse(parsed).domain == server
    }
    if not others:
        return
    found = "; ".join(f"{agent}@{user_id} ({counts})" for user_id, counts in others.items())
    storage_options = ["--config", str(runtime_paths.config_path), "--storage-path", str(runtime_paths.storage_root)]
    commands = (
        shlex.join(["mindroom", "agents", "rename-account", agent, new_user_id, "--from", user_id, *storage_options])
        for user_id in others
    )
    msg = (
        f"{matrix_state_file(runtime_paths)} already names {new_user_id} for agent_{agent}, "
        f"but the event journal {database} still has rows for {found}, so its state was never moved. "
        f"If the server renamed that account to {new_user_id}, run this to move its state:"
        + "".join(f"\n  {command}" for command in commands)
    )
    raise RenameRefusedError(msg)


def _check_login(runtime_paths: RuntimePaths, adoption: _Adoption) -> None:
    """Refuse unless the homeserver accepts the new ID with the stored password, then log that session out again.

    Plain HTTP on purpose: MindRoom's Matrix login creates a local crypto store for the ID,
    which would collide with the one MindRoom creates when it logs in after the rename.
    """
    import httpx  # noqa: PLC0415

    from mindroom.agent_rename.steps import RenameRefusedError  # noqa: PLC0415

    new_user_id = adoption.new_user_id
    if adoption.password is None:
        msg = (
            f"{matrix_state_file(runtime_paths)} holds no password for agent_{adoption.agent}, so the login check "
            "cannot run; pass --skip-login-check once the server renamed the account."
        )
        raise RenameRefusedError(msg)
    homeserver = runtime_matrix_homeserver(runtime_paths).rstrip("/")
    login = {
        "type": "m.login.password",
        "identifier": {"type": "m.id.user", "user": new_user_id},
        "password": adoption.password,
        "initial_device_display_name": _LOGIN_CHECK_DEVICE_NAME,
    }
    verify = runtime_matrix_ssl_verify(runtime_paths=runtime_paths)
    with httpx.Client(verify=verify, timeout=_LOGIN_CHECK_TIMEOUT_SECONDS) as client:
        try:
            response = client.post(f"{homeserver}/_matrix/client/v3/login", json=login)
        except httpx.HTTPError as exc:
            msg = f"Cannot reach {homeserver} for the login check ({exc}); retry, or pass --skip-login-check."
            raise RenameRefusedError(msg) from exc
        if response.status_code in _REJECTED_LOGIN_STATUSES:
            msg = (
                f"The server does not accept {new_user_id} with the stored password; "
                "rename the account on the server first."
            )
            raise RenameRefusedError(msg)
        if not response.is_success:
            msg = f"The login check for {new_user_id} failed with HTTP {response.status_code}; retry, or pass --skip-login-check."
            raise RenameRefusedError(msg)
        _log_out(client, homeserver, response)


def _log_out(client: httpx.Client, homeserver: str, login: httpx.Response) -> None:
    """End the login check's session; if that fails, only a stray device stays behind, so it warns."""
    import httpx  # noqa: PLC0415

    try:
        token = login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        client.post(f"{homeserver}/_matrix/client/v3/logout", headers=headers).raise_for_status()
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        _say(
            f"Could not log the login check out ({exc}); "
            f"remove its device {_LOGIN_CHECK_DEVICE_NAME!r} from the account by hand.",
            label="[yellow]Warning:[/yellow] ",
        )


def _plan(
    runtime_paths: RuntimePaths,
    config: Config,
    adoption: _Adoption,
    *,
    discard_new_principal: bool,
) -> RenamePlan:
    from mindroom.agent_rename.account_files import account_file_steps, matrix_state_steps  # noqa: PLC0415
    from mindroom.agent_rename.config_edit import config_value_steps  # noqa: PLC0415
    from mindroom.agent_rename.journal import journal_account_steps  # noqa: PLC0415
    from mindroom.agent_rename.steps import FinalAction, RenamePlan  # noqa: PLC0415
    from mindroom.event_journal.opening import event_journal_sqlite_path  # noqa: PLC0415

    old_user_id, new_user_id = adoption.old_user_id, adoption.new_user_id
    account_steps, account_reports = matrix_state_steps(
        runtime_paths,
        agent=adoption.agent,
        old_user_id=old_user_id,
        new_user_id=new_user_id,
    )
    journal_steps, journal_reports = journal_account_steps(
        event_journal_sqlite_path(runtime_paths.storage_root),
        agent=adoption.agent,
        old_user_id=old_user_id,
        new_user_id=new_user_id,
        discard_new_principal=discard_new_principal,
        # Without a step for matrix_state.yaml the account keeps the device it uses under the new ID.
        new_device_kept=not account_steps,
    )
    file_steps, reports = account_file_steps(
        runtime_paths,
        agent=adoption.agent,
        old_user_id=old_user_id,
        new_user_id=new_user_id,
        config=config,
        discard_new_identity_state=discard_new_principal,
    )
    config_steps = config_value_steps(runtime_paths.config_path, old_value=old_user_id, new_value=new_user_id)
    final_steps = [step for step in file_steps if isinstance(step, FinalAction)]
    # The journal edit goes first: some of its collisions only show up when it runs,
    # and a failing first step leaves nothing to undo.
    # The username goes last among the reversible steps: it is what marks the account adopted,
    # so a run killed before it never looks finished. Final actions always run after every reversible step.
    steps = (
        *journal_steps,
        *(step for step in file_steps if not isinstance(step, FinalAction)),
        *config_steps,
        *account_steps,
        *final_steps,
    )
    return RenamePlan(
        storage_root=runtime_paths.storage_root,
        steps=steps,
        reports=(*journal_reports, *account_reports, *reports),
    )


def _adoption_header(adoption: _Adoption) -> str:
    if adoption.stored_user_id == adoption.old_user_id:
        detail = f"whose stored account is {adoption.old_user_id}:"
    else:
        detail = f"from {adoption.old_user_id}; its stored account already is {adoption.new_user_id}:"
    return f"Plan to adopt {adoption.new_user_id} for agent {adoption.agent}, {detail}"


def _rename(runtime_paths: RuntimePaths, config: Config, *, old: str, new: str, apply: bool, yes: bool) -> None:
    from mindroom.agent_rename.guard import require_supported_backends, stopped_runtime  # noqa: PLC0415

    rename = _agent_rename(runtime_paths, config, old=old, new=new)
    require_supported_backends(config, runtime_paths)
    if not apply:
        # Taking MindRoom's locks would create lock files, so a dry run plans without them.
        _print_plan(_rename_header(rename), _checked_rename_plan(runtime_paths, config, rename))
        _say("Dry run: nothing was changed. Rerun with --apply to make these changes.")
        return
    with stopped_runtime(runtime_paths):
        plan = _checked_rename_plan(runtime_paths, config, rename)
        _print_plan(_rename_header(rename), plan)
        _confirm(yes=yes)
        result = _apply(plan, kind="rename")
        _say(f"agent {old} is now {new}.", label="[green]Renamed:[/green] ")
        _print_applied(result)
        storage = runtime_paths.storage_root
        _print_leftovers(storage, f"agent {old}", names=(old,), expected=_reported_paths(plan.reports, storage))


def _agent_rename(runtime_paths: RuntimePaths, config: Config, *, old: str, new: str) -> _AgentRename:
    """Check that agent ``old`` can be renamed to ``new``, and return the rename with the Matrix ID the agent keeps."""
    from mindroom.agent_rename.steps import RenameRefusedError  # noqa: PLC0415
    from mindroom.matrix.identity import MatrixID, managed_account_key  # noqa: PLC0415
    from mindroom.matrix_identifiers import agent_username_localpart  # noqa: PLC0415

    _require_renamable_agent(runtime_paths, config, old)
    _require_new_name(config, old=old, new=new)
    accounts = _stored_accounts(runtime_paths)
    domain = config.get_domain(runtime_paths)
    account = accounts.get(managed_account_key(old))
    if account is None:
        username, account_domain = agent_username_localpart(old, runtime_paths), domain
    else:
        username, account_domain = account.username, account.domain or domain
    rename = _AgentRename(old, new, MatrixID.from_username(username, account_domain).full_id, account is not None)
    new_account = managed_account_key(new) in accounts
    # A run killed before its config edit has already written the new account, journal principal or state root.
    _refuse_unfinished_rename(runtime_paths, rename, new_account=new_account)
    if new_account:
        msg = (
            f"{matrix_state_file(runtime_paths)} already holds an account {managed_account_key(new)}, "
            f"left by an agent named {new}. Choose another new name."
        )
        raise RenameRefusedError(msg)
    if not rename.has_account:
        _refuse_journal_without_account(runtime_paths, old)
    return rename


def _require_renamable_agent(runtime_paths: RuntimePaths, config: Config, old: str) -> None:
    from mindroom.agent_rename.steps import RenameRefusedError  # noqa: PLC0415
    from mindroom.constants import ROUTER_AGENT_NAME  # noqa: PLC0415

    if old == ROUTER_AGENT_NAME:
        msg = f"{old} is MindRoom's router, which cannot be renamed."
    elif old in config.teams:
        msg = f"{old} is a team; renaming teams is not supported yet."
    elif old not in config.agents:
        msg = f"{old} is not a configured agent in {runtime_paths.config_path}."
    elif config.agents[old].private is not None:
        msg = f"agent {old} is private (agents.{old}.private is set); renaming private agents is not supported yet."
    else:
        return
    raise RenameRefusedError(msg)


def _require_new_name(config: Config, *, old: str, new: str) -> None:
    """Refuse a new name the config would reject, or one MindRoom would mistake for another agent's.

    Worker keys spell an agent with ``normalize_worker_key_part``, and team runs tell members apart by Agno's
    ``url_safe_string`` id, so a name that matches the old one or another agent there would share their state.
    """
    from agno.utils.string import url_safe_string  # noqa: PLC0415

    from mindroom.agent_rename.steps import RenameRefusedError  # noqa: PLC0415
    from mindroom.config.main import AGENT_NAME_PATTERN, RESERVED_ENTITY_NAMES  # noqa: PLC0415
    from mindroom.tool_system.worker_routing import normalize_worker_key_part  # noqa: PLC0415

    worker_spelling = normalize_worker_key_part(new)
    worker_twin = next((name for name in config.agents if normalize_worker_key_part(name) == worker_spelling), None)
    agent_id = url_safe_string(new)
    id_twin = next((name for name in config.agents if name != old and url_safe_string(name) == agent_id), None)
    if new == old:
        msg = f"{new} is the agent's name already."
    elif not AGENT_NAME_PATTERN.fullmatch(new):
        msg = f"{new} is not a valid agent name; use letters, digits and _ only."
    elif new in RESERVED_ENTITY_NAMES:
        msg = f"{new} is reserved for MindRoom."
    elif new in config.agents:
        msg = f"{new} is already a configured agent."
    elif new in config.teams:
        msg = f"{new} is already a configured team."
    elif worker_twin is not None:
        msg = (
            f"{new} is written {worker_spelling} in worker keys, like agent {worker_twin}, so they would share workers."
        )
    elif id_twin is not None:
        msg = f"{new} has the id {agent_id}, like agent {id_twin}; team runs tell their members apart by that id."
    else:
        return
    raise RenameRefusedError(msg)


def _journal_principal_ids(database: Path, agent: str) -> dict[str, str]:
    """Return every Matrix ID that ``agent``'s journal principals name, with their row counts."""
    from mindroom.agent_rename.journal import other_principal_user_ids  # noqa: PLC0415

    # No principal is `<agent>@` alone, so every ID counts as other than the empty one.
    return other_principal_user_ids(database, agent=agent, user_id="")


def _refuse_journal_without_account(runtime_paths: RuntimePaths, old: str) -> None:
    """Refuse when the journal has principals of agent ``old`` although no account of it is stored.

    The rename keeps the agent's Matrix account, so it cannot tell which principal the agent keeps.
    """
    from mindroom.agent_rename.steps import RenameRefusedError  # noqa: PLC0415
    from mindroom.event_journal.opening import event_journal_sqlite_path  # noqa: PLC0415

    database = event_journal_sqlite_path(runtime_paths.storage_root)
    found = _journal_principal_ids(database, old)
    if not found:
        return
    principals = "; ".join(f"{old}@{user_id} ({counts})" for user_id, counts in found.items())
    state_path = matrix_state_file(runtime_paths)
    msg = (
        f"{state_path} holds no account agent_{old}, but the event journal {database} has rows for {principals}. "
        f"The rename keeps the agent's Matrix account; restore agent_{old} in {state_path} before renaming."
    )
    raise RenameRefusedError(msg)


def _newest_rename_manifest(storage_root: Path) -> Path | None:
    """Return the manifest of the newest `agents rename` backup; backup names start with their UTC time."""
    from mindroom.agent_rename.steps import BACKUPS_DIRECTORY_NAME  # noqa: PLC0415

    manifests = sorted((storage_root / BACKUPS_DIRECTORY_NAME).glob("*-rename/manifest.json"))
    return manifests[-1] if manifests else None


def _unfinished_rename_hint(rename: _AgentRename, manifest: Path) -> str:
    return (
        f"Agent {rename.old} is still configured, so if an earlier rename to {rename.new} stopped before it finished, "
        f"restore what it changed from {manifest} by hand before running again."
    )


def _refuse_unfinished_rename(runtime_paths: RuntimePaths, rename: _AgentRename, *, new_account: bool) -> None:
    """Refuse when a rename to ``rename.new`` looks killed before its config edit, naming what it left and its manifest.

    Such a run leaves the old name configured while the journal principal, state root or account of the new name
    exists already; its backup manifest lists what it changed.
    """
    from mindroom.agent_rename.steps import RenameRefusedError  # noqa: PLC0415
    from mindroom.event_journal.opening import event_journal_sqlite_path  # noqa: PLC0415
    from mindroom.matrix.identity import managed_account_key  # noqa: PLC0415
    from mindroom.tool_system.worker_routing import agent_state_root_path  # noqa: PLC0415

    storage = runtime_paths.storage_root
    manifest = _newest_rename_manifest(storage)
    if manifest is None:
        return
    found: list[str] = []
    database = event_journal_sqlite_path(storage)
    counts = _journal_principal_ids(database, rename.new).get(rename.user_id)
    if counts is not None:
        found.append(f"the event journal {database} has rows for {rename.new}@{rename.user_id} ({counts})")
    state_root = agent_state_root_path(storage, rename.new)
    if state_root.exists() or state_root.is_symlink():
        found.append(f"{state_root} exists")
    if new_account:
        found.append(f"{matrix_state_file(runtime_paths)} holds the account {managed_account_key(rename.new)}")
    if not found:
        return
    msg = (
        f"Agent {rename.old} is still configured, but {'; '.join(found)}, so an earlier rename to {rename.new} "
        f"stopped before it finished. Restore what it changed from {manifest} by hand before running again."
    )
    raise RenameRefusedError(msg)


def _rename_header(rename: _AgentRename) -> str:
    if rename.has_account:
        detail = f"it keeps its Matrix account {rename.user_id} and device:"
    else:
        detail = "it has no stored Matrix account yet:"
    return f"Plan to rename agent {rename.old} to {rename.new}; {detail}"


def _checked_rename_plan(runtime_paths: RuntimePaths, config: Config, rename: _AgentRename) -> RenamePlan:
    """Plan the rename; a destination that exists already points at the newest manifest, as a killed run leaves one."""
    from mindroom.agent_rename.steps import RenameRefusedError  # noqa: PLC0415
    from mindroom.agent_rename.workers import DestinationExistsError  # noqa: PLC0415

    try:
        return _rename_plan(runtime_paths, config, rename)
    except DestinationExistsError as exc:
        manifest = _newest_rename_manifest(runtime_paths.storage_root)
        if manifest is None:
            raise
        msg = f"{exc} {_unfinished_rename_hint(rename, manifest)}"
        raise RenameRefusedError(msg) from exc


def _rename_plan(runtime_paths: RuntimePaths, config: Config, rename: _AgentRename) -> RenamePlan:
    from mindroom.agent_rename.config_edit import config_entity_steps, env_value_steps  # noqa: PLC0415
    from mindroom.agent_rename.entity_files import (  # noqa: PLC0415
        entity_credential_steps,
        rewrite_worker_key_agent,
        unknown_worker_reports,
    )
    from mindroom.agent_rename.entity_state import entity_state_steps, matrix_state_key_step  # noqa: PLC0415
    from mindroom.agent_rename.indexes import index_steps  # noqa: PLC0415
    from mindroom.agent_rename.journal_entity import journal_entity_steps  # noqa: PLC0415
    from mindroom.agent_rename.memory import memory_steps  # noqa: PLC0415
    from mindroom.agent_rename.side_stores import entity_file_steps  # noqa: PLC0415
    from mindroom.agent_rename.steps import FinalAction, RenamePlan  # noqa: PLC0415
    from mindroom.agent_rename.teams import team_steps  # noqa: PLC0415
    from mindroom.agent_rename.workers import worker_move_steps  # noqa: PLC0415
    from mindroom.event_journal.opening import event_journal_sqlite_path  # noqa: PLC0415

    old, new = rename.old, rename.new
    storage = runtime_paths.storage_root
    # Without a stored account the journal holds no principal of the agent (_agent_rename refuses otherwise),
    # so only its name columns change.
    journal_steps = journal_entity_steps(
        event_journal_sqlite_path(storage),
        old_name=old,
        new_name=new,
        user_id=rename.user_id,
        configured_teams=frozenset(config.teams),
    )
    side_steps, side_reports = entity_file_steps(runtime_paths, config=config, old_name=old, new_name=new)
    worker_steps, worker_reports = worker_move_steps(
        runtime_paths,
        config=config,
        rewrite=partial(rewrite_worker_key_agent, old_name=old, new_name=new),
        discard_new_state=False,
    )
    index, index_reports = index_steps(runtime_paths, config=config, old_name=old, new_name=new)
    account_step = matrix_state_key_step(runtime_paths, old_name=old, new_name=new)
    # The journal goes first: its edit checks again in its transaction, and a failing first step leaves nothing to undo.
    # The state root moves before the stores inside it are edited, matrix_state.yaml follows every store, and the
    # config goes last, so a run killed before it still has the old name configured. The providers read the config
    # that still names the old agent.
    ordered = (
        *journal_steps,
        *entity_state_steps(runtime_paths, old_name=old, new_name=new),
        *memory_steps(runtime_paths, config=config, old_name=old, new_name=new),
        *team_steps(runtime_paths, config=config, old_name=old, new_name=new),
        *side_steps,
        *entity_credential_steps(runtime_paths, old_name=old, new_name=new),
        *worker_steps,
        *index,
        *(() if account_step is None else (account_step,)),
        *config_entity_steps(
            runtime_paths.config_path.expanduser(),
            old_name=old,
            new_name=new,
            storage_root=storage,
        ),
        *env_value_steps(runtime_paths.env_path, variable=_CONNECTIONS_AGENT_VARIABLE, old_value=old, new_value=new),
    )
    # Final actions always run after every reversible step; listing them last shows that.
    steps = (
        *(step for step in ordered if not isinstance(step, FinalAction)),
        *(step for step in ordered if isinstance(step, FinalAction)),
    )
    reports = (
        *side_reports,
        *worker_reports,
        *unknown_worker_reports(runtime_paths, old_name=old),
        *index_reports,
    )
    return RenamePlan(storage_root=storage, steps=steps, reports=reports)


def _print_plan(header: str, plan: RenamePlan) -> None:
    _say(header)
    for line in plan.describe():
        _say(f"  - {line}")


def _confirm(*, yes: bool) -> None:
    if not yes and not typer.confirm("Apply this plan?"):
        _say("Nothing was changed.")
        raise typer.Exit(1)


def _apply(plan: RenamePlan, *, kind: str) -> ApplyResult:
    from datetime import UTC, datetime  # noqa: PLC0415

    from mindroom.agent_rename.steps import apply_plan  # noqa: PLC0415
    from mindroom.workers.backend import WorkerBackendError  # noqa: PLC0415
    from mindroom.workers.runtime import shutdown_primary_worker_manager  # noqa: PLC0415

    try:
        return apply_plan(plan, kind=kind, now=datetime.now(UTC))
    finally:
        # Retiring an old Docker worker leases the process-wide worker manager; shut it down so the command exits.
        try:
            shutdown_primary_worker_manager()
        except WorkerBackendError as exc:
            _say(str(exc), label="[yellow]Warning:[/yellow] ")


def _print_applied(result: ApplyResult) -> None:
    _say(f"Backups: {result.backup_dir}. Its manifest.json lists every change, for restoring by hand.")
    for warning in result.warnings:
        _say(warning, label="[yellow]Warning:[/yellow] ")


def _reported_paths(reports: tuple[str, ...], storage_root: Path) -> frozenset[Path]:
    """Return the paths in the storage directory that the plan's reports name, which the leftover scan expects.

    Reports name paths in prose, followed by punctuation. A path with whitespace in it is cut short there,
    so it does not count and the scan lists it as not reported. The storage directory itself never counts,
    as everything in it would.
    """
    return frozenset(
        path
        for report in reports
        for word in report.split()
        if (path := Path(word.rstrip(".,;:)"))) != storage_root and path.is_relative_to(storage_root)
    )


def _print_leftovers(
    storage: Path,
    subject: str,
    *,
    needles: tuple[str, ...] = (),
    names: tuple[str, ...] = (),
    expected: frozenset[Path],
) -> None:
    """Print what in the storage directory still names ``subject``, split by whether the plan reported it."""
    from mindroom.agent_rename.scan import scan_leftovers  # noqa: PLC0415

    report = scan_leftovers(storage, needles=needles, expected=expected, names=names)
    if not report.expected and not report.unexpected:
        _say(f"Leftover scan: nothing in {storage} names {subject} any more.")
        return
    _say(f"Leftover scan: places in {storage} that still name {subject}")
    if report.expected:
        _say("  Reported above as staying:")
        for line in report.expected:
            _say(f"    - {line}")
    if report.unexpected:
        _say("  Not reported by the plan, or not scanned:")
        for line in report.unexpected:
            _say(f"    - {line}")
