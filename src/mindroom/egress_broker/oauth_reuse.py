"""Which OAuth connections Agent Keys may use: an explicit, revocable opt-in per service.

A connection serves the tools of its provider, whose writes ask for approval. Agent Keys hands the same token to the
agent's shell and python, which run without approval prompts, so the broker uses a connection for a service only once
someone allowed it. The opt-in lives in the store of the connection it unlocks (`oauth_source.broker_credential_context`):
the requester's own store for a requester-scoped provider such as GitHub, else the store of the service's keys. One
reserved document there, `egress__oauth_reuse`, holds `{"services": {key: {"provider", "enabled_by", "enabled_at"}}}`,
keyed by the service name for the provider's own connection and `service@label` for a labelled account
(`mindroom.oauth.accounts`). An entry counts as off while it names a provider other than the service's
`oauth_provider`. Workers never read it: egress documents are primary-only.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from mindroom.egress_broker.secrets import delete_egress_document, load_egress_document, save_egress_document
from mindroom.oauth.accounts import validate_account_label

if TYPE_CHECKING:
    from collections.abc import Callable

    from mindroom.oauth.credential_lifecycle import OAuthCredentialContext
    from mindroom.tool_system.worker_routing import ResolvedWorkerTarget

__all__ = [
    "OAUTH_REUSE_CREDENTIAL_SERVICE",
    "oauth_reuse_enabled",
    "set_oauth_reuse",
    "stop_account_reuse",
    "stop_service_reuse",
]

OAUTH_REUSE_CREDENTIAL_SERVICE = "egress__oauth_reuse"
# Serializes read-modify-write of the opt-in documents in this process, so concurrent changes lose nothing.
_write_lock = threading.Lock()


def oauth_reuse_enabled(
    context: OAuthCredentialContext,
    service: str,
    provider_id: str,
    account: str | None = None,
) -> bool:
    """Return whether Agent Keys may use the context's connection, or its labelled `account`, for `service`.

    The opt-in must be given for `provider_id`. A requester-scoped provider without a requester has no store to allow
    it in. This reads the credential store, so async callers run it in a thread.
    """
    if context.provider.requester_scoped_credentials and context.worker_target is None:
        return False
    match _stored(context).get(_entry_key(service, account)):
        case {"provider": provider} if provider == provider_id:
            return True
        case _:
            return False


def set_oauth_reuse(
    context: OAuthCredentialContext,
    service: str,
    provider_id: str,
    *,
    enabled: bool,
    by: str | None,
    account: str | None = None,
) -> None:
    """Allow Agent Keys to use the context's connection, or its labelled `account`, for `service`, or stop it.

    `by` names who did, if known. The last entry takes the document with it. Blocks on the store, so async callers
    run it in a thread.
    """
    key = _entry_key(service, account)
    with _write_lock:
        stored = _stored(context)
        if enabled:
            stored[key] = {
                "provider": provider_id,
                "enabled_by": by,
                "enabled_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            }
        else:
            stored.pop(key, None)
        _save(context, stored)


def stop_service_reuse(context: OAuthCredentialContext, service: str) -> None:
    """Stop Agent Keys using any account for `service`, the provider's own connection and every labelled one."""
    _keep_only(context, lambda key, _entry: key.partition("@")[0] != service)


def stop_account_reuse(context: OAuthCredentialContext, provider_id: str, account: str) -> None:
    """Stop Agent Keys using the labelled `account` of `provider_id` for every service, as disconnecting it does."""
    validate_account_label(account)

    def keep(key: str, entry: object) -> bool:
        match entry:
            case {"provider": provider} if provider == provider_id:
                return key.partition("@")[2] != account
            case _:
                return True

    _keep_only(context, keep)


def _entry_key(service: str, account: str | None) -> str:
    return service if account is None else f"{service}@{validate_account_label(account)}"


def _keep_only(context: OAuthCredentialContext, keep: Callable[[str, object], bool]) -> None:
    with _write_lock:
        stored = _stored(context)
        _save(context, {key: entry for key, entry in stored.items() if keep(key, entry)})


def _save(context: OAuthCredentialContext, stored: dict[str, object]) -> None:
    manager, target = context.credentials_manager, _store_target(context)
    if stored:
        save_egress_document(manager, target, OAUTH_REUSE_CREDENTIAL_SERVICE, {"services": stored})
    else:
        delete_egress_document(manager, target, OAUTH_REUSE_CREDENTIAL_SERVICE)


def _store_target(context: OAuthCredentialContext) -> ResolvedWorkerTarget | None:
    """Return the egress store of the context's connection; an unscoped agent's is the global store, like its keys."""
    target = context.worker_target
    return None if target is None or target.worker_scope is None else target


def _stored(context: OAuthCredentialContext) -> dict[str, object]:
    document = load_egress_document(context.credentials_manager, _store_target(context), OAUTH_REUSE_CREDENTIAL_SERVICE)
    services = None if document is None else document.get("services")
    return dict(services) if isinstance(services, dict) else {}
