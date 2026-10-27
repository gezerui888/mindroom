"""Labelled accounts of an OAuth provider: further connections that only Agent Keys uses.

A labelled account is the provider with its `credential_service` replaced, so every connect, refresh, and storage
path serves it unchanged in the same scope store as the provider's own connection. The label goes before the `_oauth`
suffix, which keeps the name under the placement and validation rules of OAuth token services.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mindroom.oauth.providers import OAuthProvider

__all__ = ["account_credential_service", "account_labels", "provider_for_account", "validate_account_label"]

_ACCOUNT_LABEL = re.compile(r"[a-z0-9][a-z0-9-]{0,30}")
_SERVICE_SUFFIX = "_oauth"


def validate_account_label(label: str) -> str:
    """Return `label` if it is a valid account label, else raise ValueError."""
    if not _ACCOUNT_LABEL.fullmatch(label):
        msg = "An account label has 1 to 31 lowercase letters, digits, or hyphens and starts with a letter or digit"
        raise ValueError(msg)
    return label


def account_credential_service(provider: OAuthProvider, account: str | None) -> str:
    """Return the credential service of the provider's own connection (None) or of its labelled `account`."""
    if account is None:
        return provider.credential_service
    base = provider.credential_service.removesuffix(_SERVICE_SUFFIX)
    return f"{base}__{validate_account_label(account)}{_SERVICE_SUFFIX}"


def provider_for_account(provider: OAuthProvider, account: str | None) -> OAuthProvider:
    """Return the provider whose connection is `account`: itself for None, else a copy storing the labelled account."""
    if account is None:
        return provider
    return replace(provider, credential_service=account_credential_service(provider, account))


def account_labels(provider: OAuthProvider, credential_services: Iterable[str]) -> list[str]:
    """Return the labels of the provider's labelled accounts among `credential_services`, in their order."""
    prefix = f"{provider.credential_service.removesuffix(_SERVICE_SUFFIX)}__"
    labels = (
        service.removeprefix(prefix).removesuffix(_SERVICE_SUFFIX)
        for service in credential_services
        if service.startswith(prefix) and service.endswith(_SERVICE_SUFFIX)
    )
    return [label for label in labels if _ACCOUNT_LABEL.fullmatch(label)]
