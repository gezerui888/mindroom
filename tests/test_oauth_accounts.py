"""Labelled accounts of an OAuth provider: label validation and the labelled credential service."""

from __future__ import annotations

from dataclasses import replace

import pytest

from mindroom.credentials.manager import validate_service_name
from mindroom.credentials.policy import credential_service_policy
from mindroom.oauth.accounts import account_credential_service, provider_for_account, validate_account_label
from mindroom.oauth.providers import OAuthProvider

_DEMO = OAuthProvider(
    id="demo",
    display_name="Demo",
    authorization_url="https://auth.example.test/authorize",
    token_url="https://auth.example.test/token",  # noqa: S106 - an endpoint, not a secret
    scopes=("read",),
    credential_service="demo_oauth",
    client_config_services=("demo_oauth_client",),
)


@pytest.mark.parametrize("label", ["work", "a", "team-mail", "0", "x" * 31, "a1-b2"])
def test_valid_labels_are_accepted(label: str) -> None:
    """Lowercase letters, digits, and inner hyphens up to 31 characters are labels."""
    assert validate_account_label(label) == label


@pytest.mark.parametrize(
    "label",
    ["", "Work", "a_b", "a/b", "..", "a..b", "-work", "x" * 32, "a b", "a\nb", "work\n", "é"],
)
def test_invalid_labels_are_refused(label: str) -> None:
    """Path parts, underscores, uppercase, and over-long labels never reach a store path."""
    with pytest.raises(ValueError, match="label"):
        validate_account_label(label)


def test_own_connection_keeps_the_provider() -> None:
    """No label is the provider's own connection, unchanged."""
    assert provider_for_account(_DEMO, None) is _DEMO
    assert account_credential_service(_DEMO, None) == "demo_oauth"


def test_labelled_account_replaces_only_the_credential_service() -> None:
    """A label changes the stored service name only, which keeps the token placement of `*_oauth` services."""
    labelled = provider_for_account(_DEMO, "work")

    assert labelled.credential_service == "demo__work_oauth"
    assert account_credential_service(_DEMO, "work") == "demo__work_oauth"
    assert labelled.id == _DEMO.id
    assert labelled.client_config_services == _DEMO.client_config_services
    assert validate_service_name(labelled.credential_service) == labelled.credential_service
    for scope in ("shared", "user", "user_agent", None):
        own = credential_service_policy("demo_oauth", scope)
        assert credential_service_policy(labelled.credential_service, scope) == replace(
            own,
            service=labelled.credential_service,
        )


def test_labelled_provider_refuses_an_invalid_label() -> None:
    """An invalid label never builds a provider copy."""
    with pytest.raises(ValueError, match="label"):
        provider_for_account(_DEMO, "../other")
