"""One Cloudflare Access team supplies the signed-assertion settings of browser and MCP external auth."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mindroom.api import auth
from mindroom.constants import resolve_primary_runtime_paths
from mindroom.mcp_gateway.external_auth import ExternalAuthSettings

if TYPE_CHECKING:
    from pathlib import Path

    from mindroom.constants import RuntimePaths

_ISSUER = "https://example.cloudflareaccess.com"
_JWKS_URL = "https://example.cloudflareaccess.com/cdn-cgi/access/certs"
_MCP_ENV = {
    "MINDROOM_MCP_AUTH_MODE": "external",
    "MINDROOM_MCP_EXTERNAL_AUDIENCE": "mcp-application-audience",
    "MINDROOM_MCP_EXTERNAL_MATRIX_USER_ID_CLAIM": "matrix_id",
}


def _paths(tmp_path: Path, **env: str) -> RuntimePaths:
    return resolve_primary_runtime_paths(
        config_path=tmp_path / "config.yaml",
        storage_path=tmp_path / "mindroom_data",
        process_env=env,
    )


def _trusted_upstream(tmp_path: Path, **env: str) -> auth._TrustedUpstreamAuthSettings:
    return auth._build_auth_settings(_paths(tmp_path, **env)).trusted_upstream


def _mcp(tmp_path: Path, **env: str) -> ExternalAuthSettings:
    settings = ExternalAuthSettings.from_paths(_paths(tmp_path, **_MCP_ENV, **env))
    assert settings is not None
    return settings


def test_team_supplies_trusted_upstream_header_keys_and_issuer(tmp_path: Path) -> None:
    """Only the browser application's audience remains to configure."""
    settings = _trusted_upstream(
        tmp_path,
        MINDROOM_CLOUDFLARE_ACCESS_TEAM="example",
        MINDROOM_TRUSTED_UPSTREAM_AUTH_ENABLED="true",
        MINDROOM_TRUSTED_UPSTREAM_JWT_AUDIENCE="browser-application-audience",
    )
    assert settings.jwt_header == "Cf-Access-Jwt-Assertion"
    assert settings.jwks_url == _JWKS_URL
    assert settings.issuer == _ISSUER
    assert settings.audience == "browser-application-audience"


def test_team_supplies_mcp_authority_keys_issuer_and_header(tmp_path: Path) -> None:
    """The team is the advertised authorization server and the issuer of the forwarded assertion."""
    settings = _mcp(tmp_path, MINDROOM_CLOUDFLARE_ACCESS_TEAM="example")
    assert settings.authorization_server == _ISSUER
    assert settings.issuer == _ISSUER
    assert settings.jwks_url == _JWKS_URL
    assert settings.token_header == "cf-access-jwt-assertion"  # noqa: S105 -- header name
    assert settings.audience == "mcp-application-audience"


def test_explicit_settings_override_the_team(tmp_path: Path) -> None:
    """Each feature's own variable wins over the value the team implies."""
    trusted_upstream = _trusted_upstream(
        tmp_path,
        MINDROOM_CLOUDFLARE_ACCESS_TEAM="example",
        MINDROOM_TRUSTED_UPSTREAM_JWT_HEADER="X-Trusted-Jwt",
        MINDROOM_TRUSTED_UPSTREAM_JWKS_URL="https://gateway.example.com/keys",
        MINDROOM_TRUSTED_UPSTREAM_JWT_ISSUER="https://gateway.example.com",
    )
    assert (trusted_upstream.jwt_header, trusted_upstream.jwks_url, trusted_upstream.issuer) == (
        "X-Trusted-Jwt",
        "https://gateway.example.com/keys",
        "https://gateway.example.com",
    )
    mcp = _mcp(
        tmp_path,
        MINDROOM_CLOUDFLARE_ACCESS_TEAM="example",
        MINDROOM_MCP_EXTERNAL_AUTHORIZATION_SERVER="https://login.example.com",
        MINDROOM_MCP_EXTERNAL_ISSUER="https://issuer.example.com",
        MINDROOM_MCP_EXTERNAL_JWKS_URL="https://issuer.example.com/keys",
        MINDROOM_MCP_EXTERNAL_TOKEN_HEADER="Authorization",  # noqa: S106 -- header name
    )
    assert (mcp.authorization_server, mcp.issuer, mcp.jwks_url, mcp.token_header) == (
        "https://login.example.com",
        "https://issuer.example.com",
        "https://issuer.example.com/keys",
        "authorization",
    )


@pytest.mark.parametrize(
    "team",
    [
        "example.cloudflareaccess.com",
        "https://example.cloudflareaccess.com",
        "Example",
        " example",
        "-example",
        "example/",
        "x" * 64,
    ],
)
def test_team_other_than_a_bare_name_is_rejected(tmp_path: Path, team: str) -> None:
    """A domain, URL, or invalid label fails startup for both features instead of deriving a wrong issuer."""
    message = "MINDROOM_CLOUDFLARE_ACCESS_TEAM must be a team name, such as example for example.cloudflareaccess.com"
    with pytest.raises(ValueError, match=message):
        _trusted_upstream(tmp_path, MINDROOM_CLOUDFLARE_ACCESS_TEAM=team)
    with pytest.raises(ValueError, match=message):
        _mcp(tmp_path, MINDROOM_CLOUDFLARE_ACCESS_TEAM=team)


@pytest.mark.parametrize("team", [None, ""])
def test_without_team_nothing_is_derived(tmp_path: Path, team: str | None) -> None:
    """Without a team, browser auth keeps its unset settings and MCP external mode still requires its own."""
    env = {} if team is None else {"MINDROOM_CLOUDFLARE_ACCESS_TEAM": team}
    trusted_upstream = _trusted_upstream(tmp_path, **env)
    assert (trusted_upstream.jwt_header, trusted_upstream.jwks_url, trusted_upstream.issuer) == (None, None, None)
    with pytest.raises(ValueError, match="Invalid external authentication text"):
        _mcp(tmp_path, **env)
    mcp = _mcp(
        tmp_path,
        **env,
        MINDROOM_MCP_EXTERNAL_AUTHORIZATION_SERVER=_ISSUER,
        MINDROOM_MCP_EXTERNAL_ISSUER=_ISSUER,
        MINDROOM_MCP_EXTERNAL_JWKS_URL=_JWKS_URL,
    )
    assert mcp.token_header == "authorization"  # noqa: S105 -- header name
