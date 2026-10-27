"""Signed-assertion settings implied by one Cloudflare Access team, shared by browser and MCP external auth."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mindroom.constants import RuntimePaths

_TEAM_ENV = "MINDROOM_CLOUDFLARE_ACCESS_TEAM"
# The one DNS label before cloudflareaccess.com, lowercase because `iss` is compared exactly.
_TEAM_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")


@dataclass(frozen=True)
class _CloudflareAccessDefaults:
    """Header, issuer and signing keys of the team's application token; all None when no team is set."""

    jwt_header: str | None = None
    issuer: str | None = None
    jwks_url: str | None = None


def cloudflare_access_defaults(runtime_paths: RuntimePaths) -> _CloudflareAccessDefaults:
    """Return the defaults `MINDROOM_CLOUDFLARE_ACCESS_TEAM` implies, rejecting anything but a bare team name."""
    team = runtime_paths.env_value(_TEAM_ENV)
    if not team:
        return _CloudflareAccessDefaults()
    if _TEAM_NAME.fullmatch(team) is None:
        msg = f"{_TEAM_ENV} must be a team name, such as example for example.cloudflareaccess.com"
        raise ValueError(msg)
    issuer = f"https://{team}.cloudflareaccess.com"
    return _CloudflareAccessDefaults(
        jwt_header="Cf-Access-Jwt-Assertion",
        issuer=issuer,
        jwks_url=f"{issuer}/cdn-cgi/access/certs",
    )
