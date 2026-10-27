"""Rendered Helm manifest checks for the web client chart."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
from starlette.routing import Route, WebSocketRoute

from mindroom.api.main import app as runtime_app
from tests.test_helm_instance_worker_isolation import _render_chart, _resource, _run_helm_template, _values_files

_NGINX_IMAGE = "mirror.gcr.io/library/nginx:alpine"
_IMMUTABLE = "public, max-age=31536000, immutable"


def _client_nginx_conf(*set_args: str) -> str:
    docs = _render_chart(Path("cluster/k8s/client"), *set_args, release_name="mindroom-client")
    return next(doc["data"]["default.conf"] for doc in docs if "default.conf" in doc.get("data", {}))


def _run_nginx(nginx_conf: str, files: dict[str, str], commands: list[str]) -> list[str]:
    """Run shell commands beside nginx serving nginx_conf, with files planted at their absolute paths.

    Returns the commands' output lines.
    """
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker is required to serve the rendered nginx config")
    if subprocess.run([docker, "info"], check=False, capture_output=True).returncode != 0:
        pytest.skip("Docker daemon is unavailable to serve the rendered nginx config")
    script = "\n".join(
        [
            "set -eu",
            "cat > /etc/nginx/conf.d/default.conf",
            *(
                f"mkdir -p {shlex.quote(str(PurePosixPath(path).parent))} && printf %s {shlex.quote(body)} > {path}"
                for path, body in files.items()
            ),
            # The image logs requests to stdout, which carries the responses.
            "nginx >&2",
            *commands,
        ],
    )

    completed = subprocess.run(
        [docker, "run", "--rm", "-i", _NGINX_IMAGE, "sh", "-c", script],
        input=nginx_conf,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    return completed.stdout.splitlines()


def _curl(path: str, *args: str, write_out: str = "%{http_code}\\t") -> str:
    """Return a shell command that prints write_out and the one-line body of a request to the served nginx."""
    url = f"http://127.0.0.1:8080{path}"
    return f"curl -s -o /tmp/body -w '{write_out}' {shlex.join([*args, url])} && tr -d '\\r\\n' < /tmp/body && echo"


def _serve(
    nginx_conf: str,
    files: dict[str, str],
    paths: list[str],
    headers: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    """Serve nginx_conf in nginx with files planted at their absolute paths.

    Returns each path's status code, the requested response headers, and the body.
    """
    write_out = "\\t".join(["%{http_code}", *(f"%header{{{header}}}" for header in headers)])
    lines = _run_nginx(nginx_conf, files, [_curl(path, write_out=f"{write_out}\\t") for path in paths])
    return {path: tuple(line.split("\t", len(headers) + 1)) for path, line in zip(paths, lines, strict=True)}


@pytest.mark.parametrize(("base_path", "app_shell_location"), [("/", "location / {"), ("/chat", "location /chat/ {")])
def test_client_chart_lets_only_its_own_origin_frame_the_app_shell(base_path: str, app_shell_location: str) -> None:
    """A page framing the signed-in client could trick its user into clicking the client's controls."""
    nginx_conf = _client_nginx_conf(f"basePath={base_path}")
    app_shell = nginx_conf.split(app_shell_location, 1)[1].split("}", 1)[0]

    assert "add_header Content-Security-Policy \"frame-ancestors 'self'\" always;" in app_shell
    assert 'add_header X-Frame-Options "SAMEORIGIN" always;' in app_shell


@pytest.mark.parametrize("base_path", ["/", "/chat", "/assets", "/public"])
def test_client_chart_serves_nested_build_files_from_their_own_path(base_path: str) -> None:
    """Element Call resolves to its own files, other build files resolve as before, and 404s are not immutable.

    The bundled Element Call ships its own assets/ directory under public/element-call/.
    Resolving its hashed files against the app's top-level assets/ returned 404, and the immutable header on that 404
    pinned the broken call in the browser.
    Base paths and route segments named assets or public must still resolve to the top-level build files.
    """
    files = {
        "/usr/share/nginx/html/assets/app-1a2b.js": "app bundle",
        "/usr/share/nginx/html/public/locales/en.json": "translations",
        "/usr/share/nginx/html/public/element-call/index.html": "call page",
        "/usr/share/nginx/html/public/element-call/assets/call-3c4d.js": "call bundle",
    }
    prefix = base_path.rstrip("/")
    expected = {
        f"{prefix}{path}": response
        for path, response in {
            "/assets/app-1a2b.js": ("200", _IMMUTABLE, "app bundle"),
            "/rooms/abc/assets/app-1a2b.js": ("200", _IMMUTABLE, "app bundle"),
            "/rooms/public/assets/app-1a2b.js": ("200", _IMMUTABLE, "app bundle"),
            "/rooms/assets/public/locales/en.json": ("200", _IMMUTABLE, "translations"),
            "/public/locales/en.json": ("200", _IMMUTABLE, "translations"),
            "/public/element-call/index.html": ("200", "no-cache", "call page"),
            "/public/element-call/assets/call-3c4d.js": ("200", _IMMUTABLE, "call bundle"),
            "/rooms/abc/public/element-call/assets/call-3c4d.js": ("200", _IMMUTABLE, "call bundle"),
            "/public/element-call/assets/missing-5e6f.js": ("404", "", None),
            "/assets/missing-5e6f.js": ("404", "", None),
        }.items()
    }

    responses = _serve(
        _client_nginx_conf(f"basePath={base_path}", "nginx.ipv6=false"),
        files,
        list(expected),
        ("cache-control",),
    )

    actual = {
        path: (status, cache_control, body if status == "200" else None)
        for path, (status, cache_control, body) in responses.items()
    }
    assert actual == expected


@pytest.mark.parametrize("base_path", ["/", "/chat"])
def test_client_chart_serves_pairing_notices_and_app_links_like_the_image(base_path: str) -> None:
    """The chart replaces the image's nginx config, so it must keep the image's routes for these client features.

    The pairing page approves a sign-in, so no origin may frame it, unlike the rest of the app.
    Notices come from the image's notices directory, which operators replace, not from the build output.
    iOS reads app links from the origin root whatever the base path.
    """
    prefix = base_path.rstrip("/")
    app_links = '{"applinks":{}}'
    files = {
        "/usr/share/nginx/html/index.html": "app shell",
        "/usr/share/nginx/html/notices.json": "[]",
        "/usr/share/nginx/html/.well-known/apple-app-site-association": app_links,
        "/opt/mindroom/notices/notices.json": '[{"id":"maintenance"}]',
    }
    no_store = "no-store, no-cache, must-revalidate, max-age=0"
    pairing = ("200", "text/html", no_store, "frame-ancestors 'none'", "DENY", "app shell")
    expected = {
        f"{prefix}/connect": pairing,
        f"{prefix}/connect/": pairing,
        f"{prefix}/connections": ("200", "text/html", no_store, "frame-ancestors 'self'", "SAMEORIGIN", "app shell"),
        f"{prefix}/notices.json": ("200", "application/json", "no-store, max-age=0", "", "", '[{"id":"maintenance"}]'),
        "/.well-known/apple-app-site-association": ("200", "application/json", "", "", "", app_links),
    }

    responses = _serve(
        _client_nginx_conf(f"basePath={base_path}", "nginx.ipv6=false"),
        files,
        list(expected),
        ("content-type", "cache-control", "content-security-policy", "x-frame-options"),
    )

    assert responses == expected


def _client_pod(tmp_path: Path, values: str) -> dict[str, Any]:
    docs = _render_chart(
        Path("cluster/k8s/client"),
        release_name="mindroom-client",
        values_files=_values_files(tmp_path, values),
    )
    return _resource(docs, "Deployment", "mindroom-client")["spec"]["template"]["spec"]


def test_client_chart_mounts_operator_notices_so_edits_reach_running_pods(tmp_path: Path) -> None:
    """A subPath mount would pin the file the pod started with, so notice edits would wait for a restart.

    The ConfigMap is optional so the client still starts, without notices, before an operator creates it.
    """
    pod = _client_pod(tmp_path, "notices:\n  existingConfigMap: client-notices\n")

    assert {"name": "notices", "configMap": {"name": "client-notices", "optional": True}} in pod["volumes"]
    assert {"name": "notices", "mountPath": "/opt/mindroom/notices", "readOnly": True} in pod["containers"][0][
        "volumeMounts"
    ]

    default_pod = _client_pod(tmp_path, "{}\n")
    assert "notices" not in {volume["name"] for volume in default_pod["volumes"]}


# Historical uppercase localparts, ports, and IPv6 server names are valid Matrix user IDs the client accepts.
_BUG_REPORT_ADMINS = ["@admin:chat.example.com", "@Ops=Team/1+x:chat.example.com:8448", "@admin:[2001:db8::1]"]
_MATRIX_RTC = """
matrixRTC:
  enabled: true
  livekitServiceUrl: https://chat.example.com/livekit/jwt
"""


def _matrix_client_well_known(tmp_path: Path, values: str) -> dict[str, Any] | None:
    docs = _render_chart(
        Path("cluster/k8s/client"),
        release_name="mindroom-client",
        values_files=_values_files(tmp_path, values),
    )
    nginx_conf = next(doc["data"]["default.conf"] for doc in docs if "default.conf" in doc.get("data", {}))
    _, found, location = nginx_conf.partition("location = /.well-known/matrix/client {")
    if not found:
        return None
    body = re.search(r"return 200 '([^']*)';", location)
    assert body is not None
    return json.loads(body[1])


@pytest.mark.parametrize("rtc", [False, True], ids=["no-rtc", "rtc"])
@pytest.mark.parametrize("admins", [False, True], ids=["no-admins", "admins"])
def test_client_chart_publishes_bug_report_admins_in_matrix_client_discovery(
    tmp_path: Path,
    *,
    rtc: bool,
    admins: bool,
) -> None:
    """MindRoom Chat sends one-click bug reports to the administrators its homeserver's well-known lists."""
    values = "matrix:\n  homeserverUrl: https://chat.example.com\n"
    if rtc:
        values += _MATRIX_RTC
    if admins:
        values += f"bugReports:\n  admins: {json.dumps(_BUG_REPORT_ADMINS)}\n"

    well_known = _matrix_client_well_known(tmp_path, values)

    if not (rtc or admins):
        assert well_known is None
        return
    assert well_known is not None
    assert well_known["m.homeserver"] == {"base_url": "https://chat.example.com"}
    assert ("org.matrix.msc4143.rtc_foci" in well_known) is rtc
    assert well_known.get("io.mindroom.bug_reports") == ({"admins": _BUG_REPORT_ADMINS} if admins else None)


def test_client_chart_keeps_matrix_rtc_discovery_unchanged_without_bug_report_admins(tmp_path: Path) -> None:
    """Deployments that publish only MatrixRTC keep the same nginx config, so its checksum triggers no rollout."""
    docs = _render_chart(
        Path("cluster/k8s/client"),
        release_name="mindroom-client",
        values_files=_values_files(tmp_path, "matrix:\n  homeserverUrl: https://chat.example.com\n" + _MATRIX_RTC),
    )
    nginx_conf = next(doc["data"]["default.conf"] for doc in docs if "default.conf" in doc.get("data", {}))

    assert (
        "\n\n  # MatrixRTC backend discovery for Matrix voice and video calls.\n"
        "  location = /.well-known/matrix/client {\n"
    ) in nginx_conf
    assert "Bug-report" not in nginx_conf


@pytest.mark.parametrize(
    ("values", "error"),
    [
        ('bugReports:\n  admins: "@admin:chat.example.com"\n', "bugReports.admins must be a list"),
        ('bugReports:\n  admins: ["admin"]\n', "bugReports.admins entries must be Matrix user IDs"),
        ('bugReports:\n  admins: ["@admin chat:example.com"]\n', "bugReports.admins entries must be Matrix user IDs"),
        # A C1 control character would otherwise break the rendered ConfigMap YAML.
        ('bugReports:\n  admins: ["@a\\x85b:chat.example.com"]\n', "bugReports.admins entries must be Matrix user IDs"),
        # A quote would end the nginx return body, nginx would unescape the JSON's backslash escapes,
        # and a $ would expand as an nginx variable.
        ('bugReports:\n  admins: ["@a\'b:chat.example.com"]\n', "bugReports.admins entries must be Matrix user IDs"),
        ("bugReports:\n  admins: ['@a\"b:chat.example.com']\n", "bugReports.admins entries must be Matrix user IDs"),
        ("bugReports:\n  admins: ['@a\\b:chat.example.com']\n", "bugReports.admins entries must be Matrix user IDs"),
        ('bugReports:\n  admins: ["@admin:chat$host"]\n', "bugReports.admins entries must be Matrix user IDs"),
        (
            'matrix:\n  homeserverUrl: ""\n  defaultServerName: chat.example.com\nbugReports:\n  admins: ["@admin:chat.example.com"]\n',
            "matrix.homeserverUrl is required when bugReports.admins is set",
        ),
        (
            'nginx:\n  existingConfigMap: client-nginx\nbugReports:\n  admins: ["@admin:chat.example.com"]\n',
            "bugReports.admins requires the chart-managed nginx config",
        ),
    ],
    ids=[
        "not-a-list",
        "no-sigil",
        "whitespace",
        "control",
        "quote",
        "double-quote",
        "backslash",
        "dollar",
        "no-homeserver-url",
        "existing-nginx",
    ],
)
def test_client_chart_rejects_bug_report_admins_it_cannot_publish(tmp_path: Path, values: str, error: str) -> None:
    """Administrators that cannot be published intact fail the render instead of silently disabling reports."""
    completed = _run_helm_template(
        Path("cluster/k8s/client"),
        release_name="mindroom-client",
        values_files=_values_files(tmp_path, values),
    )

    assert completed.returncode != 0
    assert error in completed.stderr


# One path for each runtime route the chart serves on the public host; the chart and the runtime must agree on all of them.
_PUBLIC_RUNTIME_PATHS = (
    "/connections",
    "/connections/mcp/authorize",
    "/api/connections",
    "/api/connections/session",
    "/api/oauth/google/callback",
    "/mcp",
    "/mcp/oauth/authorize",
    "/mcp/oauth/register",
    "/mcp/oauth/token",
    "/mcp/oauth/revoke",
    "/mcp/oauth/.well-known/oauth-authorization-server",
    "/mcp/scim/v2/Users",
    "/.well-known/oauth-authorization-server/mcp/oauth",
    "/.well-known/oauth-protected-resource/mcp",
    "/api/computers/sessions/s1/stream",
    "/api/triggers/t1",
    "/api/usage/export",
    "/api/budgets/service",
    "/api/budgets/service/users/u1",
    "/reports/public/r1/",
    "/reports/public/r1/assets/site.css",
    "/reports/private/r1",
    "/reports/private/agent/o1/w1/r1",
)
_HOMESERVER = "http://127.0.0.1:8008"
_RUNTIME = "http://127.0.0.1:8765"
# Stub upstreams that echo what reached them, served by the same nginx as the client.
_UPSTREAM_STUBS = "\n".join(
    f"""server {{
  listen {upstream.removeprefix("http://")};
  client_max_body_size 0;
  location / {{
    return 200 "{name} $request_method $request_uri host=$http_host xff=$http_x_forwarded_for proto=$http_x_forwarded_proto jwt=$http_x_trusted_jwt length=$http_content_length upgrade=$http_upgrade connection=$http_connection";
  }}
}}"""
    for name, upstream in (("homeserver", _HOMESERVER), ("runtime", _RUNTIME))
)


def _origin_proxy_conf(*set_args: str) -> str:
    return _client_nginx_conf(
        f"proxy.homeserverUpstream={_HOMESERVER}",
        f"proxy.runtimeUpstream={_RUNTIME}",
        "nginx.ipv6=false",
        *set_args,
    )


def _serve_origin(*commands: str) -> list[str]:
    return _run_nginx(
        _origin_proxy_conf(),
        {"/etc/nginx/conf.d/upstreams.conf": _UPSTREAM_STUBS, "/usr/share/nginx/html/index.html": "app shell"},
        list(commands),
    )


def test_client_chart_proxied_runtime_paths_match_runtime_routes() -> None:
    """A renamed or removed runtime route must fail here instead of silently sending its users to the client.

    Each runtime location must also serve at least one listed path, so the routing test below exercises it.
    """
    runtime_locations = [
        location
        for location, body in re.findall(r"\n  location ~ (\S+) \{\n(.*?)\n  \}", _origin_proxy_conf(), re.DOTALL)
        if f"proxy_pass {_RUNTIME};" in body
    ]
    routes = [
        route
        for route in runtime_app.routes
        if isinstance(route, Route | WebSocketRoute) and route.path not in {"/", "/{path:path}"}
    ]

    assert runtime_locations
    for location in runtime_locations:
        assert any(re.search(location, path) for path in _PUBLIC_RUNTIME_PATHS), location
    # The dashboard frontend's catch-all route serves the Connections bundle at /connections.
    for path in set(_PUBLIC_RUNTIME_PATHS) - {"/connections"}:
        assert any(route.path_regex.match(path) for route in routes), path


def test_client_chart_routes_the_whole_host_between_homeserver_runtime_and_client() -> None:
    """Only the public runtime routes reach the runtime; the dashboard and the rest of its API stay with the client.

    nginx matches locations on the normalized path but forwards the raw one, so a raw runtime path that the runtime
    would route to another endpoint, such as a dashboard route under dot segments, must not be forwarded.
    """
    homeserver_paths = [
        "/_matrix/client/v3/sync?timeout=30000",
        # The raw path reaches the homeserver, so an encoded room alias keeps its # sigil.
        "/_matrix/client/v3/directory/room/%23lobby:example.com",
        # Matrix paths with a public/ or assets/ segment are not client build files.
        "/_matrix/media/v3/download/example.com/public/m1",
        "/_synapse/admin/v1/whois/@u:example.com",
        "/_tuwunel/server_version",
        "/.well-known/matrix/server",
        "/.well-known/matrix/client",
    ]
    client_paths = [
        "/",
        "/api/config/raw",
        "/api/usage",
        "/api/usage/me/private-agents",
        "/api/health",
        "/v1/models",
        "/login",
        "/connectionsx",
        "/mcp/other",
        "/mcp/scim/v2",
        "/api/usage/export/",
        "/reports/public/",
        "/reports/private/a/b",
        "/.well-known/oauth-protected-resource/mcp/other",
        # Normalizes to the client route /api/config/raw.
        "/api/connections/%2e%2e/config/raw",
    ]
    refused_paths = [
        # Normalizes to /api/connections, but the runtime would route the raw path to a knowledge endpoint.
        "/api/knowledge/bases/b/files/../../../../connections",
        "/api/config/%2E%2E/connections",
        # Normalizes to /mcp, but the runtime would serve the dashboard for the raw path.
        "//mcp",
        "/api/oauth/google%2Fcallback",
    ]
    paths = [*homeserver_paths, *_PUBLIC_RUNTIME_PATHS, *client_paths, *refused_paths]

    lines = _serve_origin(*(_curl(path, "--path-as-is") for path in paths))

    responses = dict(zip(paths, lines, strict=True))
    for path in homeserver_paths:
        assert responses[path].startswith(f"200\thomeserver GET {path} "), path
    for path in _PUBLIC_RUNTIME_PATHS:
        assert responses[path].startswith(f"200\truntime GET {path} "), path
    for path in client_paths:
        assert responses[path] == "200\tapp shell", path
    for path in refused_paths:
        assert responses[path].startswith("404\t<html>"), path


def test_client_chart_forwards_identity_uploads_and_websockets_to_the_upstreams() -> None:
    """The upstreams see the browser's host and the gateway's headers, and only the homeserver limits uploads."""
    gateway = (
        "-H",
        "Host: chat.example.com",
        "-H",
        "X-Forwarded-For: 203.0.113.7",
        "-H",
        "X-Forwarded-Proto: https",
        "-H",
        "X-Trusted-Jwt: signed",
    )
    upload = ("-X", "POST", "-H", "Content-Type: application/octet-stream")

    lines = _serve_origin(
        _curl("/_matrix/client/versions", *gateway),
        _curl("/api/connections/me", *gateway),
        "head -c 2097152 /dev/zero > /tmp/upload",
        _curl("/_matrix/media/v3/upload", *upload, "--data-binary", "@/tmp/upload"),
        _curl("/_matrix/media/v3/upload", *upload, "-H", "Transfer-Encoding: chunked", "--data-binary", "@/tmp/upload"),
        _curl("/api/computers/sessions/s1/stream", "-H", "Connection: Upgrade", "-H", "Upgrade: websocket"),
    )

    identity = "host=chat.example.com xff=203.0.113.7, 127.0.0.1 proto=https jwt=signed"
    assert lines[0] == f"200\thomeserver GET /_matrix/client/versions {identity} length= upgrade= connection="
    assert lines[1] == f"200\truntime GET /api/connections/me {identity} length= upgrade= connection="
    assert lines[2].startswith("200\thomeserver POST /_matrix/media/v3/upload ")
    assert "length=2097152 " in lines[2]
    assert lines[3].startswith("200\thomeserver POST /_matrix/media/v3/upload ")
    assert lines[4].endswith("upgrade=websocket connection=upgrade")
    assert lines[4].startswith("200\truntime GET /api/computers/sessions/s1/stream ")


def test_client_chart_keeps_serving_its_own_matrix_client_discovery_with_a_homeserver_upstream() -> None:
    """The chart's discovery carries MatrixRTC and bug-report settings the homeserver does not publish."""
    nginx_conf = _origin_proxy_conf("bugReports.admins[0]=@admin:chat.example.com")

    discovery = nginx_conf.split("location = /.well-known/matrix/client {", 1)[1].split("\n  }", 1)[0]
    assert nginx_conf.count("location = /.well-known/matrix/client {") == 1
    assert "io.mindroom.bug_reports" in discovery
    assert "proxy_pass" not in discovery


def test_client_chart_keeps_proxied_browser_pages_out_of_the_cached_app_shell() -> None:
    """The client's service worker answers navigations with its cached shell unless a path is excluded."""
    nginx_conf = _origin_proxy_conf("serviceWorker.navigationFallbackExcludePaths[0]=/other-app")

    assert (
        'window.__SERVICE_WORKER_NAVIGATION_FALLBACK_EXCLUDE_PATHS__ = [\\"/connections\\",\\"/mcp\\",\\"/reports\\",'
        '\\"/_tuwunel\\",\\"/other-app\\"];'
    ) in nginx_conf


@pytest.mark.parametrize(
    ("values", "error"),
    [
        (
            "nginx:\n  existingConfigMap: client-nginx\nproxy:\n  runtimeUpstream: http://mindroom-runtime:8765\n",
            "proxy.runtimeUpstream requires the chart-managed nginx config",
        ),
        (
            "proxy:\n  homeserverUpstream: http://matrix-mindroom-tuwunel:8008/_matrix\n",
            "proxy.homeserverUpstream must be an http:// or https:// URL without a path",
        ),
        (
            "proxy:\n  runtimeUpstream: mindroom-runtime:8765\n",
            "proxy.runtimeUpstream must be an http:// or https:// URL without a path",
        ),
        (
            "proxy:\n  runtimeUpstream: http://$host:8765\n",
            "proxy.runtimeUpstream must be an http:// or https:// URL without a path",
        ),
    ],
    ids=["existing-nginx", "path", "no-scheme", "variable"],
)
def test_client_chart_rejects_origin_proxy_upstreams_it_cannot_forward_to(
    tmp_path: Path,
    values: str,
    error: str,
) -> None:
    """An upstream nginx would rewrite paths for, reject at startup, or never receive fails the render instead."""
    completed = _run_helm_template(
        Path("cluster/k8s/client"),
        release_name="mindroom-client",
        values_files=_values_files(tmp_path, values),
    )

    assert completed.returncode != 0
    assert error in completed.stderr
