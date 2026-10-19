"""Tests for planning which shared directories a dedicated worker mounts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from structlog.testing import capture_logs

from mindroom.runtime_resolution import shared_knowledge_path
from mindroom.shared_directories.store import add_shared_directory_member, create_shared_directory
from mindroom.tool_system.skills import clear_skill_cache, get_plugin_skill_roots, set_plugin_skill_roots
from mindroom.tool_system.worker_routing import ToolExecutionIdentity, resolve_unscoped_worker_key, resolve_worker_key
from mindroom.workers.backends._dedicated_worker_common import (
    SharedDirectoryMount,
    plan_shared_directory_mounts,
    plan_skill_directory_mounts,
    resolved_agent_policies_from_config_data,
)
from tests.conftest import test_runtime_paths

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mindroom.constants import RuntimePaths

_WORKER_ROOT = Path("/app/worker")


def _identity(agent_name: str) -> ToolExecutionIdentity:
    return ToolExecutionIdentity(
        channel="matrix",
        agent_name=agent_name,
        requester_id="@user:example.org",
        room_id="!room:example.org",
        thread_id=None,
        resolved_thread_id=None,
        session_id=None,
        tenant_id="tenant",
    )


def _scoped_key(scope: str, agent_name: str) -> str:
    worker_key = resolve_worker_key(scope, _identity(agent_name), agent_name=agent_name)  # type: ignore[arg-type]
    assert worker_key is not None
    return worker_key


def _agents(**scopes: str | None) -> dict[str, object]:
    return {
        "agents": {
            name: {"display_name": name, **({"worker_scope": scope} if scope else {})} for name, scope in scopes.items()
        },
    }


def _shared_report(runtime_paths: RuntimePaths, *members: str) -> Path:
    create_shared_directory(runtime_paths, owner="alice", name="report")
    for member in members:
        add_shared_directory_member(runtime_paths, owner="alice", name="report", member=member)
    return runtime_paths.storage_root / "shared_directories" / "alice" / "report"


def _plan(
    runtime_paths: RuntimePaths,
    worker_key: str,
    config_data: dict[str, object],
) -> tuple[SharedDirectoryMount, ...]:
    return plan_shared_directory_mounts(
        worker_key=worker_key,
        runtime_paths=runtime_paths,
        worker_visible_shared_storage_root=_WORKER_ROOT,
        resolved_agent_policies=resolved_agent_policies_from_config_data(config_data),
    )


def test_owner_mounts_read_write_and_member_read_only(tmp_path: Path) -> None:
    """The owner's worker writes the directory; an invited agent's worker only reads it."""
    runtime_paths = test_runtime_paths(tmp_path)
    share = _shared_report(runtime_paths, "bob")
    config_data = _agents(alice=None, bob=None, dave=None)
    visible = _WORKER_ROOT / "shared_directories" / "alice" / "report"

    assert _plan(runtime_paths, resolve_unscoped_worker_key("alice"), config_data) == (
        SharedDirectoryMount(local_path=share, worker_visible_path=visible, read_only=False),
    )
    assert _plan(runtime_paths, resolve_unscoped_worker_key("bob"), config_data) == (
        SharedDirectoryMount(local_path=share, worker_visible_path=visible, read_only=True),
    )
    assert _plan(runtime_paths, resolve_unscoped_worker_key("dave"), config_data) == ()


@pytest.mark.parametrize("scope", ["shared", "user_agent"])
def test_scoped_workers_mount_their_agents_directories(tmp_path: Path, scope: str) -> None:
    """Shared and user-agent workers mount the directories of the agent they serve."""
    runtime_paths = test_runtime_paths(tmp_path)
    _shared_report(runtime_paths, "bob")
    config_data = _agents(alice=scope, bob=scope)

    assert [mount.read_only for mount in _plan(runtime_paths, _scoped_key(scope, "alice"), config_data)] == [False]
    assert [mount.read_only for mount in _plan(runtime_paths, _scoped_key(scope, "bob"), config_data)] == [True]


def test_user_worker_addressing_owner_and_member_mounts_read_write(tmp_path: Path) -> None:
    """A user worker serves every user-scope agent, so it mounts what any of them owns read-write."""
    runtime_paths = test_runtime_paths(tmp_path)
    _shared_report(runtime_paths, "bob")
    create_shared_directory(runtime_paths, owner="carl", name="notes")
    add_shared_directory_member(runtime_paths, owner="carl", name="notes", member="bob")
    config_data = _agents(alice="user", bob="user", carl=None)

    mounts = _plan(runtime_paths, _scoped_key("user", "alice"), config_data)

    assert [(mount.worker_visible_path.parent.name, mount.read_only) for mount in mounts] == [
        ("alice", False),
        ("carl", True),
    ]


def test_planner_ignores_private_and_unknown_agents(tmp_path: Path) -> None:
    """Directories of an owner that became private or left the config are not mounted, nor for such members."""
    runtime_paths = test_runtime_paths(tmp_path)
    _shared_report(runtime_paths, "bob", "carol")
    private_owner = {
        "agents": {
            "alice": {"display_name": "alice", "private": {"per": "user"}},
            "bob": {"display_name": "bob"},
        },
    }
    private_member = {
        "agents": {
            "alice": {"display_name": "alice"},
            "carol": {"display_name": "carol", "private": {"per": "user"}, "worker_scope": "user_agent"},
        },
    }

    assert _plan(runtime_paths, resolve_unscoped_worker_key("bob"), private_owner) == ()
    assert _plan(runtime_paths, resolve_unscoped_worker_key("bob"), _agents(bob=None)) == ()
    assert _plan(runtime_paths, _scoped_key("user_agent", "carol"), private_member) == ()


def test_planner_skips_missing_and_linked_directories(tmp_path: Path) -> None:
    """A directory removed or replaced by a link is never mounted, and the skip is logged."""
    runtime_paths = test_runtime_paths(tmp_path)
    share = _shared_report(runtime_paths, "bob")
    create_shared_directory(runtime_paths, owner="alice", name="gone")
    (share.parent / "gone").rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    share.rmdir()
    share.symlink_to(outside)

    with capture_logs() as logs:
        mounts = _plan(runtime_paths, resolve_unscoped_worker_key("alice"), _agents(alice=None, bob=None))

    assert mounts == ()
    assert sorted(entry["directory"] for entry in logs if entry["log_level"] == "warning") == [
        "alice/gone",
        "alice/report",
    ]


def test_knowledge_inside_a_shared_directory_never_follows_a_planted_link(tmp_path: Path) -> None:
    """Owners' workers write shared directories, so the primary never follows a link found there."""
    runtime_paths = test_runtime_paths(tmp_path)
    share = _shared_report(runtime_paths)
    secret = tmp_path / "primary-only"
    secret.mkdir()
    (share / "docs").symlink_to(secret, target_is_directory=True)

    with pytest.raises(ValueError, match="shared knowledge base path"):
        shared_knowledge_path(str(share / "docs"), runtime_paths)


@pytest.fixture
def _restore_plugin_skill_roots() -> Iterator[None]:
    """Plugin skill roots are process-wide, so put them back after the test."""
    original = get_plugin_skill_roots()
    yield
    set_plugin_skill_roots(original)
    clear_skill_cache()


def _skill(skills_root: Path, name: str, *, script: bool) -> Path:
    skill = skills_root / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {name}\n---\n")
    if script:
        (skill / "scripts").mkdir()
        (skill / "scripts" / "run.sh").write_text("#!/bin/sh\n")
    return skill


def _plan_skills(
    runtime_paths: RuntimePaths,
    worker_key: str,
    skills_by_agent: dict[str, list[str]],
) -> tuple[SharedDirectoryMount, ...]:
    config_data = {
        "agents": {name: {"display_name": name, "skills": skills} for name, skills in skills_by_agent.items()},
    }
    return plan_skill_directory_mounts(
        worker_key=worker_key,
        runtime_paths=runtime_paths,
        worker_visible_shared_storage_root=_WORKER_ROOT,
        resolved_agent_policies=resolved_agent_policies_from_config_data(config_data),
        config_data=config_data,
    )


@pytest.mark.usefixtures("_restore_plugin_skill_roots")
def test_workers_mount_the_scripted_skills_their_agents_list_read_only(tmp_path: Path) -> None:
    """A worker sees each skill with scripts its agents list at the path below the storage mount, and nothing else."""
    runtime_paths = test_runtime_paths(tmp_path)
    bundle = runtime_paths.storage_root / "config-bundle" / "skills"
    audit = _skill(bundle, "audit", script=True)
    _skill(bundle, "guide", script=False)
    deploy = _skill(bundle, "deploy", script=True)
    # Outside the storage root, so no worker can mount it.
    elsewhere = tmp_path / "elsewhere"
    _skill(elsewhere, "lint", script=True)
    set_plugin_skill_roots([bundle, elsewhere])
    skills_by_agent = {"alice": ["audit", "guide", "lint"], "bob": ["deploy"], "dave": []}

    assert _plan_skills(runtime_paths, resolve_unscoped_worker_key("alice"), skills_by_agent) == (
        SharedDirectoryMount(audit, _WORKER_ROOT / "config-bundle" / "skills" / "audit", read_only=True),
    )
    assert _plan_skills(runtime_paths, resolve_unscoped_worker_key("bob"), skills_by_agent) == (
        SharedDirectoryMount(deploy, _WORKER_ROOT / "config-bundle" / "skills" / "deploy", read_only=True),
    )
    assert _plan_skills(runtime_paths, resolve_unscoped_worker_key("dave"), skills_by_agent) == ()


@pytest.mark.usefixtures("_restore_plugin_skill_roots")
def test_a_skill_directory_reached_through_a_link_is_not_mounted(tmp_path: Path) -> None:
    """Container runtimes follow links in mount sources, so a linked skill directory is skipped and logged."""
    runtime_paths = test_runtime_paths(tmp_path)
    bundle = runtime_paths.storage_root / "config-bundle" / "skills"
    bundle.mkdir(parents=True)
    target = _skill(runtime_paths.storage_root / "elsewhere", "audit", script=True)
    (bundle / "audit").symlink_to(target, target_is_directory=True)
    set_plugin_skill_roots([bundle])

    with capture_logs() as logs:
        mounts = _plan_skills(runtime_paths, resolve_unscoped_worker_key("alice"), {"alice": ["audit"]})

    assert mounts == ()
    assert [entry["directory"] for entry in logs if entry["log_level"] == "warning"] == ["config-bundle/skills/audit"]
