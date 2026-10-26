"""Tests for validating and storing an owner's edits to their personal agent."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from mindroom.config.main import Config
from mindroom.config.personal_agents import (
    PersonalAgentRosterEntry,
    personal_agent_roster_lock_path,
    read_personal_agent_roster,
)
from mindroom.file_locks import file_lock_is_held
from mindroom.personal_agents import (
    _append_personal_agent_roster_entry,
    _owner_agent_summary,
    _PersonalAgentEditError,
    _update_owner_overrides,
    _write_personal_agent_roster,
)
from tests.conftest import test_runtime_paths
from tests.test_personal_agents_config import BASE, _entry

if TYPE_CHECKING:
    from pathlib import Path

    from mindroom.constants import RuntimePaths

_OWNER = "@alice:example.org"
_AGENT = "mind_alice"
_BRIDGE_ALIAS = "@bridge-alice:example.org"


def _setup(
    tmp_path: Path,
    *,
    owner_edits: bool = True,
    entry: PersonalAgentRosterEntry | None = None,
    aliases: dict[str, list[str]] | None = None,
    **agent: object,
) -> tuple[Config, RuntimePaths]:
    data = deepcopy(BASE)
    data["personal_agents"]["mind"] = {
        "owner_edits": owner_edits,
        "agent": {"display_name": "Mind", "role": "R", "model": "default", "tools": ["calculator", "shell"], **agent},
    }
    data["agents"] = {"helper": {"display_name": "Helper"}}
    if aliases is not None:
        data["authorization"] = {"aliases": aliases}
    paths = test_runtime_paths(tmp_path)
    _append_personal_agent_roster_entry(paths, entry or _entry(_OWNER))
    return Config.validate_with_runtime(data, paths), paths


def _edit(config: Config, paths: RuntimePaths, changes: dict[str, Any], **kwargs: Any) -> PersonalAgentRosterEntry:  # noqa: ANN401
    return _update_owner_overrides(
        config,
        paths,
        agent_name=kwargs.pop("agent_name", _AGENT),
        requester_id=kwargs.pop("requester_id", _OWNER),
        changes=changes,
    )


def _overrides(paths: RuntimePaths) -> dict[str, Any]:
    (entry,) = read_personal_agent_roster(paths)
    return entry.overrides


def test_owner_can_edit_role_and_tools(tmp_path: Path) -> None:
    """The owner's changes become the roster entry's overrides, on disk and in the returned entry."""
    config, paths = _setup(tmp_path)
    changes = {"role": "Mine", "tools": ["calculator"]}

    entry = _edit(config, paths, changes)

    assert entry.overrides == changes
    assert read_personal_agent_roster(paths) == [entry]


def test_non_owner_is_refused(tmp_path: Path) -> None:
    """Someone else cannot change the agent, and the roster stays as it was."""
    config, paths = _setup(tmp_path)
    before = read_personal_agent_roster(paths)

    with pytest.raises(_PersonalAgentEditError, match=re.escape("Only @alice:example.org can change this agent")):
        _edit(config, paths, {"role": "Mine"}, requester_id="@bob:example.org")

    assert read_personal_agent_roster(paths) == before


def test_bridge_alias_of_owner_is_allowed(tmp_path: Path) -> None:
    """A bridged account that authorization maps to the owner may edit the owner's agent."""
    config, paths = _setup(tmp_path, aliases={_OWNER: [_BRIDGE_ALIAS]})

    _edit(config, paths, {"role": "Mine"}, requester_id=_BRIDGE_ALIAS)

    assert _overrides(paths) == {"role": "Mine"}


def test_out_of_ceiling_tool_is_refused(tmp_path: Path) -> None:
    """A tool outside the template's grant is named, with the names the owner may choose from."""
    config, paths = _setup(tmp_path)

    with pytest.raises(_PersonalAgentEditError) as raised:
        _edit(config, paths, {"tools": ["calculator", "browser"]})

    assert str(raised.value) == "tools cannot include browser; allowed: calculator, shell"
    assert _overrides(paths) == {}


@pytest.mark.parametrize("field", ["skills", "knowledge_bases", "delegate_to"])
def test_out_of_ceiling_names_are_refused_for_every_subset_field(tmp_path: Path, field: str) -> None:
    """A subset field names what is disallowed and says when the template grants nothing."""
    config, paths = _setup(tmp_path)

    with pytest.raises(_PersonalAgentEditError) as raised:
        _edit(config, paths, {field: ["x", "y"]})

    assert str(raised.value) == f"{field} cannot include x, y; allowed: none"


def test_non_list_subset_value_is_refused(tmp_path: Path) -> None:
    """A subset field must be a list of names, so a bare string is refused with the allowed names."""
    config, paths = _setup(tmp_path)

    with pytest.raises(_PersonalAgentEditError) as raised:
        _edit(config, paths, {"tools": "shell"})

    assert str(raised.value) == "tools must be a list of names; allowed: calculator, shell"


def test_fixed_field_is_refused(tmp_path: Path) -> None:
    """A field the owner may not edit is refused by name, even when set to nothing."""
    config, paths = _setup(tmp_path)

    for changes in ({"model": "x"}, {"model": None}, {"access": {"users": ["@bob:example.org"]}}):
        with pytest.raises(_PersonalAgentEditError) as raised:
            _edit(config, paths, changes)
        assert str(raised.value) == f"{next(iter(changes))} is not editable"

    assert _overrides(paths) == {}


def test_invalid_free_value_is_refused(tmp_path: Path) -> None:
    """A free field value the agent could not load with is refused by field name."""
    config, paths = _setup(tmp_path)

    with pytest.raises(_PersonalAgentEditError, match="thread_mode has an invalid value"):
        _edit(config, paths, {"thread_mode": "nonsense", "role": "Mine"})

    assert _overrides(paths) == {}


def test_free_value_that_conflicts_with_a_stored_edit_is_refused(tmp_path: Path) -> None:
    """Two history limits that exclude each other cannot both be stored, so the later one is refused."""
    config, paths = _setup(tmp_path, entry=_entry(_OWNER, overrides={"num_history_runs": 3}))

    with pytest.raises(_PersonalAgentEditError, match="num_history_messages has an invalid value"):
        _edit(config, paths, {"num_history_messages": 10})

    assert _overrides(paths) == {"num_history_runs": 3}


def test_every_refusal_is_reported_and_none_is_stored(tmp_path: Path) -> None:
    """One bad field refuses the whole request, and the message names each one."""
    config, paths = _setup(tmp_path)

    with pytest.raises(_PersonalAgentEditError) as raised:
        _edit(config, paths, {"role": "Mine", "model": "x", "tools": ["browser"]})

    assert str(raised.value) == "model is not editable; tools cannot include browser; allowed: calculator, shell"
    assert _overrides(paths) == {}


def test_a_stale_stored_edit_does_not_block_another_field(tmp_path: Path) -> None:
    """A tool the admin removed after the owner chose it is not the requested change, so it does not refuse one."""
    config, paths = _setup(tmp_path, entry=_entry(_OWNER, overrides={"tools": ["calculator", "gone"]}))

    _edit(config, paths, {"role": "Mine"})

    assert _overrides(paths) == {"tools": ["calculator", "gone"], "role": "Mine"}


def test_edits_refused_while_owner_edits_is_off(tmp_path: Path) -> None:
    """Without the template's switch the owner may not edit, and nothing is stored."""
    config, paths = _setup(tmp_path, owner_edits=False)

    with pytest.raises(_PersonalAgentEditError, match=re.escape("Owner edits are off for this agent's template")):
        _edit(config, paths, {"role": "Mine"})

    assert _overrides(paths) == {}


def test_authored_agent_is_refused(tmp_path: Path) -> None:
    """An agent written in the config belongs to no owner, so it cannot be edited this way."""
    config, paths = _setup(tmp_path)

    with pytest.raises(_PersonalAgentEditError, match="helper is not a personal agent"):
        _edit(config, paths, {"role": "Mine"}, agent_name="helper")
    with pytest.raises(_PersonalAgentEditError, match="missing is not a personal agent"):
        _edit(config, paths, {"role": "Mine"}, agent_name="missing")


def test_agent_missing_from_the_roster_is_refused(tmp_path: Path) -> None:
    """An agent the roster no longer has cannot be edited, even while the loaded config still has it."""
    config, paths = _setup(tmp_path)
    _write_personal_agent_roster(paths, [])

    with pytest.raises(_PersonalAgentEditError, match="mind_alice is not a personal agent"):
        _edit(config, paths, {"role": "Mine"})


def test_none_clears_an_override(tmp_path: Path) -> None:
    """Setting a field to None gives it back to the template."""
    config, paths = _setup(tmp_path)

    _edit(config, paths, {"role": "Mine"})
    assert _overrides(paths) == {"role": "Mine"}
    entry = _edit(config, paths, {"role": None})

    assert "role" not in entry.overrides
    assert _overrides(paths) == {}


def test_changes_merge_into_the_current_roster(tmp_path: Path) -> None:
    """The roster is read again under the lock, so a change made since the config loaded is kept."""
    config, paths = _setup(tmp_path)
    _write_personal_agent_roster(paths, [_entry(_OWNER, overrides={"role": "Newer"})])

    entry = _edit(config, paths, {"display_name": "Mine"})

    assert entry.overrides == {"role": "Newer", "display_name": "Mine"}
    assert _overrides(paths) == entry.overrides


def test_only_the_owners_entry_changes(tmp_path: Path) -> None:
    """Another user's entry from the same template is left alone and keeps its place."""
    config, paths = _setup(tmp_path)
    _append_personal_agent_roster_entry(paths, _entry("@bob:example.org"))

    _edit(config, paths, {"role": "Mine"})

    assert [(entry.user_id, entry.overrides) for entry in read_personal_agent_roster(paths)] == [
        (_OWNER, {"role": "Mine"}),
        ("@bob:example.org", {}),
    ]


def test_the_roster_is_written_under_its_lock(tmp_path: Path) -> None:
    """The write happens while the roster lock is held, so a concurrent provisioning cannot be overwritten."""
    config, paths = _setup(tmp_path)
    held: list[bool] = []

    def write(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        held.append(file_lock_is_held(personal_agent_roster_lock_path(paths)))
        _write_personal_agent_roster(*args, **kwargs)

    with patch("mindroom.personal_agents._write_personal_agent_roster", new=write):
        _edit(config, paths, {"role": "Mine"})

    assert held == [True]
    assert not file_lock_is_held(personal_agent_roster_lock_path(paths))


def test_owner_agent_summary_lists_allowed_names(tmp_path: Path) -> None:
    """The summary lists the names the template grants and the owner's clamped current values."""
    entry = _entry(_OWNER, overrides={"role": "Mine", "tools": ["calculator", "gone"], "model": "x"})
    config, paths = _setup(tmp_path, entry=entry, skills=["review"], instructions=["Be brief"])

    summary = _owner_agent_summary(config, paths, _AGENT)

    assert summary["agent_name"] == _AGENT
    assert summary["allowed"] == {
        "tools": ["calculator", "shell"],
        "skills": ["review"],
        "knowledge_bases": [],
        "delegate_to": [],
    }
    assert summary["editable"] == {
        "display_name": "Mind",
        "role": "Mine",
        "instructions": ["Be brief"],
        "markdown": None,
        "thread_mode": "thread",
        "num_history_runs": None,
        "num_history_messages": None,
        "show_tool_calls": None,
        "tools": ["calculator"],
        "skills": ["review"],
        "knowledge_bases": [],
        "delegate_to": [],
    }


def test_owner_agent_summary_refuses_what_an_edit_would(tmp_path: Path) -> None:
    """The summary refuses an authored agent and a template without owner edits, as an edit does."""
    config, paths = _setup(tmp_path)
    with pytest.raises(_PersonalAgentEditError, match="helper is not a personal agent"):
        _owner_agent_summary(config, paths, "helper")

    config, paths = _setup(tmp_path / "off", owner_edits=False)
    with pytest.raises(_PersonalAgentEditError, match=re.escape("Owner edits are off for this agent's template")):
        _owner_agent_summary(config, paths, _AGENT)
