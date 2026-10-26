"""Tests for the personal_agent_manager toolkit, which lets a personal agent change itself for its owner."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
import yaml

from mindroom.agents import create_agent
from mindroom.config.personal_agents import personal_agent_roster_path, read_personal_agent_roster
from mindroom.custom_tools.personal_agent_manager import PersonalAgentManagerTools
from mindroom.message_target import MessageTarget
from mindroom.tool_system.agno_toolkits import adapt_toolkit
from mindroom.tool_system.runtime_context import tool_runtime_context
from tests.agent_tool_helpers import offered_function
from tests.authorization_helpers import make_test_tool_runtime_context
from tests.conftest import make_conversation_reader_mock, make_relation_lookup
from tests.identity_helpers import persist_entity_accounts
from tests.test_personal_agent_owner_edits import _AGENT, _OWNER, _setup

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from mindroom.config.main import Config
    from mindroom.constants import RuntimePaths


@contextmanager
def _as_requester(
    tool: PersonalAgentManagerTools,
    config: Config,
    paths: RuntimePaths,
    *,
    requester_id: str = _OWNER,
    orchestrator: object | None = None,
) -> Iterator[None]:
    """Bind one live requester, and optionally a running orchestrator, for the toolkit's calls."""
    context = make_test_tool_runtime_context(
        agent_name=tool.agent_name,
        target=MessageTarget.resolve(room_id="!room:example.org", thread_id="$thread", reply_to_event_id="$request"),
        requester_id=requester_id,
        client=MagicMock(),
        config=config,
        runtime_paths=paths,
        relations=make_relation_lookup(),
        conversation_reader=make_conversation_reader_mock(),
        orchestrator=orchestrator,
    )
    with tool_runtime_context(context):
        yield


def _orchestrator() -> Any:  # noqa: ANN401 - a stub for the orchestrator runtime the context carries
    return MagicMock()


def test_update_agent_asks_no_confirmation(tmp_path: Path) -> None:
    """An approval card needs the room's router, which a personal room lacks, so a confirmed update would always fail."""
    config, paths = _setup(tmp_path)
    persist_entity_accounts(config, paths)
    build = create_agent(
        _AGENT,
        config,
        paths,
        execution_identity=None,
        knowledge=None,
        active_model_name=None,
        supports_native_tool_approval=True,
    )

    assert offered_function(build, "update_agent").requires_confirmation is not True
    assert offered_function(build, "list_my_agents").requires_confirmation is not True


def test_update_agent_schema_lets_a_model_send_text_and_name_lists() -> None:
    """The offered schema types each change value, so a model is not led to send empty objects for text or lists."""
    function = next(
        function
        for function in adapt_toolkit(PersonalAgentManagerTools(agent_name=_AGENT)).functions
        if function.name == "update_agent"
    )

    assert function.parameters["required"] == ["changes"]
    value_types = function.parameters["properties"]["changes"]["additionalProperties"]["anyOf"]

    assert {"type": "string"} in value_types
    assert {"type": "array", "items": {"type": "string"}} in value_types
    assert {"type": "null"} in value_types


@pytest.mark.asyncio
async def test_update_agent_writes_and_requests_reload(tmp_path: Path) -> None:
    """The owner's change lands in the roster and the running orchestrator is asked to reload once."""
    config, paths = _setup(tmp_path)
    tool = PersonalAgentManagerTools(agent_name=_AGENT)
    orchestrator = _orchestrator()

    with _as_requester(tool, config, paths, orchestrator=orchestrator):
        result = await tool.update_agent(_AGENT, changes={"role": "Mine", "tools": ["calculator"]})

    assert result == f"Updated {_AGENT}: role, tools."
    (entry,) = read_personal_agent_roster(paths)
    assert entry.overrides == {"role": "Mine", "tools": ["calculator"]}
    orchestrator.config_reload.request_reload.assert_called_once_with()


@pytest.mark.asyncio
async def test_update_agent_without_an_orchestrator_still_writes(tmp_path: Path) -> None:
    """A detached context has no orchestrator to reload, and the change is stored all the same."""
    config, paths = _setup(tmp_path)
    tool = PersonalAgentManagerTools(agent_name=_AGENT)

    with _as_requester(tool, config, paths):
        result = await tool.update_agent(_AGENT, changes={"role": "Mine"})

    assert not result.startswith("Error")
    assert read_personal_agent_roster(paths)[0].overrides == {"role": "Mine"}


@pytest.mark.asyncio
async def test_update_agent_refusal_is_returned_and_requests_no_reload(tmp_path: Path) -> None:
    """A change outside the template's grant comes back as an error string, with nothing stored or reloaded."""
    config, paths = _setup(tmp_path)
    tool = PersonalAgentManagerTools(agent_name=_AGENT)
    orchestrator = _orchestrator()

    with _as_requester(tool, config, paths, orchestrator=orchestrator):
        result = await tool.update_agent(_AGENT, changes={"tools": ["browser"]})

    assert result == "Error: tools cannot include browser; allowed: calculator, shell"
    assert read_personal_agent_roster(paths)[0].overrides == {}
    orchestrator.config_reload.request_reload.assert_not_called()


@pytest.mark.asyncio
async def test_update_agent_with_no_changes_does_nothing(tmp_path: Path) -> None:
    """An empty change set is an error, and neither rewrites the roster nor asks for a reload."""
    config, paths = _setup(tmp_path)
    tool = PersonalAgentManagerTools(agent_name=_AGENT)
    orchestrator = _orchestrator()
    roster_before = personal_agent_roster_path(paths).read_bytes()

    with _as_requester(tool, config, paths, orchestrator=orchestrator):
        result = await tool.update_agent(_AGENT, changes={})

    assert result == "Error: no changes given"
    assert personal_agent_roster_path(paths).read_bytes() == roster_before
    orchestrator.config_reload.request_reload.assert_not_called()


@pytest.mark.asyncio
async def test_update_agent_defaults_to_itself(tmp_path: Path) -> None:
    """Leaving agent_name out changes the calling agent's own entry."""
    config, paths = _setup(tmp_path)
    tool = PersonalAgentManagerTools(agent_name=_AGENT)
    orchestrator = _orchestrator()

    with _as_requester(tool, config, paths, orchestrator=orchestrator):
        result = await tool.update_agent(changes={"role": "Mine"})

    assert result == f"Updated {_AGENT}: role."
    assert read_personal_agent_roster(paths)[0].overrides == {"role": "Mine"}
    orchestrator.config_reload.request_reload.assert_called_once_with()


@pytest.mark.asyncio
async def test_update_agent_refuses_a_non_owner(tmp_path: Path) -> None:
    """Someone else in the room cannot change the owner's agent through it."""
    config, paths = _setup(tmp_path)
    tool = PersonalAgentManagerTools(agent_name=_AGENT)
    orchestrator = _orchestrator()

    with _as_requester(tool, config, paths, requester_id="@bob:example.org", orchestrator=orchestrator):
        result = await tool.update_agent(_AGENT, changes={"role": "Bob's"})

    assert result == f"Error: Only {_OWNER} can change this agent"
    assert read_personal_agent_roster(paths)[0].overrides == {}
    orchestrator.config_reload.request_reload.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong_name", ["helper", "Alice's Mind"])
async def test_update_agent_refuses_another_agent(tmp_path: Path, wrong_name: str) -> None:
    """An agent changes only itself, so another agent's name or its own display name is refused, naming its key."""
    config, paths = _setup(tmp_path)
    tool = PersonalAgentManagerTools(agent_name=_AGENT)
    orchestrator = _orchestrator()

    with _as_requester(tool, config, paths, orchestrator=orchestrator):
        result = await tool.update_agent(wrong_name, changes={"role": "Mine"})

    assert result == f"Error: you can only change yourself ({_AGENT}); omit agent_name to change yourself"
    assert read_personal_agent_roster(paths)[0].overrides == {}
    orchestrator.config_reload.request_reload.assert_not_called()


@pytest.mark.asyncio
async def test_manager_refuses_an_authored_agent(tmp_path: Path) -> None:
    """Built for an agent written in the config, both functions return an error and the roster is untouched."""
    config, paths = _setup(tmp_path)
    roster_file = personal_agent_roster_path(paths)
    before = roster_file.read_bytes()
    tool = PersonalAgentManagerTools(agent_name="helper")

    with _as_requester(tool, config, paths):
        listed = await tool.list_my_agents()
        updated = await tool.update_agent("helper", changes={"role": "Mine"})

    assert listed == "Error: helper is not a personal agent"
    assert updated == "Error: helper is not a personal agent"
    assert roster_file.read_bytes() == before


@pytest.mark.asyncio
async def test_update_agent_without_runtime_context_fails_closed(tmp_path: Path) -> None:
    """With no live requester the toolkit refuses instead of guessing who is asking."""
    _, paths = _setup(tmp_path)
    tool = PersonalAgentManagerTools(agent_name=_AGENT)

    result = await tool.update_agent(_AGENT, changes={"role": "Mine"})

    assert result.startswith("Error")
    assert read_personal_agent_roster(paths)[0].overrides == {}


@pytest.mark.asyncio
async def test_list_my_agents_without_runtime_context_fails_closed(tmp_path: Path) -> None:
    """Listing needs the requester too, so it refuses without a context."""
    _setup(tmp_path)
    tool = PersonalAgentManagerTools(agent_name=_AGENT)

    assert (await tool.list_my_agents()).startswith("Error")


@pytest.mark.asyncio
async def test_list_my_agents_returns_the_summary_as_yaml(tmp_path: Path) -> None:
    """The owner gets the editable values and the names the template grants."""
    config, paths = _setup(tmp_path, skills=["review"])
    tool = PersonalAgentManagerTools(agent_name=_AGENT)

    with _as_requester(tool, config, paths):
        result = await tool.list_my_agents()

    summary = yaml.safe_load(result)
    assert summary["agent_name"] == _AGENT
    assert summary["allowed"]["tools"] == ["calculator", "shell"]
    assert summary["allowed"]["skills"] == ["review"]
    assert summary["editable"]["role"] == "R"


@pytest.mark.asyncio
async def test_list_my_agents_refuses_a_non_owner(tmp_path: Path) -> None:
    """Someone else in the room cannot read the owner's agent settings through it."""
    config, paths = _setup(tmp_path)
    tool = PersonalAgentManagerTools(agent_name=_AGENT)

    with _as_requester(tool, config, paths, requester_id="@bob:example.org"):
        result = await tool.list_my_agents()

    assert result == f"Error: Only {_OWNER} can change this agent"
