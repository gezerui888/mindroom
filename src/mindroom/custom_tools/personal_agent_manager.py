"""Personal agent manager tool: lets a personal agent change itself for its owner."""

from __future__ import annotations

from functools import partial

import yaml
from agno.tools import Toolkit

from mindroom.background_tasks import run_blocking_until_complete
from mindroom.personal_agents import PersonalAgentEditError, owner_agent_summary, update_owner_overrides
from mindroom.tool_system.runtime_context import get_tool_runtime_context

# Every value an owner can set: text, a number, a switch, a list of names, or null to give a field back to the template.
_ChangeValue = str | int | bool | list[str] | None

_NO_CONTEXT_ERROR = "Error: This tool only works while answering a request, and there is none."


class PersonalAgentManagerTools(Toolkit):
    """Tools that let a personal agent read and change its own owner-editable settings."""

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        super().__init__(
            name="personal_agent_manager",
            tools=[self.list_my_agents, self.update_agent],
        )

    async def list_my_agents(self) -> str:
        """List what the owner may change about this agent: its key, its current editable values and the allowed names.

        Call this before `update_agent`.
        The result has `agent_name` (this agent's own key), `editable` (the current values) and `allowed`
        (the names the template grants for tools, skills, knowledge_bases and delegate_to).
        display_name, role, instructions and similar fields are free text.
        Only this agent's owner may ask; anyone else gets an error.

        Returns:
            The agent's key, editable settings and allowed names as YAML, or an error message.

        """
        context = get_tool_runtime_context()
        if context is None:
            return _NO_CONTEXT_ERROR
        try:
            summary = await run_blocking_until_complete(
                partial(
                    owner_agent_summary,
                    context.current_config,
                    context.runtime_paths,
                    self.agent_name,
                    requester_id=context.requester_id,
                ),
            )
        except PersonalAgentEditError as exc:
            return f"Error: {exc}"
        return yaml.safe_dump(summary, default_flow_style=False, sort_keys=False)

    async def update_agent(self, agent_name: str | None = None, *, changes: dict[str, _ChangeValue]) -> str:
        """Change this agent's settings for its owner. Only the fields in `changes` are touched.

        Call `list_my_agents` first. Omit `agent_name`: an agent can only change itself.
        `changes` maps field names to new values, such as {"role": "Concise reviewer", "tools": ["calculator"]}.
        Send `tools`, `skills`, `knowledge_bases` and `delegate_to` as names taken from `allowed` in the
        `list_my_agents` result; for tools these are toolkit names such as `todo`, never function names such as `add_todo`.
        Send only the fields you change, because each value replaces the whole field,
        and a field set to null goes back to the template's value.
        To remove a tool, send `tools` with only the names to keep.
        Only this agent's owner may ask; anyone else gets an error.
        A request for a name outside the template's grant is refused whole, with the allowed names in the error.
        No confirmation is asked, because every edit stays within the template's grant and only the owner can make it.
        An administrator can add a `tool_approval` rule for confirmation, but its approval card needs the router
        in the room, and personal rooms do not have it by default.
        The change applies after a short configuration reload.

        Args:
            agent_name: Leave out to change this agent; a different name is refused.
            changes: Field names mapped to their new values, or to null to restore the template's value.

        Returns:
            Success message naming the changed fields, or an error message.

        """
        context = get_tool_runtime_context()
        if context is None:
            return _NO_CONTEXT_ERROR
        if agent_name not in (None, self.agent_name):
            return f"Error: you can only change yourself ({self.agent_name}); omit agent_name to change yourself"
        if not changes:
            return "Error: no changes given"
        try:
            await run_blocking_until_complete(
                partial(
                    update_owner_overrides,
                    context.current_config,
                    context.runtime_paths,
                    agent_name=self.agent_name,
                    requester_id=context.requester_id,
                    changes=changes,
                ),
            )
        except PersonalAgentEditError as exc:
            return f"Error: {exc}"
        if context.orchestrator is not None:
            context.orchestrator.config_reload.request_reload()
        return f"Updated {self.agent_name}: {', '.join(changes)}."
