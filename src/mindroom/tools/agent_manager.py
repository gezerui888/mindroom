"""Agent manager tool metadata registration.

Registers the ``agent_manager`` tool in the metadata registry for UI display.
The actual toolkit (``mindroom.custom_tools.agent_manager.AgentManagerTools``)
requires ``agent_name`` at instantiation and is injected directly in ``create_agent()``
for generated personal agents whose template allows owner edits, so it is NOT added to ``TOOL_REGISTRY``.
"""

from mindroom.tool_system.declarations import (
    SetupType,
    ToolCategory,
    ToolFileAccess,
    ToolMetadata,
    ToolStatus,
)
from mindroom.tool_system.registration import register_builtin_tool_metadata

register_builtin_tool_metadata(
    ToolMetadata(
        name="agent_manager",
        file_access=ToolFileAccess.NONE,
        display_name="Agent Manager",
        description="Let a personal agent change its own editable settings for its owner, within its template's grant",
        docs_url="https://docs.mindroom.chat/tools/agent-orchestration/#agent_manager",
        category=ToolCategory.DEVELOPMENT,
        status=ToolStatus.AVAILABLE,
        setup_type=SetupType.NONE,
        requires_primary_runtime=True,
        icon="Settings",
        icon_color="text-teal-500",
        config_fields=[],
        dependencies=[],
        function_names=("list_my_agents", "update_agent"),
    ),
)
