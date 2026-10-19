"""The agent's own shell tool, through which ``get_skill_script`` runs every skill script."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from agno.tools.function import FunctionCall

from mindroom.approval.tool_approval import approval_gated

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from agno.tools.function import Function, FunctionExecutionResult

# Bundled skills ship with MindRoom, workspace skills live in the agent's workspace,
# and configured skills come from plugins and the user skills directory.
SkillOrigin = Literal["bundled", "workspace", "configured"]


@dataclass(frozen=True)
class SkillScriptShell:
    """The agent's ``run_shell_command``, with its hooks, approval, worker routing, environment, and limits."""

    function: Function
    storage_root: Path
    # Where the shell's worker sees ``storage_root``: None when the shell runs in this process, and a lookup
    # returning None for a shared runner. Looked up on each use, so an incomplete worker backend config fails
    # that call instead of agent creation.
    worker_storage_root: Callable[[], Path | None] | None = None

    @property
    def needs_approval(self) -> bool:
        """Whether the shell's calls wait for approval, which a skill function cannot wait for."""
        return approval_gated(self.function)

    def visible_path(self, path: Path, origin: SkillOrigin) -> Path | None:
        """Return where the shell finds ``path``, or None when its worker cannot see it.

        Worker images ship the bundled skills where this process has them. Dedicated workers mount
        workspaces, and the directories of the configured skills with scripts that their agents list,
        at their place below the storage mount, and a shared runner sees workspaces where this process
        does but no configured skill directory.
        """
        if self.worker_storage_root is None or origin == "bundled":
            return path
        if not path.is_relative_to(self.storage_root):
            return None
        worker_root = self.worker_storage_root()
        if worker_root is None:
            return path if origin == "workspace" else None
        return worker_root / path.relative_to(self.storage_root)

    async def run(
        self,
        command: str,
        *,
        timeout: int,  # noqa: ASYNC109 - the command's own time limit, which the shell enforces
    ) -> FunctionExecutionResult:
        """Run ``command`` as one call of the shell, so it goes where the agent's own shell commands go."""
        return await FunctionCall(function=self.function, arguments={"args": command, "timeout": timeout}).aexecute()
