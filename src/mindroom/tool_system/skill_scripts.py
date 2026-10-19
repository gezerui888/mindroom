"""The agent's own shell tool, through which ``get_skill_script`` runs every skill script."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agno.tools.function import FunctionCall

from mindroom.approval.tool_approval import approval_gated

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from agno.tools.function import Function, FunctionExecutionResult


@dataclass(frozen=True)
class SkillScriptShell:
    """The agent's ``run_shell_command``, with its hooks, approval, worker routing, environment, and limits."""

    function: Function
    storage_root: Path
    # Where the shell's worker sees ``storage_root``, or None when the shell sees this process's paths.
    # Called on first use, so an incomplete worker backend config fails that call instead of agent creation.
    worker_storage_root: Callable[[], Path | None] | None = None

    @property
    def needs_approval(self) -> bool:
        """Whether the shell's calls wait for approval, which a skill function cannot wait for."""
        return approval_gated(self.function)

    def visible_path(self, path: Path) -> Path:
        """Return where the shell finds ``path``: a worker mounts the storage root at its own path."""
        worker_root = None if self.worker_storage_root is None else self.worker_storage_root()
        if worker_root is None or not path.is_relative_to(self.storage_root):
            return path
        return worker_root / path.relative_to(self.storage_root)

    async def run(
        self,
        command: str,
        *,
        timeout: int,  # noqa: ASYNC109 - the command's own time limit, which the shell enforces
    ) -> FunctionExecutionResult:
        """Run ``command`` as one call of the shell, so it goes where the agent's own shell commands go."""
        return await FunctionCall(function=self.function, arguments={"args": command, "timeout": timeout}).aexecute()
