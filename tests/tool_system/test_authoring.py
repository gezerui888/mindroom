"""MindRoom's tool authoring API: a toolkit's functions, their adaptation without Agno, and the call each runs for."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator  # noqa: TC003 - A tool's annotations resolve when its toolkit adapts.
from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from mindroom.constants import resolve_runtime_paths
from mindroom.history.types import HistoryScope
from mindroom.tool_system.agno_toolkits import Function, adapt_toolkit, agno_toolkit
from mindroom.tool_system.agno_toolkits import Toolkit as AgnoToolkit
from mindroom.tool_system.authoring import CurrentCall, Toolkit, current_call, tool, toolkit_functions
from mindroom.tool_system.declarations import ToolCategory, ToolFileAccess
from mindroom.tool_system.invocation import ToolCallContext, execute_tool_call
from mindroom.tool_system.metadata import get_tool_by_name
from mindroom.tool_system.output_files import OUTPUT_PATH_ARGUMENT
from mindroom.tool_system.registration import register_tool_with_metadata
from mindroom.tool_system.registry_state import capture_tool_registry_snapshot, restore_tool_registry_snapshot
from mindroom.tool_system.runtime_context import ToolRuntimeContext, tool_runtime_context
from tests.tool_system import agno_parity_support as support

if TYPE_CHECKING:
    from pathlib import Path

    from mindroom.tool_system.tool_function import ToolFunction


class GreeterTools(Toolkit):
    """A toolkit that names itself and its instructions as class attributes."""

    name = "greeter"
    instructions = "Greet people by the name they gave."

    @tool
    async def greet(self, name: str, times: int = 1) -> str:
        """Greet someone by name.

        Args:
            name: The person's name.
            times: How often to greet.

        """
        return " ".join([f"Hello, {name}!"] * times)

    @tool(name="farewell", description="Say goodbye to someone.")
    def bye(self, name: str) -> str:
        """Part with someone."""
        return f"Goodbye, {name}."

    def helper(self) -> str:
        """Help the toolkit, which offers it to no model."""
        return "not offered"


def shout(text: str) -> str:
    """Shout the text.

    Args:
        text: What to shout.

    """
    return text.upper()


_CONTEXT = ToolCallContext(run_id="run-1", session_id="session-1", user_id=None)


def _by_name(functions: tuple[ToolFunction, ...]) -> dict[str, ToolFunction]:
    return {function.name: function for function in functions}


def test_a_toolkits_functions_are_its_tool_methods_then_its_tools() -> None:
    """``@tool`` methods come in definition order under the names the model calls them by, then ``tools``."""
    toolkit = GreeterTools(tools=[shout])

    assert list(toolkit.functions) == ["greet", "farewell", "shout"]
    assert toolkit.functions["farewell"] == toolkit.bye
    assert toolkit.functions["shout"] is shout
    assert (toolkit.name, toolkit.instructions) == ("greeter", "Greet people by the name they gave.")


@pytest.mark.asyncio
async def test_tool_methods_stay_directly_callable() -> None:
    """``@tool`` returns the method itself, so code calls it as written."""
    toolkit = GreeterTools()

    assert await toolkit.greet("Ada", times=2) == "Hello, Ada! Hello, Ada!"
    assert toolkit.bye("Ada") == "Goodbye, Ada."


def test_a_subclass_passes_agnos_constructor_keywords() -> None:
    """A toolkit written for Agno's constructor keeps its ``name``, ``tools``, and ``instructions`` keywords."""

    class LookupTools(Toolkit):
        def __init__(self) -> None:
            super().__init__(name="lookup", tools=[self.find], instructions="Look things up.")

        def find(self, query: str) -> str:
            return query

    toolkit = LookupTools()

    assert (toolkit.name, toolkit.instructions, list(toolkit.functions)) == ("lookup", "Look things up.", ["find"])


def test_a_subclass_offers_its_bases_tool_methods_first() -> None:
    """Inherited ``@tool`` methods keep the base's order, a marked override keeps its place, and new ones follow."""

    class BaseTools(Toolkit):
        @tool
        def first(self) -> str:
            return "first"

        @tool
        def second(self) -> str:
            return "second"

    class SubTools(BaseTools):
        @tool
        def third(self) -> str:
            return "third"

        @tool
        def second(self) -> str:
            return "overridden"

    toolkit = SubTools()

    assert list(toolkit.functions) == ["first", "second", "third"]
    assert toolkit.functions["second"]() == "overridden"


def test_only_tool_marks_make_a_class_attribute_a_function() -> None:
    """An attribute that answers any attribute lookup, such as a lazy client proxy, is no tool function."""

    class AnswersEverything:
        def __getattr__(self, name: str) -> str:
            return name

    class ProxyHoldingTools(Toolkit):
        client = AnswersEverything()

        @tool
        def ping(self) -> str:
            return "pong"

    assert list(ProxyHoldingTools().functions) == ["ping"]


def test_two_functions_under_one_name_raise() -> None:
    """The model calls a function by its name, so a toolkit offers one function per name."""

    def farewell(name: str) -> str:
        return name

    with pytest.raises(ValueError, match="two functions named 'farewell'"):
        GreeterTools(tools=[farewell])


def test_an_owned_toolkit_adapts_without_agnos_entrypoint_processing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Schemas come from the owned generator, descriptions from ``@tool`` or the docstring, and nothing is injected."""

    def refuse(*_args: object, **_kwargs: object) -> None:
        msg = "An owned toolkit must not reach Agno's entrypoint processing"
        raise AssertionError(msg)

    monkeypatch.setattr(Function, "process_entrypoint", refuse)

    adapted = adapt_toolkit(GreeterTools(tools=[shout]), key="greeter")
    functions = _by_name(adapted.functions)

    # Agno's async-run order: the synchronous functions, then the asynchronous ones.
    assert list(functions) == ["farewell", "shout", "greet"]
    instructions = "Greet people by the name they gave."
    assert (adapted.name, adapted.instructions, adapted.guidance) == ("greeter", instructions, instructions)
    assert functions["farewell"].description == "Say goodbye to someone."
    assert functions["shout"].description == "Shout the text."
    assert functions["greet"].parameters == {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The person's name."},
            "times": {"type": "integer", "description": "How often to greet."},
        },
        "required": ["name"],
    }
    assert all(function.injected_params == {} and function.entrypoint_processed for function in adapted.functions)
    assert {function.toolkit for function in adapted.functions} == {"greeter"}


@pytest.mark.asyncio
async def test_an_owned_function_coerces_and_validates_its_arguments() -> None:
    """Arguments reach the body coerced to its annotations, and one that cannot be coerced fails the call."""
    greet = _by_name(adapt_toolkit(GreeterTools()).functions)["greet"]

    coerced = await execute_tool_call(greet, {"name": "Ada", "times": "2"}, _CONTEXT, middleware=())
    invalid = await execute_tool_call(greet, {"name": "Ada", "times": "twice"}, _CONTEXT, middleware=())

    assert (coerced.content, coerced.is_error) == ("Hello, Ada! Hello, Ada!", False)
    assert invalid.is_error
    assert "1 validation error for GreeterTools.greet\ntimes\n  Input should be a valid integer" in invalid.content


@pytest.mark.asyncio
async def test_an_owned_streaming_function_stays_an_async_generator() -> None:
    """A streaming body keeps running on the event loop, with its arguments coerced, and its chunks join."""

    class CounterTools(Toolkit):
        @tool
        async def count_up(self, limit: int) -> AsyncIterator[str]:
            """Count up to a limit."""
            for number in range(1, limit + 1):
                yield str(number)

    [count_up] = adapt_toolkit(CounterTools()).functions

    outcome = await execute_tool_call(count_up, {"limit": "3"}, _CONTEXT, middleware=())

    assert inspect.isasyncgenfunction(count_up.entrypoint)
    assert (outcome.content, outcome.is_error) == ("123", False)


def test_an_owned_function_that_takes_an_agno_framework_object_is_refused() -> None:
    """Nothing hands an owned function its agent or run context, so a parameter expecting one is refused."""

    class SpoofableTools(Toolkit):
        @tool
        def whoami(self, query: str, run_context: object = None) -> str:
            return f"{query}{run_context}"

    with pytest.raises(ValueError, match=r"'whoami' takes run_context.*current_call\(\)"):
        adapt_toolkit(SpoofableTools())


@pytest.mark.asyncio
async def test_an_owned_functions_media_named_parameter_is_the_models() -> None:
    """Only Agno hands a tool the run's media by name; an owned function's ``files`` is an argument like any other."""

    class AttachTools(Toolkit):
        @tool
        def attach(self, files: list[str]) -> str:
            """Attach files.

            Args:
                files: The file names.

            """
            return ",".join(files)

    [attach] = adapt_toolkit(AttachTools()).functions

    outcome = await execute_tool_call(attach, {"files": ["a.txt", "b.txt"]}, _CONTEXT, middleware=())

    assert attach.injected_params == {}
    assert attach.parameters["required"] == ["files"]
    assert outcome.content == "a.txt,b.txt"


@pytest.mark.asyncio
async def test_current_call_names_the_call_a_function_runs_for() -> None:
    """Asynchronous and synchronous bodies read the call's id, toolkit, session, and runtime context."""
    seen: list[CurrentCall | None] = []

    class RecorderTools(Toolkit):
        @tool
        async def record_async(self) -> str:
            seen.append(current_call())
            return "recorded"

        @tool
        def record_sync(self) -> str:
            seen.append(current_call())
            return "recorded"

    adapted = adapt_toolkit(RecorderTools(), key="recorder")
    state = {"visits": 1}
    scope = HistoryScope(kind="team", scope_id="crew")
    runtime = Mock(spec=ToolRuntimeContext)
    context = replace(_CONTEXT, call_id="call-1", session_state=state, history_scope=scope)
    with tool_runtime_context(runtime):
        outcomes = [await execute_tool_call(function, {}, context, middleware=()) for function in adapted.functions]

    expected = CurrentCall(
        call_id="call-1",
        toolkit="recorder",
        session_id="session-1",
        session_state=state,
        history_scope=scope,
        runtime=runtime,
    )
    assert [outcome.content for outcome in outcomes] == ["recorded", "recorded"]
    assert seen == [expected, expected]
    assert all(call is not None and call.session_state is state for call in seen)
    assert current_call() is None


def test_toolkit_functions_returns_one_callable_per_name() -> None:
    """An owned toolkit gives its functions as authored; an Agno toolkit its async variant where it has one."""

    def lookup() -> str:
        return "sync"

    async def alookup() -> str:
        return "async"

    owned = GreeterTools(tools=[shout])
    agno = AgnoToolkit(name="dual", tools=[lookup])
    agno.async_functions["lookup"] = Function(name="lookup", entrypoint=alookup)
    # A function rebuilt from its stored definition has nothing to call.
    agno.async_functions["described_only"] = Function(name="described_only")

    assert toolkit_functions(owned) == {"greet": owned.greet, "farewell": owned.bye, "shout": shout}
    assert toolkit_functions(agno) == {"lookup": alookup}
    assert toolkit_functions(agno_toolkit(owned)) == {"farewell": owned.bye, "shout": shout, "greet": owned.greet}


def test_a_registered_owned_toolkit_builds_and_adapts_as_its_agno_twin(tmp_path: Path) -> None:
    """``get_tool_by_name`` builds an owned toolkit with output files, and it adapts to its Agno twin's functions."""

    class OwnedTwinTools(Toolkit):
        name = "twin"
        instructions = "Count and shout words."

        @tool
        def count_words(self, text: str, limit: int | None = None) -> str:
            """Count the words of a text.

            Args:
                text: The text.
                limit: The most words to count.

            """
            return str(len(text.split()[:limit]))

        @tool
        def shout_text(self, text: str) -> str:
            """Shout a text.

            Args:
                text: The text.

            """
            return text.upper()

    class AgnoTwinTools(AgnoToolkit):
        def __init__(self) -> None:
            super().__init__(
                name="twin",
                tools=[OwnedTwinTools().count_words, OwnedTwinTools().shout_text],
                instructions="Count and shout words.",
            )

    snapshot = capture_tool_registry_snapshot()
    for tool_name, toolkit_class in (("owned_twin", OwnedTwinTools), ("agno_twin", AgnoTwinTools)):
        register_tool_with_metadata(
            name=tool_name,
            file_access=ToolFileAccess.NONE,
            display_name=tool_name,
            description="Test-only toolkit for owned authoring.",
            category=ToolCategory.DEVELOPMENT,
        )(lambda toolkit_class=toolkit_class: toolkit_class)
    runtime_paths = resolve_runtime_paths(config_path=tmp_path / "config.yaml", storage_path=tmp_path / "storage")
    try:
        owned, agno = (
            adapt_toolkit(
                get_tool_by_name(tool_name, runtime_paths, tool_output_workspace_root=tmp_path, worker_target=None),
            )
            for tool_name in ("owned_twin", "agno_twin")
        )
    finally:
        restore_tool_registry_snapshot(snapshot)

    assert support.golden_entry(owned) == support.golden_entry(agno)
    assert OUTPUT_PATH_ARGUMENT in _by_name(owned.functions)["shout_text"].parameters["properties"]


@pytest.mark.asyncio
async def test_an_owned_function_writes_its_result_where_the_model_asks(tmp_path: Path) -> None:
    """The output-file wrapper an owned function is built with saves its result under the workspace."""

    class ShoutTools(Toolkit):
        name = "shouter"

        @tool
        def shout_text(self, text: str) -> str:
            """Shout a text."""
            return text.upper()

    snapshot = capture_tool_registry_snapshot()
    register_tool_with_metadata(
        name="owned_shouter",
        file_access=ToolFileAccess.NONE,
        display_name="Owned Shouter",
        description="Test-only toolkit for owned output files.",
        category=ToolCategory.DEVELOPMENT,
    )(lambda: ShoutTools)
    runtime_paths = resolve_runtime_paths(config_path=tmp_path / "config.yaml", storage_path=tmp_path / "storage")
    try:
        toolkit = get_tool_by_name(
            "owned_shouter",
            runtime_paths,
            tool_output_workspace_root=tmp_path,
            worker_target=None,
        )
    finally:
        restore_tool_registry_snapshot(snapshot)
    [function] = adapt_toolkit(toolkit).functions

    outcome = await execute_tool_call(
        function,
        {"text": "hi", OUTPUT_PATH_ARGUMENT: "out/shout.txt"},
        _CONTEXT,
        middleware=(),
    )

    assert not outcome.is_error
    assert (tmp_path / "out" / "shout.txt").read_text(encoding="utf-8") == "HI"
