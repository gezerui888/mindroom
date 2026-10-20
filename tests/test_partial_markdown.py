"""Tests for close_partial_markdown(), which shapes a reply still streaming in before it renders."""

from __future__ import annotations

import pytest

from mindroom.matrix.message_builder import close_partial_markdown, markdown_to_html
from tests.cpu_budget_helpers import cpu_budget


@pytest.mark.parametrize(
    ("streamed", "shown"),
    [
        # Prefixes seen while recorded replies streamed.
        ("**Lisbon is on both your", "**Lisbon is on both your**"),
        ("| Rule | Vantr", ""),
        ("[Open the draf", "Open the draf"),
        ("| Candidate | Role | Waiting on", ""),
        # Emphasis, strikethrough, and code spans still open.
        ("Both *agree", "Both *agree*"),
        ("Both _agree", "Both _agree_"),
        ("Fares ~~rose", "Fares ~~rose~~"),
        ("**Bold and *italic", "**Bold and *italic***"),
        ("**Note:", "**Note:**"),
        ("Run `pip ins", "Run `pip ins`"),
        ("Run `ls | grep [a-z]* _x", "Run `ls | grep [a-z]* _x`"),
        ("**Then run `pip ins", "**Then run `pip ins`**"),
        ("Run ``a ` b", "Run ``a ` b``"),
        ("Run ``a `", "Run ``a ` ``"),
        # A run with nothing after it yet, which renders literally until its text arrives.
        ("The answer is **", "The answer is"),
        ("Run `", "Run"),
        ("Fares ~~", "Fares"),
        ("Then **~~", "Then"),
        # A link whose label is still arriving shows its label; one whose URL is still arriving keeps its brackets.
        ("[Open the draft]", "[Open the draft]"),
        ("[Open the draft](", "[Open the draft]"),
        ("[Open the draft](https://docs.example/dra", "[Open the draft]"),
        ("See [the **draft**](https://docs.example/a_b", "See [the **draft**]"),
        ("__Result__[source](https://example.org/paper", "__Result__[source]"),
        ("- [1. Introduction](https://example.org/intro", "- [1. Introduction]"),
        ("[**Open the draf", "**Open the draf**"),
        ("Open [`notes.md`](https://docs.example/no", "Open [`notes.md`]"),
        # Only the block still streaming changes; a finished one renders as written.
        ("**Not closed\n\nNext *one", "**Not closed\n\nNext *one*"),
        ("- **First**\n- Second **is", "- **First**\n- Second **is**"),
        ("> Quoted **text", "> Quoted **text**"),
        ("> **Quoted\n> text", "> **Quoted\n> text**"),
        ("## The **plan", "## The **plan**"),
        ("- **First\n  second", "- **First\n  second**"),
        ("Here's what stands out:\n- **", "Here's what stands out:"),
        # Inline math, autolinks, links, and images are parsed as rendering parses them.
        ("The interval $[0, 1)$ is **half", "The interval $[0, 1)$ is **half**"),
        ("See <https://x.example/_search> and *more", "See <https://x.example/_search> and *more*"),
        ("*See [the *draft](https://docs.example/a) now", "*See [the *draft](https://docs.example/a) now*"),
        ("- **`", ""),
        (
            "See ![chart `v2](https://x.example/i.png) and **more",
            "See ![chart `v2](https://x.example/i.png) and **more**",
        ),
        # A table waits for its separator row, then streams row by row.
        ("Three rules differ:\n\n| Rule | Vantr", "Three rules differ:"),
        ("Three rules differ:\n| Rule | Vantrell | Kessler |\n|---|--", "Three rules differ:"),
        ("| Rule | Vantrell |\n|---|---|\n| Hotels | **$25", "| Rule | Vantrell |\n|---|---|\n| Hotels | **$25**"),
        ("| **Rule | Vantrell |\n|---|---|", "| **Rule | Vantrell |\n|---|---|"),
        # A pipe line a blank line has ended is a paragraph, not a table header.
        ("|x| denotes the absolute value of x.\n\n", "|x| denotes the absolute value of x."),
    ],
)
def test_partial_markdown_is_shown_as_it_will_render(streamed: str, shown: str) -> None:
    """Open syntax at the end of a streamed reply closes, and syntax that cannot render yet waits."""
    assert close_partial_markdown(streamed) == shown


@pytest.mark.parametrize(
    "streamed",
    [
        # Code blocks, open or closed, render as code while they stream.
        "Try this:\n\n```python\nrows = [r * 2 for r in data\n| not a table",
        "```\n| a | b\n[link\n```",
        "    indented = a * b [0",
        # An empty heading after a code block, whose text has not arrived yet.
        "```\na\nb\nc\n```\n\n##",
        "```python\nx = 1\ny = 2\n```\n##",
        # Code spans keep their content.
        "Use `a * b` and `x[0]` with `a | b`",
        # Delimiters that cannot open, and finished links and brackets.
        "5 * 3 = 15 and snake_case_name",
        "2*3 is six",
        "[@Quill](https://matrix.to/#/@mindroom_quill:localhost) has the **draft**",
        "🔧 `web_search` [1]",
        "Done \\*not bold",
        # A trailing backslash would escape a closer.
        "Open **C:\\",
        # Nested emphasis closed by one run, an opener inside a closed pair, spans across quote or list item lines,
        # and a setext heading.
        "**Bold *italic*** more text",
        "**_config.yaml** contains",
        "*a **b*** more text",
        "> **first\n> second**",
        "- **first\n  second**",
        "> Use `a\n> b` and",
        "Title **a\n**a\n===",
        # Runs that close what they follow, by CommonMark's flanking and rule-of-3 matching.
        "**a *b***",
        "**bold***italic*",
        "*foo**bar*",
        "Use the class_",
        # Inline math, autolinks, and image alt text keep their content.
        "The interval $[0, 1)$",
        "See <https://x.example/_search>",
        "See ![a `b](https://x.example/i.png) and more",
        # Tables that already render.
        "| Rule | Vantrell |\n|---|---|\n| Hotels | $250",
        "| Rule | Vantrell |\n|---|---|",
    ],
)
def test_code_and_finished_markdown_are_left_as_written(streamed: str) -> None:
    """Code, finished constructs, and delimiters that cannot open are not rewritten."""
    assert close_partial_markdown(streamed) == streamed


def test_closed_markdown_renders_without_raw_syntax() -> None:
    """The shaped prefixes render as formatting rather than literal Markdown."""
    assert markdown_to_html(close_partial_markdown("**Lisbon is on both your")) == (
        "<p><strong>Lisbon is on both your</strong></p>\n"
    )
    assert markdown_to_html(close_partial_markdown("**Bold and *italic")) == (
        "<p><strong>Bold and <em>italic</em></strong></p>\n"
    )
    assert markdown_to_html(close_partial_markdown("**Then run `pip ins")) == (
        "<p><strong>Then run <code>pip ins</code></strong></p>\n"
    )
    assert markdown_to_html(close_partial_markdown("Fares ~~rose")) == "<p>Fares <s>rose</s></p>\n"
    assert markdown_to_html(close_partial_markdown("Run ``a `")) == "<p>Run <code>a ` </code></p>\n"
    assert "<strong>$25</strong>" in markdown_to_html(
        close_partial_markdown("| Rule | Vantrell |\n|---|---|\n| Hotels | **$25"),
    )


def test_long_unbalanced_reply_closes_within_a_cpu_budget() -> None:
    """Many unmatched delimiters of several kinds do not make the cleanup quadratic."""
    streamed = "*a " * 20_000 + "b_ " * 20_000 + "**c"
    with cpu_budget(seconds=2.0):
        shown = close_partial_markdown(streamed)
    assert shown.startswith(f"{streamed}**")
