"""Merge duplicate and delete junk entities in a shared agent's knowledge graph through the running MindRoom."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from urllib.parse import quote

import typer

from mindroom.cli.api import get_api_response, post_api_response
from mindroom.cli.config import activate_cli_runtime
from mindroom.cli.help_format import examples
from mindroom.http_error_detail import error_detail_from_response
from mindroom.knowledge_graph.maintenance import DeleteReport, EntityCount, MergeReport

_DRY_RUN = "Dry run: nothing changed. Re-run with --apply."

knowledge_graph_app = typer.Typer(
    help="Merge duplicates and delete junk in a knowledge graph.",
    epilog=examples(
        ("mindroom knowledge-graph entities mind ann", "Find names with ann"),
        ('mindroom knowledge-graph merge mind "Ann Lee" Ann', "Preview a merge"),
        ("mindroom knowledge-graph delete mind --junk", "Preview deleting junk"),
    ),
)

_AGENT = typer.Argument(..., help="A shared agent with the knowledge graph enabled.")
_CONFIG = typer.Option(None, "--config", "-c", help="Select the runtime environment file.")
_URL = typer.Option(None, "--url", help="MindRoom base URL; defaults to MINDROOM_URL or localhost:8765.")
# A change waits for the episode being added to finish, which can take minutes.
_TIMEOUT = typer.Option(600.0, "--timeout", min=0.001, help="HTTP timeout in seconds.")
_APPLY = typer.Option(False, "--apply", help="Make the change; without it, only report what would change.")


def _run[T](
    config_path: Path | None,
    url: str | None,
    path: str,
    timeout: float,
    reply: type[T],
    *,
    params: dict[str, str | int] | None = None,
    json: dict[str, object] | None = None,
) -> T:
    """Post ``json`` if given, else get with ``params``; return the reply as a ``reply``, or print why and exit 2."""
    from pydantic import TypeAdapter  # noqa: PLC0415  # lazy: keeps pydantic out of every other command's start

    try:
        runtime_paths = activate_cli_runtime(config_path)
        if json is None:
            response = get_api_response(runtime_paths, url, path, timeout, require_key=True, params=params)
        else:
            response = post_api_response(runtime_paths, url, path, timeout, require_key=True, json=json)
    except TimeoutError:
        typer.echo("MindRoom did not answer in time; a change may still finish, so run the dry run again.", err=True)
        raise typer.Exit(2) from None
    except (ValueError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from None
    if response.is_error:
        typer.echo(f"Request failed (HTTP {response.status_code}): {error_detail_from_response(response)}", err=True)
        raise typer.Exit(2)
    try:
        return TypeAdapter(reply).validate_json(response.content)
    except ValueError:
        typer.echo("MindRoom returned an invalid response; check --url and the running version.", err=True)
        raise typer.Exit(2) from None


def _route(agent: str, operation: str) -> str:
    return f"/api/knowledge-graph/{quote(agent, safe='')}/{operation}"


def _counts(entity: EntityCount) -> str:
    return f"{entity.name} ({entity.facts} facts, {entity.mentions} mentions)"


def _finish(unmatched: list[str], applied: bool, done: str) -> None:
    if unmatched:
        typer.echo(f"No entity is named: {', '.join(unmatched)}")
    typer.echo(done if applied else _DRY_RUN)


@knowledge_graph_app.command("entities")
def entities(
    agent: str = _AGENT,
    query: str = typer.Argument("", help="Part of the name, ignoring case; empty lists all."),
    limit: int = typer.Option(50, "--limit", min=1, max=1000, help="Most entities to list."),
    config_path: Path | None = _CONFIG,
    url: str | None = _URL,
    timeout: float = _TIMEOUT,
) -> None:
    """List entities whose name contains QUERY, most facts first."""
    found = _run(
        config_path,
        url,
        _route(agent, "entities"),
        timeout,
        list[EntityCount],
        params={"query": query, "limit": limit},
    )
    if not found:
        typer.echo("No entity's name contains that text.")
    for entity in found:
        typer.echo(f"{entity.facts:>7} facts {entity.mentions:>7} mentions  {entity.name}")


@knowledge_graph_app.command("merge")
def merge(
    agent: str = _AGENT,
    keep: str = typer.Argument(..., help="The entity to keep, by its exact name, ignoring case."),
    duplicates: list[str] = typer.Argument(..., help="Entities to merge into it, by their exact names."),  # noqa: B008
    apply: bool = _APPLY,
    config_path: Path | None = _CONFIG,
    url: str | None = _URL,
    timeout: float = _TIMEOUT,
) -> None:
    """Merge duplicate entities into one, moving their facts and mentions.

    Another entity with exactly the name to keep is merged too. Facts between the merged entities are dropped.
    """
    report = _run(
        config_path,
        url,
        _route(agent, "merge"),
        timeout,
        MergeReport,
        json={"keep": keep, "duplicates": duplicates, "apply": apply},
    )
    typer.echo(f"Keep: {_counts(report.keep)}")
    for entity in report.merged:
        typer.echo(
            f"Merge {entity.name}: {entity.facts_moved} facts move, {entity.self_loops_dropped} self-loops dropped, "
            f"{entity.mentions_moved} mentions move, {entity.mentions_dropped} dropped",
        )
    typer.echo(f"In all: {report.facts_moved} facts move, {report.self_loops_dropped} self-loops dropped.")
    _finish(report.unmatched, report.applied, f"Merged {len(report.merged)} entities into {report.keep.name}.")


@knowledge_graph_app.command("delete")
def delete(
    agent: str = _AGENT,
    names: list[str] | None = typer.Argument(None, help="Entities to delete, by their exact names."),  # noqa: B008
    junk: bool = typer.Option(
        False,
        "--junk",
        help="Also delete paths, files, URLs, env vars, hashes, and room links.",
    ),
    apply: bool = _APPLY,
    config_path: Path | None = _CONFIG,
    url: str | None = _URL,
    timeout: float = _TIMEOUT,
) -> None:
    """Delete named or junk-looking entities with their facts.

    Read the dry run's list before --apply: the patterns are guesses, so delete by name when it holds a real entity.
    """
    if not names and not junk:
        typer.echo("Name the entities to delete, or pass --junk.", err=True)
        raise typer.Exit(2)
    report = _run(
        config_path,
        url,
        _route(agent, "delete"),
        timeout,
        DeleteReport,
        json={"names": names or [], "junk": junk, "apply": apply},
    )
    for entity in report.deleted:
        typer.echo(f"Delete {_counts(entity)}")
    _finish(report.unmatched, report.applied, f"Deleted {len(report.deleted)} entities.")
