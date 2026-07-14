"""Console commands contributed to the host app (wired in provider.boot())."""

from __future__ import annotations

import typer

cli = typer.Typer()


@cli.command("ai:models")
def models() -> None:
    """Show the configured default driver and model aliases."""
    from arvel import config

    typer.echo(f"default driver: {config('ai.default')}")
    aliases: dict[str, str] = config("ai.models", {}) or {}
    if not aliases:
        typer.echo("model aliases: (none configured — set config ai.models)")
    for alias, concrete in aliases.items():
        typer.echo(f"  {alias} -> {concrete}")
