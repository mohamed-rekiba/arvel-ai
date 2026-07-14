"""Console commands contributed to the host app (wired in provider.boot())."""

from __future__ import annotations

import typer

cli = typer.Typer()


@cli.command("ai:models")
def models() -> None:
    """Show the configured default driver and model aliases."""
    from arvel.kernel.globals import app

    config = app("config")
    typer.echo(f"default driver: {config.get('ai.default')}")
    aliases: dict[str, str] = config.get("ai.models", {}) or {}
    if not aliases:
        typer.echo("model aliases: (none configured — set config ai.models)")
    for alias, concrete in aliases.items():
        typer.echo(f"  {alias} -> {concrete}")
