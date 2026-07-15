"""Console commands contributed to the host app (wired in provider.boot())."""

from __future__ import annotations

import typer

cli = typer.Typer()


@cli.command("ai:models")
def models() -> None:
    """Show the configured default driver and model aliases."""
    from .settings import AiSettings

    settings = AiSettings()
    typer.echo(f"default driver: {settings.default}")
    if not settings.models:
        typer.echo("model aliases: (none configured — set config ai.models)")
    for alias, concrete in settings.models.items():
        typer.echo(f"  {alias} -> {concrete}")
