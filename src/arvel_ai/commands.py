"""Console commands contributed to the host app (wired in provider.boot())."""

from __future__ import annotations

import typer

cli = typer.Typer()


@cli.command("ai:hello")
def hello() -> None:
    """Prove the package's CLI wiring works."""
    typer.echo("hello from arvel-ai")
