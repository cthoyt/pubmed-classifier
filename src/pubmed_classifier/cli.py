"""Command line interface for :mod:`pubmed_classifier`."""

import click

__all__ = [
    "main",
]


@click.command()
def main() -> None:
    """CLI for pubmed_classifier."""


if __name__ == "__main__":
    main()
