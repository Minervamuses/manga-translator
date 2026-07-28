"""Developer verification commands."""

from __future__ import annotations

import argparse

from .baseline import run_baseline_verification


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m manga_translator.dev")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser(
        "verify-baseline",
        help="verify the locked v0.3.2 tree, fixtures, compile, and unit suite",
    )
    args = parser.parse_args(argv)
    if args.command == "verify-baseline":
        return run_baseline_verification()
    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
