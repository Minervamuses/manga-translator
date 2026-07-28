"""Developer verification commands."""

from __future__ import annotations

import argparse

from .baseline import run_baseline_verification


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m manga_translator.dev")
    subcommands = parser.add_subparsers(dest="command", required=True)
    verify_baseline = subcommands.add_parser(
        "verify-baseline",
        help="verify the locked v0.3.2 tree, fixtures, compile, and unit suite",
    )
    verify_baseline.add_argument(
        "--mode",
        choices=("snapshot", "regression"),
        default="snapshot",
        help="snapshot checks the historical tree; regression allows committed phase changes",
    )
    args = parser.parse_args(argv)
    if args.command == "verify-baseline":
        return run_baseline_verification(mode=args.mode)
    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
