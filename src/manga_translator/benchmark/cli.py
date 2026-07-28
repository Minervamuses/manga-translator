"""CLI for preparing and validating benchmark corpora."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ground_truth import prepare_profile, validate_profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m manga_translator.benchmark")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("profile")
    validate = commands.add_parser("validate")
    validate.add_argument("profile")
    validate.add_argument("--require-verified", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "prepare":
        written = prepare_profile(args.root.resolve(), args.profile)
        print(json.dumps({"profile": args.profile, "pages": len(written)}, ensure_ascii=False))
        return 0

    report = validate_profile(
        args.root.resolve(), args.profile, require_verified=args.require_verified
    )
    print(
        json.dumps(
            {
                "profile": args.profile,
                "pages": report.pages,
                "regions": report.regions,
                "unverified": report.unverified,
                "warnings": report.warnings,
                "errors": report.errors,
            },
            ensure_ascii=False,
        )
    )
    return 0 if report.ok else 1
