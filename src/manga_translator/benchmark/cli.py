"""CLI for preparing and validating benchmark corpora."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ground_truth import prepare_profile, validate_profile
from .performance import run_performance_baseline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m manga_translator.benchmark")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("profile")
    validate = commands.add_parser("validate")
    validate.add_argument("profile")
    validate.add_argument("--require-verified", action="store_true")
    performance = commands.add_parser("performance")
    performance.add_argument("--profile", default="v032_baseline")
    args = parser.parse_args(argv)

    if args.command == "prepare":
        written = prepare_profile(args.root.resolve(), args.profile)
        print(json.dumps({"profile": args.profile, "pages": len(written)}, ensure_ascii=False))
        return 0

    if args.command == "performance":
        run_path, report = run_performance_baseline(args.root.resolve(), args.profile)
        print(
            json.dumps(
                {
                    "profile": args.profile,
                    "run_id": report["run_id"],
                    "real_status": report["real_run"]["status"],
                    "run_path": run_path.relative_to(args.root.resolve()).as_posix(),
                },
                ensure_ascii=False,
            )
        )
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
