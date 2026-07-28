"""Print or refresh the deterministic v0.3.2 baseline fingerprint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from manga_translator.baseline import (
    MANIFEST_RELATIVE_PATH,
    build_manifest,
    fingerprint_tree,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--archive-sha256")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.write_manifest:
        if not args.archive_sha256:
            parser.error("--archive-sha256 is required with --write-manifest")
        manifest = build_manifest(root, args.archive_sha256)
        destination = root / MANIFEST_RELATIVE_PATH
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(destination)
        return 0

    print(json.dumps(fingerprint_tree(root).as_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
