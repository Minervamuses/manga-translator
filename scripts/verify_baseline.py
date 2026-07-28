"""Script wrapper for the package baseline verifier."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from manga_translator.dev import main  # noqa: I001


if __name__ == "__main__":
    raise SystemExit(main(["verify-baseline", *sys.argv[1:]]))
