import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from perm2ask.preparation.pipeline import rebuild


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild and verify the released PerM2ask benchmark."
    )
    parser.add_argument(
        "--raw",
        required=True,
        type=Path,
        help="Directory with the original source files",
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="New directory for the verified output"
    )
    parser.add_argument(
        "--fetch", action="store_true", help="Download missing pinned sources"
    )
    args = parser.parse_args()
    if not __debug__:
        parser.error(
            "Run without Python's -O option so source assertions remain enabled."
        )
    rebuild(args.raw, args.out, args.fetch)
    print(f"Verified the benchmark against the release audit in {args.out}")


if __name__ == "__main__":
    main()
