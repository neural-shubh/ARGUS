
import argparse
from pathlib import Path
import sys

# Make the src-layout package importable when running this script directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argus.pipeline import ArgusPipeline
from argus.storage.database import ArgusDatabase


SUPPORTED = {".pdf", ".html", ".htm", ".txt"}


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into ARGUS Core."
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Path to a document or directory of documents.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Display local database statistics.",
    )

    args = parser.parse_args()

    if args.stats:
        print(ArgusDatabase().stats())

        if not args.path:
            return

    if not args.path:
        parser.error("Provide a document path or use --stats")

    target = Path(args.path)

    if not target.exists():
        parser.error(f"Path does not exist: {target}")

    if target.is_file():
        files = [target]
    else:
        files = sorted(
            path for path in target.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED
        )

    if not files:
        print("No supported documents found.")
        return

    pipeline = ArgusPipeline()

    for file_path in files:
        try:
            result = pipeline.process_file(file_path)
            print(result)
        except Exception as exc:
            print(
                f"ERROR processing {file_path}: "
                f"{type(exc).__name__}: {exc}"
            )

    print("\nFinal database statistics:")
    print(pipeline.database.stats())


if __name__ == "__main__":
    main()
