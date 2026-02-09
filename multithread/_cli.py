"""
CLI entry point for Multithread (exposed via pip as the 'multithread' command).
"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="multithread",
        description="Multithread: adaptive thread pool for Python 3.14 free-threading",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the package version and exit",
    )
    args = parser.parse_args()

    if args.version:
        import multithread
        print(f"Multithread {multithread.__version__}")
        sys.exit(0)

    parser.print_help()
    sys.exit(0)


if __name__ == "__main__":
    main()
