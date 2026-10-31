"""Launch the archived RQ2 analysis with repository-relative imports/defaults."""

import sys
from pathlib import Path


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    args = list(sys.argv[1:] if argv is None else argv)
    if not any(arg == "--protocol" or arg.startswith("--protocol=") for arg in args):
        args += ["--protocol", str(root / "rq2" / "protocol.json")]
    from rq2.reliability import main as analysis_main

    previous = sys.argv
    try:
        sys.argv = [str(Path(__file__)), *args]
        return analysis_main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
