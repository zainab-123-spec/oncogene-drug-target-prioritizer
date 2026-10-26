"""Allow ``python -m odtp``."""

from odtp.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
