"""Allow ``python -m credit.cli`` as an alternative to the ``credit`` entry point."""

from credit.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
