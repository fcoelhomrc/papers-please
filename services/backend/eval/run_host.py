"""Run an eval module from the host rather than inside compose.

    uv run python -m eval.run_host testset generate

config.yaml names Postgres by its compose service name (`db`), which does
not resolve outside the network. Rather than keep a second config file in
sync, this patches the one field that differs and hands off to the module's
own CLI - so a host run and a container run cannot drift apart in any other
setting.
"""
import importlib
import os
import sys

HOST_DB = os.environ.get("PAPERS_PLEASE_HOST_DB", "localhost:5433")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m eval.run_host <module> [args...]")

    from config import load

    load().database.host = HOST_DB

    module = sys.argv[1]
    sys.argv = [f"eval.{module}", *sys.argv[2:]]
    importlib.import_module(f"eval.{module}").main()


if __name__ == "__main__":
    main()
