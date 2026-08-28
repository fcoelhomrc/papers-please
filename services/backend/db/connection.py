import os
import time

from sqlalchemy import create_engine

_engine = None


class PostgresInterface:
    def __init__(self):
        self.engine = self.connect()

    @staticmethod
    def connect():
        """Lazy singleton engine, not a fresh one per call. Each engine owns
        its own connection pool (default: 5 + 10 overflow) that isn't
        released promptly just because the Engine object goes out of scope -
        status.py/orchestrator tools called this per-request, and the Queue
        dashboard's 10s auto-refresh was enough to exhaust Postgres's
        max_connections within a normal test session."""
        global _engine
        if _engine is not None:
            return _engine

        user = os.environ["POSTGRES_USER"]
        password = os.environ["POSTGRES_PASSWORD"]
        db = os.environ["POSTGRES_DB"]
        from config import load
        host = load().database.host

        _engine = create_engine(
            f"postgresql+psycopg2://{user}:{password}@{host}/{db}",
        )
        return _engine

    @staticmethod
    def rate_limit(wait):
        time.sleep(wait)
