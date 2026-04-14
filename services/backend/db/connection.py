import os
import time

from sqlalchemy import create_engine


class PostgresInterface:
    def __init__(self):
        self.engine = self.connect()

    @staticmethod
    def connect():
        user = os.environ["POSTGRES_USER"]
        password = os.environ["POSTGRES_PASSWORD"]
        db = os.environ["POSTGRES_DB"]
        from config import load
        host = load().database.host

        engine = create_engine(
            f"postgresql+psycopg2://{user}:{password}@{host}/{db}",
        )
        return engine

    @staticmethod
    def rate_limit(wait):
        time.sleep(wait)
