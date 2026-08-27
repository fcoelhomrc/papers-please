"""Shared fixtures for integration tests.

Integration tests hit real infra:
  - Postgres: a disposable `postgres:16` container started via `podman run`,
    schema applied from services/db/schema.sql, torn down after the session.
  - Pinecone: the real `papers-please-bge-small` index, isolated to a `test`
    namespace that's wiped before and after use. Never touches the default
    namespace prod data lives in.

Run with: uv run pytest -m integration
Needs: podman on PATH, network access, PINECONE_API_KEY in the environment.
"""
import os
import subprocess
import time
import uuid

import pytest
from sqlalchemy import create_engine, text

SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "db", "schema.sql"
)
PINECONE_TEST_NAMESPACE = "test"


@pytest.fixture(scope="session")
def pg_container():
    """A real, disposable Postgres container with the project schema applied."""
    name = f"papers-please-test-pg-{uuid.uuid4().hex[:8]}"
    port = 55432
    user, password, db = "test", "test", "test"

    subprocess.run(
        [
            "podman", "run", "-d", "--rm",
            "--name", name,
            "-e", f"POSTGRES_USER={user}",
            "-e", f"POSTGRES_PASSWORD={password}",
            "-e", f"POSTGRES_DB={db}",
            "-p", f"{port}:5432",
            "docker.io/library/postgres:16",
        ],
        check=True,
    )

    url = f"postgresql+psycopg2://{user}:{password}@localhost:{port}/{db}"
    engine = create_engine(url)

    try:
        _wait_for_postgres(engine, timeout_s=30)
        with open(SCHEMA_PATH) as f:
            schema_sql = f.read()
        with engine.begin() as conn:
            conn.execute(text(schema_sql))
        yield engine
    finally:
        engine.dispose()
        subprocess.run(["podman", "stop", name], check=False)


def _wait_for_postgres(engine, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    last_err = None
    while time.monotonic() < deadline:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    raise RuntimeError(f"Postgres container never became ready: {last_err}")


@pytest.fixture
def pg_session(pg_container):
    """Truncates all tables before each test so tests don't leak state into each other."""
    with pg_container.begin() as conn:
        conn.execute(text(
            "TRUNCATE documents, objects, chunks, embedding_models, "
            "chunk_embeddings RESTART IDENTITY CASCADE"
        ))
    yield pg_container


@pytest.fixture
def configured_db(monkeypatch, pg_session):
    """Points config.load()/PostgresInterface at the pg_container, like the real
    entrypoints do, so code under test (e.g. PdfEmbedder()) can be instantiated
    normally instead of being handed an engine directly."""
    import config as config_module

    url = pg_session.url
    monkeypatch.setenv("POSTGRES_USER", url.username)
    monkeypatch.setenv("POSTGRES_PASSWORD", url.password)
    monkeypatch.setenv("POSTGRES_DB", url.database)
    monkeypatch.setattr(
        config_module,
        "_config",
        config_module.Config(
            database=config_module.DatabaseConfig(host=f"localhost:{url.port}")
        ),
    )
    yield pg_session
    monkeypatch.setattr(config_module, "_config", None)


@pytest.fixture(scope="session")
def pinecone_index():
    """The real papers-please-bge-small index, restricted to a `test` namespace."""
    from pinecone.grpc import PineconeGRPC as Pinecone

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index("papers-please-bge-small")

    def _clear():
        try:
            index.delete(delete_all=True, namespace=PINECONE_TEST_NAMESPACE)
        except Exception:
            pass  # namespace doesn't exist yet - nothing to clear

    _clear()
    yield index
    _clear()
