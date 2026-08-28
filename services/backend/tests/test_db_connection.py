"""Unit test for PostgresInterface.connect()'s singleton behavior.

Regression test for a real bug: connect() used to call create_engine() on
every invocation - each engine owns its own connection pool, so repeated
calls (status.py/orchestrator tools call it per-request) leaked connections
until Postgres's max_connections was exhausted.
"""
from unittest.mock import MagicMock, patch

import db.connection as connection_module


def test_connect_reuses_the_same_engine_across_calls():
    connection_module._engine = None  # isolate from whatever other tests left behind
    fake_engine = MagicMock()

    with (
        patch.dict("os.environ", {"POSTGRES_USER": "u", "POSTGRES_PASSWORD": "p", "POSTGRES_DB": "d"}),
        patch("db.connection.create_engine", return_value=fake_engine) as mock_create,
        patch("config.load", return_value=MagicMock(database=MagicMock(host="db"))),
    ):
        first = connection_module.PostgresInterface.connect()
        second = connection_module.PostgresInterface.connect()
        third = connection_module.PostgresInterface.connect()

    assert first is second is third is fake_engine
    mock_create.assert_called_once()

    connection_module._engine = None  # don't leak into other tests
