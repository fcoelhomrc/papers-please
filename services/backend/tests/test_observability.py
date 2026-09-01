"""Unit tests for observability.py's Phoenix setup - idempotent, off unless
configured, and never lets a broken/unreachable collector break the app
(this is diagnostic tooling, not a hard dependency)."""
from unittest.mock import patch

import pytest

import observability


@pytest.fixture(autouse=True)
def reset():
    observability._instrumented = False
    yield
    observability._instrumented = False


def test_setup_observability_calls_register_once(monkeypatch):
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:4317")
    with patch("phoenix.otel.register") as mock_register:
        observability.setup_observability("test-service")
        observability.setup_observability("test-service")  # second call: no-op

        mock_register.assert_called_once_with(
            project_name="test-service", auto_instrument=True, batch=True
        )


def test_uses_a_batch_processor_not_the_inline_default(monkeypatch):
    """phoenix.otel.register defaults to a SimpleSpanProcessor, which exports
    each span inline on the thread that produced it. With the collector down
    that is ~6s of gRPC connect-and-retry per span, which turned a two-search
    agent turn into a request that never visibly finished. Diagnostics do not
    belong in the request path."""
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:4317")
    with patch("phoenix.otel.register") as mock_register:
        observability.setup_observability("test-service")

    assert mock_register.call_args.kwargs["batch"] is True


def test_no_endpoint_means_no_tracing(monkeypatch):
    """Registering without an endpoint points the exporter at phoenix.otel's
    localhost default, so every span pays for a connection that was never
    going to work. Running the backend on its own has to be free."""
    monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)
    with patch("phoenix.otel.register") as mock_register:
        observability.setup_observability("test-service")

    mock_register.assert_not_called()
    assert observability._instrumented is False


def test_setup_observability_swallows_errors(monkeypatch):
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:4317")
    with patch("phoenix.otel.register", side_effect=RuntimeError("collector unreachable")):
        observability.setup_observability("test-service")  # must not raise
