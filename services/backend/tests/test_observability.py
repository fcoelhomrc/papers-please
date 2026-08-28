"""Unit tests for observability.py's Phoenix setup - idempotent, and never
lets a broken/unreachable collector break the app (this is diagnostic
tooling, not a hard dependency)."""
from unittest.mock import patch

import observability


def test_setup_observability_calls_register_once():
    observability._instrumented = False
    with patch("phoenix.otel.register") as mock_register:
        observability.setup_observability("test-service")
        observability.setup_observability("test-service")  # second call: no-op

        mock_register.assert_called_once_with(project_name="test-service", auto_instrument=True)
    observability._instrumented = False


def test_setup_observability_swallows_errors():
    observability._instrumented = False
    with patch("phoenix.otel.register", side_effect=RuntimeError("collector unreachable")):
        observability.setup_observability("test-service")  # must not raise
    observability._instrumented = False
