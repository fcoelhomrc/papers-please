"""Wires up Phoenix (self-hosted, single-container OTel-based LLM
observability - see compose.yaml's phoenix service) so every LangChain/
LangGraph call - agent turns, tool calls, judge calls in eval - shows up
live in a dashboard at http://localhost:6006, instead of only being
visible after the fact in scrollback.

Auto-instruments process-wide via openinference-instrumentation-langchain,
which patches LangChain's runnable machinery directly - LangGraph sits on
top of that, so agent/tool spans show up too, without per-call
CallbackHandler wiring.

PHOENIX_COLLECTOR_ENDPOINT (set in compose.yaml) points at the phoenix
container. No API key - local, self-hosted, nothing leaves this machine.
"""
import logging

logger = logging.getLogger(__name__)

_instrumented = False


def setup_observability(service_name: str) -> None:
    """Call once per process, before any LangChain/LangGraph code runs.
    Idempotent - a second call is a no-op. Never lets a missing or
    unreachable Phoenix collector break the app: this is diagnostic
    tooling, not a hard dependency, and the eval-run credit-exhaustion
    incident is a reminder that a broken guard shouldn't compound into a
    broken app.
    """
    global _instrumented
    if _instrumented:
        return
    try:
        from phoenix.otel import register

        register(project_name=service_name, auto_instrument=True)
        _instrumented = True
        logger.info(f"phoenix observability enabled (project={service_name})")
    except Exception as e:
        logger.warning(f"phoenix observability setup failed, continuing without it: {e}")
