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
Unset means no tracing at all, which is what makes it possible to run the
backend on its own (see the replay mode in the README).
"""
import logging
import os

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

    # No endpoint configured means nobody asked for tracing. Registering
    # anyway points the exporter at phoenix.otel's localhost default and
    # every span then pays for a connection that was never going to work -
    # see the batch=True note below for why that is not merely untidy.
    if not os.environ.get("PHOENIX_COLLECTOR_ENDPOINT"):
        logger.info("PHOENIX_COLLECTOR_ENDPOINT unset - tracing disabled")
        return

    try:
        from phoenix.otel import register

        # batch=True, not phoenix's SimpleSpanProcessor default. Simple
        # exports each span inline on the thread that produced it, so with
        # the collector down every LLM call and every tool call blocks on
        # gRPC connect-and-retry - measured at ~6s per span, which turned a
        # two-search agent turn into a request that never visibly finished.
        # Diagnostics must not sit in the request path; a batch processor
        # exports on its own thread and drops on the floor if nothing is
        # listening.
        register(project_name=service_name, auto_instrument=True, batch=True)
        _instrumented = True
        logger.info(f"phoenix observability enabled (project={service_name})")
    except Exception as e:
        logger.warning(f"phoenix observability setup failed, continuing without it: {e}")
