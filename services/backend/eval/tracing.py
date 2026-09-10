"""Hand-rolled spans for the parts of the judged branch LangChain cannot see.

`observability.py` auto-instruments LangChain, which covers the answerer's
`llm.invoke` and the ragas judge calls. It covers nothing else in a judged
run, and the two things it misses are exactly the two the arm sweep is about:

- **The transform.** `query -> multi_query/hyde/decompose` is read from the
  `queries-*.json` disk cache, not computed, so there is no LLM call to patch.
  Re-generating it just to make a span appear would cost money and break the
  guarantee in `ArmPipeline` that a judged arm and a free arm are the *same*
  arm - so the cached transform is emitted as a span instead, carrying
  `arm.cached=True` so nobody reads it as a live rewrite.
- **Retrieval.** `SearchEngine` is a `PostgresInterface`, not a LangChain
  runnable, so `_vector_candidates`, the BM25 path and `rrf_fuse` are
  invisible. Without them you can see that an arm scored badly but not which
  paraphrase pulled which chunk, which is the only actionable half.

Spans are OpenInference-shaped (`openinference.span.kind`, `retrieval.documents`)
because that is the vocabulary Phoenix renders natively - a CHAIN shows its
input/output, a RETRIEVER gets the document table rather than a JSON blob.

Nothing here is allowed to break a run. Tracing is diagnostic; a $3.60 sweep
must not die because a span attribute was the wrong type or the collector went
away mid-run. Every helper degrades to a no-op context manager.
"""
import json
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Resolved once. Import failure means the OTel packages aren't installed, which
# is a legitimate way to run the backend (see the replay mode in the README),
# not an error to report on every span.
try:
    from openinference.semconv.trace import (
        DocumentAttributes,
        OpenInferenceSpanKindValues,
        SpanAttributes,
    )
    from opentelemetry import trace as _trace

    _tracer = _trace.get_tracer(__name__)
except Exception:  # pragma: no cover - exercised by the disabled-path test
    _tracer = None


def enabled() -> bool:
    """Whether spans will actually be emitted.

    Checks the endpoint rather than a module flag: `setup_observability`
    returns early without instrumenting when it is unset, and a tracer that
    nothing exports is just overhead on every question.
    """
    import os

    return _tracer is not None and bool(os.environ.get("PHOENIX_COLLECTOR_ENDPOINT"))


def _flatten(value):
    """A value OTel will accept as an attribute.

    Scalars pass through with their type intact rather than being stringified:
    Phoenix filters numerically on an int and as a checkbox on a bool, so
    stringifying `top_k` or `arm.cached` would cost the ability to slice the
    sweep by them. `bool` is checked before `int` because it is a subclass of
    one and would otherwise arrive as 1/0.

    Everything else becomes JSON, which Phoenix pretty-prints - so a list of
    paraphrases stays readable instead of rendering as `['a', 'b']`.
    """
    if isinstance(value, (bool, int, float, str)):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


@contextmanager
def _noop():
    yield None


@contextmanager
def span(name: str, kind: str = "CHAIN", **attributes):
    """One span, or nothing at all if tracing is off.

    Yields the span so a caller can attach an output it does not know until
    the body has run; `None` when disabled, which every call site must
    tolerate - hence `set_output`'s own guard rather than `span.set_attribute`
    at the call site.

    Attribute keys must be namespaced (`arm.name`, not `name`), per the
    OpenInference convention every call site here already follows - a bare
    `name` or `kind` would collide with this signature's own parameters.
    """
    if not enabled():
        with _noop() as s:
            yield s
        return

    try:
        kind_value = getattr(OpenInferenceSpanKindValues, kind).value
    except AttributeError:
        kind_value = OpenInferenceSpanKindValues.CHAIN.value

    with _tracer.start_as_current_span(name) as s:
        try:
            s.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, kind_value)
            for key, value in attributes.items():
                if value is not None:
                    s.set_attribute(key, _flatten(value))
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"span attribute failed on {name}: {e}")
        yield s


def set_input(s, value) -> None:
    if s is None:
        return
    try:
        s.set_attribute(SpanAttributes.INPUT_VALUE, _flatten(value))
    except Exception as e:  # pragma: no cover - defensive
        logger.debug(f"set_input failed: {e}")


def set_output(s, value) -> None:
    if s is None:
        return
    try:
        s.set_attribute(SpanAttributes.OUTPUT_VALUE, _flatten(value))
    except Exception as e:  # pragma: no cover - defensive
        logger.debug(f"set_output failed: {e}")


def set_documents(s, chunks: list[dict]) -> None:
    """Attach retrieved chunks as OpenInference documents.

    Phoenix renders these as a ranked table with the text inline, which is
    what makes "which paraphrase found the gold chunk" answerable in the UI
    instead of only in the results JSON.
    """
    if s is None:
        return
    try:
        for i, chunk in enumerate(chunks):
            prefix = f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}"
            s.set_attribute(
                f"{prefix}.{DocumentAttributes.DOCUMENT_ID}", str(chunk.get("chunk_id", ""))
            )
            content = chunk.get("text") or ""
            s.set_attribute(f"{prefix}.{DocumentAttributes.DOCUMENT_CONTENT}", content)
            score = chunk.get("score")
            if score is not None:
                s.set_attribute(f"{prefix}.{DocumentAttributes.DOCUMENT_SCORE}", float(score))
            # doc_id is ours, not part of the OpenInference document schema, but
            # the labels are doc-level as well as chunk-level so it earns a slot.
            doc_id = chunk.get("doc_id")
            if doc_id is not None:
                s.set_attribute(f"{prefix}.metadata.doc_id", str(doc_id))
    except Exception as e:  # pragma: no cover - defensive
        logger.debug(f"set_documents failed: {e}")
