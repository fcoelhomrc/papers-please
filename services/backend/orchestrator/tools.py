"""LangChain tools for the orchestrator agent.

Deliberately small. The ingest pipeline (download -> chunk -> embed) is
deterministic: given the DB state there's exactly one correct action, so
it's handled by the stages/*.py workers running on a timer, not by an LLM
deciding whether to call a tool. An agent has no business being a queue
manager for that - it adds latency/cost/a new way to silently skip a step,
for a decision that was never actually being made.

What's left is where an LLM's judgment is real: what to search for (fetch),
what/how to query for retrieval (search_chunks/get_document), and get_status
as context for a fetch decision (e.g. "we already have papers on this",
don't refetch) - not a router between pipeline stages.
"""
from db.connection import PostgresInterface
from db.models import Document
from ingest.fetcher import SemanticScholarFetcher
from langchain_core.tools import tool
from search import get_search_engine
from sqlalchemy.orm import Session
from status import pipeline_status

# Every tool below catches its own exceptions rather than letting them
# propagate out of the graph's tools node - a real incident showed why: an
# uncaught exception (Postgres connection exhaustion) crashed the request
# mid-turn, leaving a tool_use message checkpointed with no matching
# tool_result. Every subsequent call on that thread_id then got rejected by
# Anthropic ("tool_use ids were found without tool_result blocks"),
# permanently corrupting that conversation. Returning an error dict instead
# keeps the message sequence valid either way - the agent sees the failure
# and can report it, rather than the whole turn (and the thread's future)
# blowing up.


@tool
def fetch_papers(query: str, max_papers: int = 100) -> str:
    """Fetch new paper metadata from Semantic Scholar for a search query."""
    try:
        n = SemanticScholarFetcher().fetch(query=query, max_papers=max_papers)
        return f"fetched {n} papers"
    except Exception as e:
        return f"error: fetch failed ({e})"


@tool
def get_status() -> dict:
    """Return counts of documents/objects/chunks at each pipeline stage.

    Context for fetch decisions - e.g. how much we already have on a topic
    before fetching more. Not a router for the ingest pipeline, which runs
    on its own deterministic schedule (stages/*.py). Same query the REST
    /status endpoint uses (status.pipeline_status), for the Queue dashboard.
    """
    try:
        return pipeline_status()
    except Exception as e:
        return {"error": f"status check failed: {e}"}


@tool
def search_chunks(query: str, top_k: int = 5, rerank: bool = True) -> list[dict]:
    """Search the paper library for chunks relevant to a question.

    Returns chunk text with its doc_id, title, and page - cite doc_id/page
    when answering so the user can find the source. Use get_document for
    more of a paper's context (e.g. its abstract) once you've found it here.
    """
    try:
        from config import load

        # Retrieve wide, return narrow: the cross-encoder gets a full pool to
        # pick from rather than the same `top_k` it is being asked to return,
        # which left it reordering the shortlist instead of choosing it.
        response = get_search_engine().search(
            query,
            top_k=top_k,
            rerank=rerank,
            rerank_top_k=top_k,
            candidates=load().search.rerank_candidates,
        )
        return [
            {
                "doc_id": r.doc_id,
                "title": r.title,
                "authors": r.authors,
                "year": r.year,
                "page_num": r.page_num,
                # The widened window when there is one: the model is reading
                # for meaning, and a sentence cut at a chunk boundary reads
                # as a non-answer. The UI still shows r.text, the chunk that
                # actually matched.
                "text": r.context or r.text,
                "score": r.score,
            }
            for r in response.results
        ]
    except Exception as e:
        return [{"error": f"search failed: {e}"}]


@tool
def get_document(doc_id: int) -> dict:
    """Look up a paper's metadata (title, authors, year, abstract) by doc_id."""
    try:
        with Session(PostgresInterface.connect()) as session:
            doc = session.get(Document, doc_id)
        if doc is None:
            return {"error": f"no document with doc_id={doc_id}"}
        return {
            "doc_id": doc.id,
            "title": doc.title,
            "authors": doc.authors,
            "venue": doc.venue,
            "year": doc.year,
            "abstract": doc.abstract,
        }
    except Exception as e:
        return {"error": f"lookup failed: {e}"}
