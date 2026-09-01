"""Offline replay: run the agent with no API key, no Postgres and no Pinecone.

Every remaining piece of UI work is about rendering what the agent produced -
citations, tool traces, abstention states - and none of it should require an
Anthropic key or spend a token to develop against. Before this there was no
way to see the agent panel do anything at all without all three services
live.

What is faked and what is not matters. The LangGraph graph runs for real:
the tool-calling loop, the checkpointer, the message sequence, and whatever
reads that sequence afterwards (evidence extraction, the trace) are all
exercised exactly as in production. Only the two ends are canned - the model
that decides, and the tools that fetch. So a bug in how a tool result turns
into a citation is still catchable here; only a bug in retrieval quality is
not.

Fixtures are hand-authored JSON in eval/replay/, deliberately not recorded
from a live run - recording would mean spending the tokens this exists to
avoid, and a hand-written fixture can cover an abstention or a tool failure
that is awkward to provoke on demand.

    llm:
      provider: replay        # config.yaml
    PAPERS_PLEASE_REPLAY=1    # or the environment, which wins

A fixture matches on a substring of the conversation's first human message.
Anything unmatched falls through to eval/replay/_fallback.json rather than
erroring, because a UI that dead-ends on an unanticipated question is not
much of a demo.
"""
import json
import os
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from pydantic import Field

FIXTURES_DIR = Path(__file__).parent.parent / "eval" / "replay"
FALLBACK_NAME = "_fallback"

# Truthy values for the env toggle. "0"/"false"/"" mean off so that an
# explicitly-disabled variable behaves the way anyone would expect, rather
# than switching replay on because the string was non-empty.
_OFF = ("", "0", "false", "no", "off")


def replay_enabled(cfg) -> bool:
    """Environment beats config: the compose file and the test suite both
    need to force replay on without editing a checked-in config.yaml."""
    env = os.environ.get("PAPERS_PLEASE_REPLAY", "")
    if env.strip().lower() not in _OFF:
        return True
    return cfg.llm.provider == "replay"


def load_fixtures(directory: Path | None = None) -> list[dict]:
    directory = directory or FIXTURES_DIR
    if not directory.is_dir():
        return []
    return [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]


def select_fixture(fixtures: list[dict], question: str) -> dict | None:
    """First fixture whose `match` appears in the question, else the fallback.

    Longest match first, so a specific fixture beats a general one no matter
    what order the directory happened to list them in - "fall recovery in
    legged robots" should not be answered by a fixture matching "robot".
    """
    text = (question or "").lower()
    candidates = [f for f in fixtures if f.get("name") != FALLBACK_NAME]
    for fixture in sorted(candidates, key=lambda f: -len(f.get("match", ""))):
        if fixture.get("match", "").lower() in text:
            return fixture
    return next((f for f in fixtures if f.get("name") == FALLBACK_NAME), None)


def _first_question(messages) -> str:
    for m in messages:
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


class ReplaySession:
    """The state the fake model and the fake tools have to agree on.

    Both ends need to know which fixture is in play, and only the model ever
    sees the question - the tools are handed arguments, not a conversation.
    A shared object is how the tools find out; the alternative (re-deriving
    the fixture inside each tool from a question it was never given) is not
    available.

    Per-tool call counters rather than one global sequence: a fixture that
    calls search_chunks twice and get_document once reads as two lists, not
    as an interleaving that has to be kept in step with the turns above it.
    """

    def __init__(self, fixtures: list[dict]):
        self.fixtures = fixtures
        self.question = ""
        self._calls: dict[str, int] = {}

    def observe(self, question: str) -> dict | None:
        """Called by the model on every turn. A changed question means a new
        conversation, which resets the tool counters - without this a second
        question in the same process would start reading its fixture's tool
        results from wherever the first one left off."""
        if question != self.question:
            self.question = question
            self._calls = {}
        return self.fixture()

    def fixture(self) -> dict | None:
        return select_fixture(self.fixtures, self.question)

    def next_result(self, tool_name: str):
        results = ((self.fixture() or {}).get("tool_results") or {}).get(tool_name)
        n = self._calls.get(tool_name, 0)
        self._calls[tool_name] = n + 1
        if not results:
            return {"error": f"no replay result for {tool_name!r}"}
        # A tool called more often than its list is long repeats the last
        # entry: an unanticipated extra call should degrade to a stale answer,
        # not an IndexError that corrupts the thread.
        return results[min(n, len(results) - 1)]


class ReplayChatModel(BaseChatModel):
    """Replays a fixture's scripted assistant turns.

    Turn number comes from how many AIMessages are already in the history -
    the same trick tests/fakes.py uses. The graph re-sends the whole
    conversation at each step, so its length is the only turn counter
    available without holding state the graph might replay or roll back.

    bind_tools() is a no-op: the fixture already decided which tools get
    called, and pretending to reason over schemas would only add a way for
    the fake to disagree with itself.
    """

    session: ReplaySession

    model_config = {"arbitrary_types_allowed": True}

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        fixture = self.session.observe(_first_question(messages)) or {}
        turns = fixture.get("turns") or []
        index = sum(1 for m in messages if isinstance(m, AIMessage))

        if index < len(turns):
            turn = turns[index]
            message = AIMessage(
                content=turn.get("content", ""),
                tool_calls=[
                    {"name": c["name"], "args": c.get("args", {}), "id": f"replay_{index}_{i}"}
                    for i, c in enumerate(turn.get("tool_calls") or [])
                ],
            )
        else:
            # Ran past the script - the conversation continued beyond what the
            # fixture anticipated. Ending the turn cleanly beats raising: an
            # exception here would checkpoint a tool_use with no tool_result
            # and corrupt the thread for every later message.
            message = AIMessage(
                content="(replay fixture exhausted - no further scripted turns)"
            )

        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "replay"


def replay_tools(session: ReplaySession) -> list[StructuredTool]:
    """The four real tool signatures, answered from disk.

    Signatures are restated rather than wrapping orchestrator/tools.py:
    wrapping would still import the real bodies, which pull in Pinecone and
    Postgres at construction time - the thing this module exists to avoid.
    """

    def fetch_papers(query: str, max_papers: int = 100) -> str:
        """Fetch new paper metadata from Semantic Scholar for a search query."""
        return session.next_result("fetch_papers")

    def get_status() -> dict:
        """Return counts of documents/objects/chunks at each pipeline stage."""
        return session.next_result("get_status")

    def search_chunks(query: str, top_k: int = 5, rerank: bool = True) -> list[dict]:
        """Search the paper library for chunks relevant to a question."""
        return session.next_result("search_chunks")

    def get_document(doc_id: int) -> dict:
        """Look up a paper's metadata (title, authors, year, abstract) by doc_id."""
        return session.next_result("get_document")

    return [
        StructuredTool.from_function(f)
        for f in (fetch_papers, get_status, search_chunks, get_document)
    ]


def build_replay(fixtures: list[dict] | None = None) -> tuple[ReplayChatModel, list]:
    """A model and a tool set wired to the same fixtures."""
    session = ReplaySession(load_fixtures() if fixtures is None else fixtures)
    return ReplayChatModel(session=session), replay_tools(session)
