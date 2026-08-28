"""The orchestrator agent: a langchain create_agent over the pipeline tools.

(mvp-plan.md sketches this via langgraph.prebuilt.create_react_agent, but
that's deprecated as of LangGraph v1.0 in favor of langchain.agents.create_agent
- same shape, moved package. Using the current, non-deprecated entrypoint.)

create_agent builds a small graph for us:

    agent node (call the LLM, maybe with tool calls)
        <-> tools node (run whichever tool the LLM asked for)

    (loops between the two until the LLM responds with no tool calls,
    then the graph ends)

That's the entire "agent loop" - we don't hand-write a while loop or a
StateGraph ourselves. What we own is: which LLM (llm.py), which tools
(tools.py), and the system prompt telling it how to use them.
"""
from langchain.agents import create_agent

from orchestrator.tools import fetch_papers, get_document, get_status, search_chunks

SYSTEM_PROMPT = """You're the assistant for a research paper library. You do two
different kinds of work - tell them apart before acting:

1. FETCHING new papers into the library ("get me papers on X", "find recent
   papers on Y"). Call get_status if it helps judge whether we already have
   relevant papers before fetching more. Call fetch_papers with a search
   query capturing what's being asked for. Downloading, OCR/chunking, and
   embedding happen automatically once papers are fetched - not your job,
   don't try to trigger them.

2. ANSWERING questions using papers already in the library ("has anyone
   studied X", "what did paper Y conclude"). Call search_chunks to find
   relevant passages; call get_document if you need more of a paper's
   context (e.g. its abstract). Answer from what you find, and cite the
   doc_id and page for every claim so the user can dig into the source
   themselves. If nothing relevant turns up, say so - don't guess, and
   don't fetch new papers just because search came up empty."""

TOOLS = [fetch_papers, get_status, search_chunks, get_document]


def build_agent(llm):
    return create_agent(llm, TOOLS, system_prompt=SYSTEM_PROMPT)
