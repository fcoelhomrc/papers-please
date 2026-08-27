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

from orchestrator.tools import fetch_papers, get_status

SYSTEM_PROMPT = """You decide what papers to fetch into a research library, given a request.

Call get_status if it helps you judge whether we already have relevant
papers before fetching more. Call fetch_papers with a search query that
captures what's being asked for. Downloading, OCR/chunking, and embedding
happen automatically on their own schedule once papers are fetched - that's
not your job, don't try to trigger them."""

TOOLS = [fetch_papers, get_status]


def build_agent(llm):
    return create_agent(llm, TOOLS, system_prompt=SYSTEM_PROMPT)
