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

from orchestrator.tools import (
    chunk_pending,
    download_pending,
    embed_pending,
    fetch_papers,
    get_status,
)

SYSTEM_PROMPT = """You manage a paper-ingestion pipeline: fetch -> download -> chunk -> embed.

Call get_status first to see what's pending at each stage. Then call at most
one stage tool to make progress on whichever stage is most behind. If
nothing is pending anywhere, say so and stop - don't call fetch_papers
speculatively unless asked to fetch something new."""

TOOLS = [fetch_papers, download_pending, chunk_pending, embed_pending, get_status]


def build_agent(llm):
    return create_agent(llm, TOOLS, system_prompt=SYSTEM_PROMPT)
