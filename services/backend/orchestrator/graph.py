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
(tools.py), and the system prompt telling it how to use them - the last of
which lives in prompts/orchestrator/<version>.md, versioned so an eval score
can name the exact prompt that produced it (see prompts/registry.py).
"""
from langchain.agents import create_agent

from orchestrator.tools import fetch_papers, get_document, get_status, search_chunks
from prompts.registry import load_prompt

TOOLS = [fetch_papers, get_status, search_chunks, get_document]

# LangGraph's default recursion_limit is 25 - way more than this 4-tool
# agent legitimately needs (get_status -> one action tool -> maybe
# get_document -> final answer is ~4 real turns, ~8 graph steps). Every
# caller must pass this in invoke()'s config - a runaway loop (e.g. the
# LLM repeatedly calling a tool without converging) should hit a wall well
# before 25 steps' worth of API calls.
MAX_AGENT_RECURSION = 10


def build_agent(
    llm,
    checkpointer=None,
    system_prompt: str | None = None,
    version: str | None = None,
    tools=None,
):
    """Build the agent. `system_prompt` wins if given (tests script it
    directly); otherwise the prompt is loaded at the configured version, or
    at `version` when a caller - eval, comparing candidates - overrides it.

    `tools` overrides TOOLS, which offline replay needs: a replayed model
    still emits real tool calls, and those must land on replay tools rather
    than on the ones that reach Pinecone and Postgres."""
    if system_prompt is None:
        if version is None:
            from config import load

            version = load().prompts.orchestrator
        system_prompt = load_prompt("orchestrator", version)
    return create_agent(
        llm, tools or TOOLS, system_prompt=system_prompt, checkpointer=checkpointer
    )
