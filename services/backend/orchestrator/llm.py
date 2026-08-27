"""Provider-agnostic LLM factory for the orchestrator agent.

Both ChatAnthropic and ChatOpenAI implement langchain_core's BaseChatModel
interface, so LangGraph's create_react_agent doesn't care which one it gets -
it just calls .invoke()/.bind_tools() on whatever make_llm() returns. That's
the whole trick behind "swap Claude for a self-hosted vLLM model via one
config flag": vLLM serves an OpenAI-compatible endpoint, so pointing
ChatOpenAI's base_url at it is a drop-in replacement.
"""
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI


def make_llm(cfg) -> BaseChatModel:
    if cfg.llm.provider == "anthropic":
        return ChatAnthropic(model=cfg.llm.model)
    if cfg.llm.provider == "vllm":
        return ChatOpenAI(
            base_url=cfg.llm.vllm_url,
            api_key="EMPTY",
            model=cfg.llm.vllm_model,
        )
    raise ValueError(f"unknown llm.provider: {cfg.llm.provider!r}")
