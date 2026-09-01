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

from orchestrator.replay import build_replay, replay_enabled


def make_llm(cfg) -> BaseChatModel:
    # Checked before the provider switch, because the env toggle has to be
    # able to override a config.yaml that says "anthropic" - that is the
    # whole point of having an env toggle.
    if replay_enabled(cfg):
        return build_replay()[0]
    if cfg.llm.provider == "anthropic":
        return ChatAnthropic(model=cfg.llm.model, max_tokens=cfg.llm.max_tokens)
    if cfg.llm.provider == "vllm":
        return ChatOpenAI(
            base_url=cfg.llm.vllm_url,
            api_key="EMPTY",
            model=cfg.llm.vllm_model,
            max_tokens=cfg.llm.max_tokens,
        )
    raise ValueError(f"unknown llm.provider: {cfg.llm.provider!r}")


def make_agent_parts(cfg):
    """The (llm, tools) pair for this configuration.

    Replay has to swap both halves: a replayed model scripts tool *calls*,
    and letting those reach the real tools would send the fake straight at
    Pinecone and Postgres. Returning None for tools means "the real ones",
    so every non-replay path is unchanged.
    """
    if replay_enabled(cfg):
        return build_replay()
    return make_llm(cfg), None
