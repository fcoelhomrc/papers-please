"""Provider-agnostic LLM factory for the orchestrator agent.

Both ChatAnthropic and ChatOpenAI implement langchain_core's BaseChatModel
interface, so LangGraph's create_agent doesn't care which one it gets - it
just calls .invoke()/.bind_tools() on whatever make_llm() returns. That's
the whole trick behind "swap the model via one config flag": anything
serving an OpenAI-compatible endpoint is a drop-in replacement for
ChatOpenAI with a different base_url.

OpenRouter is exactly that - one OpenAI-compatible endpoint fronting every
provider - so it needs no new client library and no new code path beyond a
base_url and a key. It is now the default: "which model" becomes a config
string rather than a code change, which is what makes an ablation across
models possible at all.

The anthropic branch is kept and still works. Set `llm.provider: anthropic`
to use it. Note that reaching Claude *through* OpenRouter
(`anthropic/claude-haiku-4.5`) also works and costs the same - the direct
branch exists for when you want to bypass the middleman entirely.
"""
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from orchestrator.replay import build_replay, replay_enabled

# Sent on every OpenRouter request. Optional, and purely for attribution -
# OpenRouter uses them to label traffic on its own dashboards and public
# rankings. Harmless to send, and it makes usage identifiable when reading
# the account's activity page later.
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/fcoelhomrc/papers-please",
    "X-Title": "Papers Please",
}


def openrouter_chat(model: str, max_tokens: int, cfg) -> ChatOpenAI:
    """A ChatOpenAI pointed at OpenRouter.

    The key is read here rather than at import time so the rest of the app
    (tests, replay mode, the search-only endpoints) still works without one.
    """
    return ChatOpenAI(
        base_url=cfg.llm.openrouter_url,
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=model,
        max_tokens=max_tokens,
        default_headers=OPENROUTER_HEADERS,
    )


def make_llm(cfg) -> BaseChatModel:
    # Checked before the provider switch, because the env toggle has to be
    # able to override a config.yaml that says "anthropic" - that is the
    # whole point of having an env toggle.
    if replay_enabled(cfg):
        return build_replay()[0]
    if cfg.llm.provider == "openrouter":
        return openrouter_chat(cfg.llm.model, cfg.llm.max_tokens, cfg)
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
