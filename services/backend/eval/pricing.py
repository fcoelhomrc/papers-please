"""USD per token, for every model this project spends money through.

Shared by the judged eval (`eval.run`) and test-set generation
(`eval.testset`) because both report spend and must agree on the numbers.
Stdlib only, so importing it does not drag ragas - and its import-time
`nest_asyncio.apply()` - into a module that has no need for either.

Rates are OpenRouter list prices. They drift; a wrong rate here is a wrong
figure in a report, not a crash, which is exactly why an unlisted model
reports tokens with no dollar figure instead of a guess.
"""

PRICING = {
    "claude-haiku-4-5": (1.00 / 1e6, 5.00 / 1e6),
    "claude-sonnet-5": (2.00 / 1e6, 10.00 / 1e6),
    "claude-opus-5": (5.00 / 1e6, 25.00 / 1e6),
    # Same models reached through OpenRouter, which passes list price
    # through. Namespaced ids, so they can't collide with the direct ones.
    "anthropic/claude-haiku-4.5": (1.00 / 1e6, 5.00 / 1e6),
    "anthropic/claude-sonnet-5": (2.00 / 1e6, 10.00 / 1e6),
    "anthropic/claude-opus-5": (5.00 / 1e6, 25.00 / 1e6),
    # The judge chosen in docs/judge-model-selection.md.
    "deepseek/deepseek-v4-flash": (0.078 / 1e6, 0.156 / 1e6),
    "qwen/qwen3-235b-a22b-2507": (0.087 / 1e6, 0.350 / 1e6),
    # Writes the test set - see config.LLMConfig.generator_model.
    "z-ai/glm-5.3-flash": (0.075 / 1e6, 0.250 / 1e6),
    # Budget pipeline answerer.
    "qwen/qwen3-30b-a3b-instruct-2507": (0.048 / 1e6, 0.193 / 1e6),
    "minimax/minimax-m2.7": (0.300 / 1e6, 1.200 / 1e6),
}


def model_price(model: str) -> tuple[float, float] | None:
    """USD per (input, output) token, or None if unknown.

    OpenRouter's `:free` tier is genuinely $0 - the tokens are still counted
    and still worth reporting, they just cost nothing - so it reports a real
    zero rather than "unknown". An unlisted paid model reports tokens with
    no dollar figure rather than inventing one.
    """
    if model.endswith(":free"):
        return (0.0, 0.0)
    return PRICING.get(model)
