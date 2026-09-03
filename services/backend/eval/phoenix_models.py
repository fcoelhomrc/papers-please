"""Teach Phoenix what our models cost, so its dashboards show dollars.

    uv run python -m eval.phoenix_models        # register every priced model
    uv run python -m eval.phoenix_models --list # what Phoenix knows

Phoenix computes span cost from its own model table, which knows nothing
about `z-ai/glm-5.3-flash` or the other OpenRouter ids we use - so without
this every span reports tokens and $0.00.

**Costs are applied at ingest and are never backfilled.** Registering a model
prices the spans that arrive *after* it, not the ones already stored. Run
this before a run you want costed, not after; a run whose spans landed first
will read $0 in Phoenix forever, and its real figure comes from the harness's
own `Spend:` block.

Rates come from eval/pricing.py, so Phoenix and the harness cannot disagree.
Phoenix's own store lives in the phoenix_data volume - this exists so a
`podman compose down -v` does not silently take the pricing with it.
"""
import argparse
import json
import os
import urllib.error
import urllib.request

from eval.pricing import PRICING

PHOENIX_URL = os.environ.get("PHOENIX_URL", "http://localhost:6006")

CREATE = (
    "mutation($i: CreateModelMutationInput!){ createModel(input: $i){ "
    "model { id name namePattern } } }"
)
LIST = "{ generativeModels { name namePattern } }"

def _post(query: str, variables: dict | None = None) -> dict:
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(
        f"{PHOENIX_URL}/graphql", data=body, headers={"Content-Type": "application/json"}
    )
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def known() -> set[str]:
    """Name patterns Phoenix already has, so re-running is a no-op."""
    models = (_post(LIST).get("data") or {}).get("generativeModels") or []
    return {m["namePattern"] for m in models}


def pattern_for(model: str) -> str:
    """An anchored, escaped regex, so `glm-5.3-flash` cannot also match
    `glm-513-flash` - the dot is a wildcard otherwise."""
    return "^" + model.replace(".", r"\.") + "$"


def register(model: str, cost_in: float, cost_out: float) -> dict:
    return _post(
        CREATE,
        {
            "i": {
                "name": model,
                "provider": "openrouter",
                "namePattern": pattern_for(model),
                # PRICING is per token; Phoenix wants per million.
                "costs": [
                    {
                        "tokenType": "input",
                        "kind": "PROMPT",
                        "costPerMillionTokens": cost_in * 1e6,
                    },
                    {
                        "tokenType": "output",
                        "kind": "COMPLETION",
                        "costPerMillionTokens": cost_out * 1e6,
                    },
                ],
            }
        },
    )


def sync() -> tuple[list[str], list[str]]:
    """Register every namespaced model not already present."""
    existing = known()
    added, skipped = [], []
    for model, (cin, cout) in sorted(PRICING.items()):
        if "/" not in model:
            continue
        if pattern_for(model) in existing:
            skipped.append(model)
            continue
        register(model, cin, cout)
        added.append(model)
    return added, skipped


def main():
    parser = argparse.ArgumentParser(description="Register model prices in Phoenix")
    parser.add_argument("--list", action="store_true", help="show what Phoenix knows")
    args = parser.parse_args()

    try:
        if args.list:
            for pattern in sorted(known()):
                print(f"  {pattern}")
            return

        added, skipped = sync()
        for model in added:
            print(f"  + {model}")
        for model in skipped:
            print(f"  = {model} (already registered)")
        print(f"\n{len(added)} added, {len(skipped)} already present")
        if added:
            print(
                "Costs apply to spans ingested from now on - Phoenix does not "
                "backfill, so a run already in progress keeps its $0 spans."
            )
    except urllib.error.URLError as e:
        # Phoenix is diagnostic tooling; not having it running is normal.
        raise SystemExit(f"Phoenix unreachable at {PHOENIX_URL}: {e}")


if __name__ == "__main__":
    main()
