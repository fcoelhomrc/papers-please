"""Does the judge agree with a human? Cohen's kappa on 83 labelled statements.

    uv run python -m eval.judge_kappa              # cached if already run
    uv run python -m eval.judge_kappa --force

Every judged number this project reports rests on the judge being right, and
nothing so far has checked that. This does, against labels a person actually
produced rather than against another model's opinion.

What is measured, and why it is the NLI step rather than faithfulness
---------------------------------------------------------------------
Ragas' faithfulness is two steps: decompose an answer into statements, then
decide for each whether the context supports it. Judges differ at the second
step; the first mostly adds variance that has nothing to do with the judge.
So `eval/judge_cases.py` supplies (context, statement) pairs directly and this
asks the one binary question. Contexts are real abstracts the pipeline indexes,
not synthetic text, so the task is as hard as the job.

Reading the number
------------------
Accuracy alone is misleading here: a judge that answered "supported" to
everything would score respectably because most real statements are supported.
Kappa corrects for exactly that agreement-by-chance, which is why it is the
headline and accuracy is reported beside it rather than instead.

Two kappas are printed. The primary uses every label. The sensitivity figure
drops the four cases where the human overrode the constructed label - if a
judge's standing depends on those four, the finding is about the cases rather
than about the judge, and a reader should be able to see that without asking.
"""
import argparse
import asyncio
import json
import logging
from pathlib import Path

from eval.judge_cases import (
    agreement_with_proposed,
    as_records,
    load_labels,
    scoring_labels,
)

logger = logging.getLogger(__name__)

CACHE_PATH = Path(__file__).parent / "results" / "judge-verdicts.json"

# Deliberately the bare NLI question, with no room to hedge. The judge under
# test answers a binary in production too - ragas parses a verdict field out
# of structured output - so anything that let it answer "partially" here
# would measure a different task from the one it does.
PROMPT = """You are checking whether a statement is supported by a context.

Answer 1 if the context supports the statement, 0 if it does not. A statement \
is unsupported when it asserts anything the context does not state - a number \
that differs, an entity the context never mentions, a causal claim the context \
does not make. Inference the context clearly licenses counts as supported.

Reply with a single character, 1 or 0, and nothing else."""


def parse_verdict(text: str) -> int | None:
    """First 1 or 0 in the reply, or None if the judge said neither.

    None rather than a default: scoring an unparseable reply as either class
    would put the judge's formatting failures into its accuracy, which is a
    different property from whether it can tell supported from unsupported.
    """
    for ch in (text or "").strip():
        if ch in "01":
            return int(ch)
    return None


async def judge_all(records: list[dict], llm, concurrency: int = 8) -> dict[str, int]:
    from langchain_core.messages import HumanMessage, SystemMessage

    sem = asyncio.Semaphore(concurrency)

    async def one(rec):
        async with sem:
            reply = await llm.ainvoke([
                SystemMessage(content=PROMPT),
                HumanMessage(
                    content=f"CONTEXT:\n{rec['context']}\n\nSTATEMENT:\n{rec['statement']}"
                ),
            ])
        return rec["id"], parse_verdict(reply.content)

    pairs = await asyncio.gather(*(one(r) for r in records))
    return {cid: v for cid, v in pairs if v is not None}


def cohen_kappa(a: list[int], b: list[int]) -> float:
    """Agreement corrected for the agreement two raters would reach by chance."""
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(a, b))


def confusion(human: dict[str, int], judge: dict[str, int]) -> dict[str, int]:
    shared = sorted(set(human) & set(judge))
    out = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for cid in shared:
        h, j = human[cid], judge[cid]
        if h == 1 and j == 1:
            out["tp"] += 1
        elif h == 0 and j == 0:
            out["tn"] += 1
        elif h == 0 and j == 1:
            # The judge called an unsupported statement supported: a
            # hallucination it would have waved through.
            out["fp"] += 1
        else:
            out["fn"] += 1
    return out


def score(human: dict[str, int], judge: dict[str, int]) -> dict:
    shared = sorted(set(human) & set(judge))
    if not shared:
        return {"n": 0}
    h = [human[c] for c in shared]
    j = [judge[c] for c in shared]
    return {
        "n": len(shared),
        "kappa": round(cohen_kappa(h, j), 4),
        "accuracy": round(sum(x == y for x, y in zip(h, j)) / len(shared), 4),
        **confusion(human, judge),
    }


def by_case_type(human: dict[str, int], judge: dict[str, int]) -> dict[str, dict]:
    types = {r["id"]: r["case_type"] for r in as_records()}
    groups: dict[str, list[str]] = {}
    for cid in set(human) & set(judge):
        groups.setdefault(types[cid], []).append(cid)
    return {
        name: {
            "n": len(ids),
            "correct": sum(human[c] == judge[c] for c in ids),
        }
        for name, ids in sorted(groups.items())
    }


def main():
    parser = argparse.ArgumentParser(description="Judge agreement with human labels")
    parser.add_argument("--force", action="store_true", help="re-ask the judge")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from config import load

    from eval.run import judge_llm

    cfg = load()
    records = as_records()
    human = load_labels()
    if not human:
        raise SystemExit("no human labels - run the labelling page first")

    llm, model = judge_llm(cfg)

    cached = {}
    if CACHE_PATH.is_file() and not args.force:
        blob = json.loads(CACHE_PATH.read_text())
        if blob.get("model") == model:
            cached = blob["verdicts"]
            print(f"cached verdicts for {model} ({len(cached)})")

    if not cached:
        print(f"asking {model} about {len(records)} statements")
        cached = asyncio.run(judge_all(records, llm))
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps({"model": model, "verdicts": cached}, indent=2))

    unparsed = len(records) - len(cached)
    primary = score(scoring_labels(human), cached)
    sensitivity = score(scoring_labels(human, drop_contested=True), cached)

    print(f"\njudge: {model}")
    print(f"labelled statements: {len(human)}  judged: {len(cached)}"
          + (f"  unparseable: {unparsed}" if unparsed else ""))

    print(f"\n{'':<22}{'n':>5}{'kappa':>9}{'accuracy':>10}")
    print(f"{'all labels':<22}{primary['n']:>5}{primary['kappa']:>9.3f}{primary['accuracy']:>10.3f}")
    print(f"{'contested dropped':<22}{sensitivity['n']:>5}"
          f"{sensitivity['kappa']:>9.3f}{sensitivity['accuracy']:>10.3f}")

    print(f"\nconfusion (human vs judge): tp={primary['tp']} tn={primary['tn']} "
          f"fp={primary['fp']} fn={primary['fn']}")
    print("  fp = an unsupported statement the judge waved through "
          "(a hallucination it would miss)")

    print("\nby case type:")
    for name, s in by_case_type(human, cached).items():
        print(f"  {name:<22} {s['correct']}/{s['n']}")

    moved = agreement_with_proposed(human)
    print(f"\nhuman overrode the constructed label on {moved['n_disagreed']}"
          f"/{moved['n_labelled']} cases ({moved['rate']:.0%})")


if __name__ == "__main__":
    main()
