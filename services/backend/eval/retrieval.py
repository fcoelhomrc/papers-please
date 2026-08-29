"""Rank metrics for retrieval, computed without an LLM.

Everything here scores a ranked list of retrieved documents against the
`relevant_source_ids` labels in eval/dataset.jsonl. No judge, no API calls -
so unlike the Ragas generation metrics these are cheap enough to sweep
parameters with (see eval/sweep.py).

On the two kinds of question in the dataset:

- **retrieval questions** (42 of 50) have at least one relevant document.
  recall/precision/MRR/nDCG are meaningful.
- **abstention questions** (8 of 50) have none - "does the library cover
  social media sentiment analysis?" (it doesn't). Recall is undefined there:
  there is nothing to recall, and scoring them 0.0 would drag the mean down
  for behaving correctly, while scoring 1.0 would reward a system that
  retrieves piles of junk. They're reported separately, measured by whether
  retrieval stayed quiet.
"""
import math


def _relevant_ranks(retrieved: list[str], relevant: set[str]) -> list[int]:
    """1-based ranks of the retrieved docs that are relevant."""
    return [i for i, doc in enumerate(retrieved, start=1) if doc in relevant]


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant documents that made the top k.

    Deduped: retrieval returns chunks, several of which routinely come from
    the same paper, and counting one paper twice would let a single lucky
    document report recall > 1.
    """
    if not relevant:
        raise ValueError("recall is undefined with no relevant documents")
    found = set(retrieved[:k]) & relevant
    return len(found) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top k that is relevant. Divides by k, not by the
    number retrieved: a system returning 2 results where 10 were asked for
    shouldn't score the same as one that filled all 10 correctly."""
    if k <= 0:
        raise ValueError("k must be positive")
    return len(set(retrieved[:k]) & relevant) / k


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """1.0 if anything relevant appears in the top k. The metric that matters
    when the generator only needs one good passage to answer from."""
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant document, 0 if none.

    Rewards putting a good document *first* rather than merely somewhere in
    the list - which is what matters when the generator reads top-down and
    has a limited context budget.
    """
    ranks = _relevant_ranks(retrieved, relevant)
    return 1.0 / ranks[0] if ranks else 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalised discounted cumulative gain, binary relevance.

    Unlike recall it is rank-aware, and unlike MRR it credits *every*
    relevant document found, not just the first - the one that behaves
    sensibly on the multi-document questions ("which papers use an
    autoencoder?").
    """
    if not relevant:
        raise ValueError("nDCG is undefined with no relevant documents")

    seen: set[str] = set()
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k], start=1):
        if doc in relevant and doc not in seen:
            seen.add(doc)
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def score_question(retrieved: list[str], relevant: set[str], k: int) -> dict:
    """All metrics for one question.

    `retrieved` is the ranked list of source_ids, duplicates included (they
    are deduped per-metric where that's the correct behaviour). Abstention
    questions return only the metrics that mean something for them.
    """
    if not relevant:
        return {
            "abstention": True,
            # Nothing is relevant, so anything retrieved is a false positive.
            # 1.0 = retrieved nothing, which is the correct behaviour here.
            "abstention_precision": 1.0 if not retrieved[:k] else 0.0,
            "false_positives": len(set(retrieved[:k])),
        }

    return {
        "abstention": False,
        "recall": recall_at_k(retrieved, relevant, k),
        "precision": precision_at_k(retrieved, relevant, k),
        "hit_rate": hit_rate_at_k(retrieved, relevant, k),
        "mrr": mrr(retrieved, relevant),
        "ndcg": ndcg_at_k(retrieved, relevant, k),
    }


RETRIEVAL_METRICS = ("recall", "precision", "hit_rate", "mrr", "ndcg")


def aggregate(per_question: list[dict]) -> dict:
    """Mean of each metric over the questions it's defined for.

    Retrieval and abstention questions are averaged separately on purpose -
    mixing them produces a single number that moves for two unrelated
    reasons and can't be acted on.
    """
    retrieval = [q for q in per_question if not q["abstention"]]
    abstention = [q for q in per_question if q["abstention"]]

    out = {f"{m}": _mean([q[m] for q in retrieval]) for m in RETRIEVAL_METRICS}
    out["n_retrieval"] = len(retrieval)
    out["n_abstention"] = len(abstention)
    if abstention:
        out["abstention_precision"] = _mean([q["abstention_precision"] for q in abstention])
        out["mean_false_positives"] = _mean([q["false_positives"] for q in abstention])
    return out


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
