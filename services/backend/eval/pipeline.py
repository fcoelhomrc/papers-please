"""Two answer-producing paths over the same paper library, so eval can
compare them on the same questions instead of eval being a claim with no
evidence behind it:

- FixedPipeline: retrieve top-k (with rerank) then one LLM call synthesizes
  an answer from those chunks. No tool loop, no judgment about what/how
  much to search - a fixed-shape RAG baseline.
- AgenticPipeline: the actual orchestrator agent (search_chunks/get_document
  tool-calling loop, same one wired to /agent/chat) decides how to search
  and answers from what it finds.

Both implement the same Pipeline protocol so eval/run.py doesn't care which
one it's scoring.
"""
from typing import Protocol, TypedDict

from langchain_core.messages import HumanMessage

from orchestrator.evidence import extract_contexts
from orchestrator.graph import MAX_AGENT_RECURSION


class AnswerResult(TypedDict):
    answer: str
    contexts: list[str]
    # Ranked doc_ids behind those contexts, so a run can be scored on
    # retrieval (recall/nDCG/...) as well as on what the LLM judge thinks of
    # the answer - the judge metrics can't tell "retrieval missed it" apart
    # from "retrieval found it and the model ignored it".
    doc_ids: list[int]
    # The same list at chunk granularity. Document-level labels cannot say
    # whether the right *passage* was found, and the curated test set labels
    # chunks - so scoring the judged branch against the free branch's own
    # ground truth needs these, not doc_ids.
    chunk_ids: list[int]


class Pipeline(Protocol):
    def answer(self, question: str) -> AnswerResult: ...


class FixedPipeline:
    # The synthesis prompt now lives in prompts/fixed_rag/<version>.md - the
    # baseline's prompt is as much a part of a reported score as the agent's,
    # so it gets the same version treatment.
    def __init__(
        self,
        llm,
        search_engine,
        system_prompt: str,
        top_k: int = 5,
        rerank: bool = True,
        candidates: int | None = None,
    ):
        self._llm = llm
        self._search_engine = search_engine
        self._system_prompt = system_prompt
        self._top_k = top_k
        self._rerank = rerank
        self._candidates = candidates

    def answer(self, question: str) -> AnswerResult:
        # Same wide-pool retrieval the agent's search_chunks uses, so the
        # baseline stays a comparison of *pipeline shape* rather than an
        # accidental comparison of retrieval settings.
        response = self._search_engine.search(
            question,
            top_k=self._top_k,
            rerank=self._rerank,
            rerank_top_k=self._top_k,
            candidates=self._candidates,
        )
        contexts = [r.text for r in response.results]
        doc_ids = [r.doc_id for r in response.results]
        chunk_ids = [r.chunk_id for r in response.results]

        context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
        prompt = f"Context:\n{context_block}\n\nQuestion: {question}"

        result = self._llm.invoke(
            [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        return AnswerResult(
            answer=result.content,
            contexts=contexts,
            doc_ids=doc_ids,
            chunk_ids=chunk_ids,
        )


class ArmPipeline:
    """FixedPipeline, but retrieval goes through a query arm.

    Exists so the judged branch can score the arms on answer quality, not only
    on chunk-id retrieval. The free branch already showed every arm losing
    ground on retrieval; whether that costs anything a reader would notice in
    the *answer* is a different question, and only a judge can answer it.

    The transform itself is read from the arm's disk cache - the same queries
    the free branch swept - so a judged arm and a free arm are the same arm,
    not two independent LLM rewrites of the same question.
    """

    def __init__(self, llm, search_engine, system_prompt, arm, queries,
                 mode, top_k=10):
        self._llm = llm
        self._engine = search_engine
        self._system_prompt = system_prompt
        self._arm = arm
        self._queries = queries
        self._mode = mode
        self._top_k = top_k

    def answer(self, question: str) -> AnswerResult:
        from config import load
        from eval.query_arms import retrieve_for

        qs = self._queries.get(question) or [question]
        chunks = retrieve_for(
            self._engine, self._arm, qs, self._top_k, load().search, self._mode
        )
        # `text`, not `chunk_text`: _row_to_chunk names it `text`, and the
        # KeyError from getting this wrong is swallowed by answer_all's
        # per-question guard, so it surfaces as every metric scoring 0 rather
        # than as a crash.
        contexts = [c["text"] for c in chunks]
        context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))

        result = self._llm.invoke([
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"},
        ])
        return AnswerResult(
            answer=result.content,
            contexts=contexts,
            doc_ids=[c["doc_id"] for c in chunks],
            chunk_ids=[c["chunk_id"] for c in chunks],
        )


class AgenticPipeline:
    def __init__(self, agent):
        self._agent = agent

    def answer(self, question: str) -> AnswerResult:
        result = self._agent.invoke(
            {"messages": [HumanMessage(question)]},
            config={"recursion_limit": MAX_AGENT_RECURSION},
        )
        messages = result["messages"]

        # Shared with the API's citation cards (orchestrator/evidence.py), so
        # a reported score describes the same retrieval the user is shown.
        contexts, doc_ids = extract_contexts(messages)

        # No chunk_ids: extract_contexts reports document granularity, and the
        # agentic arm is out of the evaluation anyway. An empty list scores as
        # "retrieved nothing" rather than silently as a miss, which is the
        # honest reading if this path is ever measured again.
        return AnswerResult(
            answer=messages[-1].content,
            contexts=contexts,
            doc_ids=doc_ids,
            chunk_ids=[],
        )
