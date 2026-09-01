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
import json
from typing import Protocol, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from orchestrator.graph import MAX_AGENT_RECURSION


class AnswerResult(TypedDict):
    answer: str
    contexts: list[str]
    # Ranked doc_ids behind those contexts, so a run can be scored on
    # retrieval (recall/nDCG/...) as well as on what the LLM judge thinks of
    # the answer - the judge metrics can't tell "retrieval missed it" apart
    # from "retrieval found it and the model ignored it".
    doc_ids: list[int]


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

        context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
        prompt = f"Context:\n{context_block}\n\nQuestion: {question}"

        result = self._llm.invoke(
            [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        return AnswerResult(answer=result.content, contexts=contexts, doc_ids=doc_ids)


class AgenticPipeline:
    def __init__(self, agent):
        self._agent = agent

    def answer(self, question: str) -> AnswerResult:
        result = self._agent.invoke(
            {"messages": [HumanMessage(question)]},
            config={"recursion_limit": MAX_AGENT_RECURSION},
        )
        messages = result["messages"]

        # search_chunks tool_call id -> its ToolMessage result, so we can
        # pull out the chunk texts the agent actually retrieved and used.
        contexts: list[str] = []
        doc_ids: list[int] = []
        search_call_ids = {
            call["id"]
            for m in messages
            if isinstance(m, AIMessage)
            for call in m.tool_calls
            if call["name"] == "search_chunks"
        }
        for m in messages:
            if isinstance(m, ToolMessage) and m.tool_call_id in search_call_ids:
                try:
                    for chunk in json.loads(m.content):
                        contexts.append(chunk["text"])
                        if "doc_id" in chunk:
                            doc_ids.append(chunk["doc_id"])
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue

        return AnswerResult(answer=messages[-1].content, contexts=contexts, doc_ids=doc_ids)
