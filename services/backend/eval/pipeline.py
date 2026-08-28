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


class AnswerResult(TypedDict):
    answer: str
    contexts: list[str]


class Pipeline(Protocol):
    def answer(self, question: str) -> AnswerResult: ...


FIXED_SYSTEM_PROMPT = """Answer the question using only the context passages below. If the
context doesn't contain the answer, say so plainly - don't guess or use
outside knowledge. Be concise."""


class FixedPipeline:
    def __init__(self, llm, search_engine, top_k: int = 5, rerank: bool = True):
        self._llm = llm
        self._search_engine = search_engine
        self._top_k = top_k
        self._rerank = rerank

    def answer(self, question: str) -> AnswerResult:
        response = self._search_engine.search(
            question, top_k=self._top_k, rerank=self._rerank, rerank_top_k=self._top_k
        )
        contexts = [r.text for r in response.results]

        context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
        prompt = f"Context:\n{context_block}\n\nQuestion: {question}"

        result = self._llm.invoke(
            [
                {"role": "system", "content": FIXED_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        return AnswerResult(answer=result.content, contexts=contexts)


class AgenticPipeline:
    def __init__(self, agent):
        self._agent = agent

    def answer(self, question: str) -> AnswerResult:
        result = self._agent.invoke({"messages": [HumanMessage(question)]})
        messages = result["messages"]

        # search_chunks tool_call id -> its ToolMessage result, so we can
        # pull out the chunk texts the agent actually retrieved and used.
        contexts: list[str] = []
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
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue

        return AnswerResult(answer=messages[-1].content, contexts=contexts)
