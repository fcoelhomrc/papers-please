"""Reading an agent run's message list back out: what it found, and what it did.

The agent's own prose cites papers as "[doc 3, p4]", which is unlinked text -
a reader has no way to open the source and no way to check the claim. The
chunks behind those citations were right there in the tool results and were
being thrown away one function call before the response was built.

Both extractors live here rather than in the API or in eval/pipeline.py
because both places need them and they must not drift: eval scores the
retrieval behind an answer using exactly the evidence the UI shows, so a
divergence would mean the reported score describes something the user never
saw.

Neither function raises. They read whatever a completed (or half-completed)
run left behind, and a run that failed mid-tool is precisely when the trace
is most worth showing.
"""
import json

from langchain_core.messages import AIMessage, ToolMessage

SEARCH_TOOL = "search_chunks"

# Kept off the wire: `text` is the whole chunk (or its neighbour-expanded
# window), which is already in the model's answer and would multiply the
# response size by ~20x for a citation card that shows a title and a page.
EVIDENCE_FIELDS = ("doc_id", "chunk_id", "title", "authors", "year", "page_num", "score")


def _tool_results(messages) -> dict[str, ToolMessage]:
    return {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}


def _parse(message: ToolMessage):
    """A tool result as data, or None if it isn't JSON.

    Tools return dicts and lists, but LangChain serialises them to a string
    for the model - and a tool that returned a bare error string never was
    JSON. Both are normal, neither is an error worth propagating.
    """
    content = message.content
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None


def extract_evidence(messages) -> list[dict]:
    """The chunks the agent actually retrieved, in the order it saw them.

    Deduplicated on chunk_id keeping the best score: an agent that searches
    twice with refined queries usually surfaces the same strong chunk in both
    result sets, and showing it as two citations would misrepresent one piece
    of evidence as corroboration by two.
    """
    search_calls = {
        call["id"]
        for m in messages
        if isinstance(m, AIMessage)
        for call in m.tool_calls
        if call["name"] == SEARCH_TOOL
    }

    best: dict[int, dict] = {}
    order: list[int] = []
    for message in messages:
        if not isinstance(message, ToolMessage) or message.tool_call_id not in search_calls:
            continue
        chunks = _parse(message)
        if not isinstance(chunks, list):
            continue
        for chunk in chunks:
            # An error result is a dict with only "error" in it, and a chunk
            # with no id can't be cited or deduplicated.
            if not isinstance(chunk, dict) or "chunk_id" not in chunk:
                continue
            cid = chunk["chunk_id"]
            item = {k: chunk.get(k) for k in EVIDENCE_FIELDS}
            if cid not in best:
                best[cid] = item
                order.append(cid)
            elif (item.get("score") or 0) > (best[cid].get("score") or 0):
                best[cid] = item
    return [best[cid] for cid in order]


def _summarise(tool_name: str, result) -> str:
    """One line describing what came back, for the trace strip.

    A summary rather than the payload: the point of a trace is to be
    skimmable while the answer is what gets read, and dumping five chunks of
    prose under the answer buries it.
    """
    if isinstance(result, list):
        if result and isinstance(result[0], dict) and "error" in result[0]:
            return result[0]["error"]
        if tool_name == SEARCH_TOOL:
            return f"{len(result)} chunk{'s' if len(result) != 1 else ''}"
        return f"{len(result)} result{'s' if len(result) != 1 else ''}"
    if isinstance(result, dict):
        if "error" in result:
            return str(result["error"])
        if "title" in result:
            return str(result["title"])
        return ", ".join(f"{k}={v}" for k, v in list(result.items())[:3])
    return str(result) if result is not None else ""


def _is_error(result) -> bool:
    if isinstance(result, dict):
        return "error" in result
    if isinstance(result, list):
        return bool(result) and isinstance(result[0], dict) and "error" in result[0]
    return isinstance(result, str) and result.lower().startswith("error")


def extract_trace(messages) -> list[dict]:
    """Every tool call in this run, with its arguments and what came back.

    This is the "tool use" reporting the panel was missing: the API kept only
    `call["name"]`, so the UI could say a search happened but not what was
    searched for, what came back, or whether it worked.

    Calls with no matching result are still listed - an unanswered tool call
    means the run was cut short (recursion limit, a crash), which is exactly
    the moment someone reads a trace.
    """
    results = _tool_results(messages)
    trace = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            result_message = results.get(call["id"])
            parsed = _parse(result_message) if result_message else None
            trace.append(
                {
                    "tool": call["name"],
                    "args": call.get("args") or {},
                    "summary": (
                        _summarise(call["name"], parsed)
                        if result_message
                        else "(no result — run ended before this returned)"
                    ),
                    "ok": bool(result_message) and not _is_error(parsed),
                }
            )
    return trace


def extract_contexts(messages) -> tuple[list[str], list[int]]:
    """The retrieved chunk texts and their doc_ids, for eval's scoring.

    Not deduplicated, unlike extract_evidence: eval measures what the
    pipeline actually put in front of the model, and collapsing a repeated
    chunk would understate the context the answer was generated from.
    """
    search_calls = {
        call["id"]
        for m in messages
        if isinstance(m, AIMessage)
        for call in m.tool_calls
        if call["name"] == SEARCH_TOOL
    }

    contexts: list[str] = []
    doc_ids: list[int] = []
    for message in messages:
        if not isinstance(message, ToolMessage) or message.tool_call_id not in search_calls:
            continue
        chunks = _parse(message)
        if not isinstance(chunks, list):
            continue
        for chunk in chunks:
            if not isinstance(chunk, dict) or "text" not in chunk:
                continue
            contexts.append(chunk["text"])
            if "doc_id" in chunk:
                doc_ids.append(chunk["doc_id"])
    return contexts, doc_ids
