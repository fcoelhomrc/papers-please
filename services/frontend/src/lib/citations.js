/* Turning the agent's prose citations into things a reader can open.
 *
 * The orchestrator prompt asks for "cite the doc_id and page for every
 * claim", and the model obliges with "[doc 3, p4]" - plain text. A reader
 * could not click it, could not see which paper doc 3 is, and could not
 * check the claim without going and searching for the passage themselves.
 * The evidence list from /agent/chat has all of that; this is the join.
 *
 * Pure string work, kept out of the component so `node --test` can cover the
 * pattern matching - which is where the bugs are.
 */

/* Matches the shapes the model actually produces. The prompt asks for
 * "[doc 3, p4]" but a language model will also write "[doc 3]",
 * "[doc 3, p. 4]", "[Doc 3, page 4]" and "[doc 3, pp. 4-5]", and a citation
 * that fails to match renders as raw brackets in the middle of a sentence -
 * worse than not having tried.
 */
const CITATION = /\[\s*doc\.?\s*(\d+)\s*(?:,\s*(?:pp?\.?|pages?)\s*(\d+)(?:\s*[-–]\s*\d+)?)?\s*\]/gi

/* doc_id -> the 1-based number shown against its evidence.
 *
 * Not the doc_id itself: those are database keys in the hundreds, and
 * "[147]" in running text reads as a footnote to a bibliography the reader
 * doesn't have.
 *
 * Counted over distinct *documents*, not over cards. A citation names a
 * paper, and a paper often contributes two or three passages - numbering by
 * card position would make the sequence skip (1, 2, then 4) and would point
 * "[doc 3]" at only the first of doc 3's passages. Several cards sharing a
 * number is the honest rendering: one source, several supporting passages.
 */
export function citationIndex(evidence = []) {
  const index = new Map()
  for (const e of evidence) {
    if (e.doc_id != null && !index.has(e.doc_id)) index.set(e.doc_id, index.size + 1)
  }
  return index
}

/* Rewrites citations into markdown links with a `citation:` href, which the
 * renderer turns into a superscript. Left as markdown rather than injected
 * as HTML so the reply keeps flowing through one markdown pass - and so a
 * citation inside a bold span or a list item still works.
 *
 * A citation naming a doc that isn't in the evidence is left exactly as the
 * model wrote it: silently dropping it would hide that the model cited
 * something it never retrieved, which is the single most useful signal in
 * the whole panel.
 */
export function linkCitations(text, evidence = []) {
  if (!text) return ''
  const index = citationIndex(evidence)

  return text.replace(CITATION, (match, docId, page) => {
    const n = index.get(Number(docId))
    if (!n) return match
    return `[${n}](citation:${docId}${page ? `:${page}` : ''})`
  })
}

/* Parses a `citation:` href back into its parts. */
export function parseCitationHref(href = '') {
  const m = /^citation:(\d+)(?::(\d+))?$/.exec(href)
  if (!m) return null
  return { docId: Number(m[1]), page: m[2] ? Number(m[2]) : null }
}

/* Whether a turn is an honest "nothing relevant here".
 *
 * Retrieval can return nothing on purpose - `search.min_rerank_score` exists
 * so it can decline rather than always hand back its top k. That looks
 * identical to a failed search unless the trace is consulted: both produce
 * an answer with no citations. The difference matters, because one means
 * "the library doesn't cover this" and the other means "try again".
 */
export function isAbstention(trace = [], evidence = []) {
  const searches = trace.filter((s) => s.tool === 'search_chunks')
  return (
    searches.length > 0 && searches.every((s) => s.ok) && evidence.length === 0
  )
}

export function hasFailure(trace = []) {
  return trace.some((s) => !s.ok)
}
