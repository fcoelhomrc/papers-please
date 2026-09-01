/* URL construction, kept separate from api.js's fetch wrappers so it can be
 * tested by `node --test` without stubbing the network.
 */

export const BASE = '/api'

/* `page` becomes a #page=N fragment, which every built-in browser PDF viewer
 * honours. Every search result already knew which page its passage came from
 * and the preview still opened at page 1, leaving the reader to go find it.
 *
 * The fragment goes last, after any query string: it is an instruction to the
 * viewer rather than a request parameter, and putting it first would swallow
 * the rest of the URL into the fragment.
 */
export function pdfUrl(docId, { download = false, page = null } = {}) {
  const query = download ? '?download=true' : ''
  const fragment = page ? `#page=${page}` : ''
  return `${BASE}/documents/${docId}/pdf${query}${fragment}`
}
