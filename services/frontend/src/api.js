import { BASE, pdfUrl } from './lib/urls.js'

export { pdfUrl }

async function handle(res) {
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    let detail = text
    try { detail = JSON.parse(text).detail ?? text } catch { /* not json */ }
    throw new Error(detail || `HTTP ${res.status}`)
  }
  return res.json()
}

// One endpoint for every retrieval mode (semantic | keyword | hybrid).
// /search/keyword still exists server-side for back-compat, but there's no
// reason for the UI to special-case one mode onto its own route.
export function search(q, { mode = 'semantic', topK = 10, rerank = false, rerankTopK = 5 } = {}) {
  const params = new URLSearchParams({
    q, mode, top_k: topK, rerank, rerank_top_k: rerankTopK,
  })
  return fetch(`${BASE}/search?${params}`).then(handle)
}

export function fetchPapers({ query = '', venue = '', year = '', maxPapers = 500 } = {}) {
  return fetch(`${BASE}/fetch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, venue: venue || null, year: year || null, max_papers: maxPapers }),
  }).then(handle)
}

export function listDocuments({
  offset = 0,
  limit = 20,
  q = '',
  onlyAvailable = false,
  onlyProcessed = false,
  sort = 'newest',
} = {}) {
  const params = new URLSearchParams({
    offset,
    limit,
    only_available: onlyAvailable,
    only_processed: onlyProcessed,
    sort,
  })
  if (q) params.set('q', q)
  return fetch(`${BASE}/documents?${params}`).then(handle)
}

export function getDocument(docId) {
  return fetch(`${BASE}/documents/${docId}`).then(handle)
}

export function listDocumentChunks(docId, { offset = 0, limit = 100 } = {}) {
  const params = new URLSearchParams({ offset, limit })
  return fetch(`${BASE}/documents/${docId}/chunks?${params}`).then(handle)
}

export function getStatus() {
  return fetch(`${BASE}/status`).then(handle)
}

export function chat(message, threadId) {
  return fetch(`${BASE}/agent/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, thread_id: threadId }),
  }).then(handle)
}
