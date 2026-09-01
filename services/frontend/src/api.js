import { createSSEParser } from './lib/sse.js'
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

export function sendFeedback(body) {
  return fetch(`${BASE}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(handle)
}

export function getQueue({ limit = 50 } = {}) {
  return fetch(`${BASE}/queue?limit=${limit}`).then(handle)
}

export function getWorkers() {
  return fetch(`${BASE}/workers`).then(handle)
}

export function getWorkerLogs(service, { tail = 200 } = {}) {
  // Plain text, not JSON - it's a log.
  return fetch(`${BASE}/workers/${service}/logs?tail=${tail}`).then(async (res) => {
    if (!res.ok) throw new Error((await res.text()) || `HTTP ${res.status}`)
    return res.text()
  })
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

/* The same turn, reported as it happens.
 *
 * fetch + a stream reader rather than EventSource: EventSource is GET-only
 * and cannot carry a JSON body, and the chat turn is a POST.
 *
 * `onStep` fires per graph step so the panel can say what the agent is doing
 * while it does it; the promise resolves with the same payload the plain
 * endpoint returns, so a caller that only wants the answer can ignore the
 * callback entirely.
 */
export async function chatStream(message, threadId, { onStep, signal } = {}) {
  const res = await fetch(`${BASE}/agent/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, thread_id: threadId }),
    signal,
  })
  if (!res.ok) return handle(res)  // reuse the error shape of every other call

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  const feed = createSSEParser()
  let done = null

  for (;;) {
    const { value, done: finished } = await reader.read()
    if (finished) break
    for (const { event, data } of feed(decoder.decode(value, { stream: true }))) {
      if (event === 'step') onStep?.(data)
      else if (event === 'done') done = data
      // Reported in-band because the response already began - by the time a
      // stream fails there is no status code left to fail with.
      else if (event === 'error') throw new Error(data.detail || 'stream failed')
    }
  }

  if (!done) throw new Error('stream ended without a result')
  return done
}
