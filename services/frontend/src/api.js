const BASE = '/api'

async function handle(res) {
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(text || `HTTP ${res.status}`)
  }
  return res.json()
}

export function search(q, { topK = 10, rerank = false, rerankTopK = 5 } = {}) {
  const params = new URLSearchParams({
    q,
    top_k: topK,
    rerank,
    rerank_top_k: rerankTopK,
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

export function listDocuments({ offset = 0, limit = 20 } = {}) {
  const params = new URLSearchParams({ offset, limit })
  return fetch(`${BASE}/documents?${params}`).then(handle)
}

export function pdfUrl(docId) {
  return `${BASE}/documents/${docId}/pdf`
}
