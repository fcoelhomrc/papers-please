import { pdfUrl, search } from './api.js'
import { esc } from './dom.js'

export function mountSearch(root) {
  root.innerHTML = `
    <h2>Search</h2>
    <p class="subtitle">Search indexed paper chunks by meaning.</p>
    <form id="search-form">
      <div class="search-row">
        <input type="text" id="search-q" placeholder="Search research papers…" autocomplete="off">
        <button type="submit" class="btn" id="search-btn">Search</button>
      </div>
      <div class="search-opts">
        <label><input type="checkbox" id="search-rerank"> Rerank results</label>
        <label>Top
          <select id="search-topk">
            <option value="5">5</option>
            <option value="10" selected>10</option>
            <option value="20">20</option>
            <option value="50">50</option>
          </select>
        </label>
      </div>
    </form>
    <div id="search-status"></div>
    <div id="search-results" class="result-list"></div>
  `

  const form = root.querySelector('#search-form')
  const qInput = root.querySelector('#search-q')
  const status = root.querySelector('#search-status')
  const results = root.querySelector('#search-results')
  const btn = root.querySelector('#search-btn')

  form.addEventListener('submit', async (e) => {
    e.preventDefault()
    const q = qInput.value.trim()
    if (!q) return

    const rerank = root.querySelector('#search-rerank').checked
    const topK = Number(root.querySelector('#search-topk').value)

    btn.disabled = true
    btn.textContent = 'Searching…'
    status.innerHTML = ''
    results.innerHTML = ''

    try {
      const data = await search(q, { topK, rerank, rerankTopK: 5 })
      renderResults(results, status, data)
    } catch (err) {
      status.innerHTML = `<p class="notice error">${esc(err.message)}</p>`
    } finally {
      btn.disabled = false
      btn.textContent = 'Search'
    }
  })
}

function renderResults(results, status, data) {
  status.innerHTML = `<p class="muted">${data.results.length} results${data.reranked ? ' · reranked' : ''} · <code>${esc(data.model)}</code></p>`

  if (data.results.length === 0) {
    results.innerHTML = `<p class="muted">No results found.</p>`
    return
  }

  results.innerHTML = data.results.map((r) => {
    const authors = r.authors?.slice(0, 3).join(', ') ?? ''
    const authorsLabel = r.authors?.length > 3 ? `${authors} et al.` : authors
    return `
      <article class="result-card">
        <div class="head">
          <div>
            <h3>${esc(r.title)}</h3>
            <p class="meta">${esc(authorsLabel)}${r.year ? ` · ${r.year}` : ''}</p>
          </div>
          <a class="pdf-link" href="${pdfUrl(r.doc_id)}" target="_blank" rel="noopener noreferrer">PDF</a>
        </div>
        <blockquote>${esc(r.text)}</blockquote>
        <div class="result-footer">
          ${r.page_num != null ? `<span>Page ${r.page_num}</span>` : ''}
          <span class="score">score ${r.score.toFixed(3)}</span>
        </div>
      </article>
    `
  }).join('')
}
