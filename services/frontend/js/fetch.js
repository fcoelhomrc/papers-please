import { fetchPapers } from './api.js'
import { esc } from './dom.js'

export function mountFetch(root) {
  root.innerHTML = `
    <h2>Fetch Papers</h2>
    <p class="subtitle">Query Semantic Scholar and register new papers for download and indexing.</p>
    <form id="fetch-form" class="card">
      <div class="field">
        <label>Query</label>
        <input type="text" id="fetch-query" placeholder="e.g. attention transformers">
      </div>
      <div class="row">
        <div class="field">
          <label>Venue</label>
          <input type="text" id="fetch-venue" placeholder="e.g. NeurIPS">
        </div>
        <div class="field">
          <label>Year</label>
          <input type="text" id="fetch-year" placeholder="e.g. 2023 or 2020-2024">
        </div>
      </div>
      <div class="field">
        <label>Max papers</label>
        <input type="number" id="fetch-max" value="500" min="1" max="5000">
      </div>
      <button type="submit" class="btn" id="fetch-btn" style="width:100%">Fetch</button>
    </form>
    <div id="fetch-status"></div>
  `

  const form = root.querySelector('#fetch-form')
  const btn = root.querySelector('#fetch-btn')
  const status = root.querySelector('#fetch-status')

  form.addEventListener('submit', async (e) => {
    e.preventDefault()
    btn.disabled = true
    btn.textContent = 'Fetching…'
    status.innerHTML = ''

    try {
      const data = await fetchPapers({
        query: root.querySelector('#fetch-query').value,
        venue: root.querySelector('#fetch-venue').value,
        year: root.querySelector('#fetch-year').value,
        maxPapers: Number(root.querySelector('#fetch-max').value),
      })
      status.innerHTML = `<p class="notice success">Registered <strong>${data.fetched}</strong> papers. The worker will download and index them shortly.</p>`
    } catch (err) {
      status.innerHTML = `<p class="notice error">${esc(err.message)}</p>`
    } finally {
      btn.disabled = false
      btn.textContent = 'Fetch'
    }
  })
}
