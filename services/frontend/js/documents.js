import { listDocuments, pdfUrl } from './api.js'
import { esc } from './dom.js'

const PAGE_SIZE = 20

export function mountDocuments(root) {
  root.innerHTML = `
    <h2>Documents</h2>
    <p class="subtitle">All papers registered in the database.</p>
    <div id="docs-body"></div>
  `
  const body = root.querySelector('#docs-body')
  let offset = 0

  async function load() {
    body.innerHTML = `<p class="muted">Loading…</p>`
    try {
      const docs = await listDocuments({ offset, limit: PAGE_SIZE })
      render(docs)
    } catch (err) {
      body.innerHTML = `<p class="notice error">${esc(err.message)}</p>`
    }
  }

  function render(docs) {
    if (docs.length === 0 && offset === 0) {
      body.innerHTML = `<p class="muted">No documents yet. Use Fetch to add papers.</p>`
      return
    }

    const rows = docs.map((doc) => {
      const authors = doc.authors?.slice(0, 3).join(', ') ?? ''
      const authorsLabel = doc.authors?.length > 3 ? `${authors} et al.` : authors
      return `
        <tr>
          <td>
            <div>${esc(doc.title)}</div>
            ${authorsLabel ? `<div class="muted">${esc(authorsLabel)}</div>` : ''}
          </td>
          <td>${esc(doc.venue ?? '—')}</td>
          <td>${doc.year ?? '—'}</td>
          <td style="text-align:right">
            <a class="pdf-link" href="${pdfUrl(doc.id)}" target="_blank" rel="noopener noreferrer">PDF</a>
          </td>
        </tr>
      `
    }).join('')

    const hasMore = docs.length === PAGE_SIZE

    body.innerHTML = `
      <div class="card" style="padding:0">
        <table>
          <thead><tr><th>Title</th><th>Venue</th><th>Year</th><th></th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      <div class="pager">
        <button id="docs-prev" ${offset === 0 ? 'disabled' : ''}>Previous</button>
        <span class="muted">${offset + 1}–${offset + docs.length}</span>
        <button id="docs-next" ${hasMore ? '' : 'disabled'}>Next</button>
      </div>
    `

    body.querySelector('#docs-prev')?.addEventListener('click', () => {
      offset = Math.max(0, offset - PAGE_SIZE)
      load()
    })
    body.querySelector('#docs-next')?.addEventListener('click', () => {
      offset += PAGE_SIZE
      load()
    })
  }

  load()
}
