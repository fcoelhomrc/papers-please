import { mountSearch } from './search.js'
import { mountFetch } from './fetch.js'
import { mountDocuments } from './documents.js'
import { mountChat } from './chat.js'

const views = {
  search: { root: document.getElementById('view-search'), mount: mountSearch },
  fetch: { root: document.getElementById('view-fetch'), mount: mountFetch },
  documents: { root: document.getElementById('view-documents'), mount: mountDocuments },
}
const mounted = new Set()

function showView(name) {
  for (const [key, view] of Object.entries(views)) {
    view.root.classList.toggle('active', key === name)
  }
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.view === name)
  })
  if (!mounted.has(name)) {
    views[name].mount(views[name].root)
    mounted.add(name)
  } else if (name === 'documents') {
    // documents list can go stale between visits - remount for a fresh page
    views.documents.mount(views.documents.root)
  }
}

document.querySelectorAll('.tab-btn').forEach((btn) => {
  btn.addEventListener('click', () => showView(btn.dataset.view))
})

showView('search')

mountChat({
  toggleBtn: document.getElementById('chat-toggle'),
  panel: document.getElementById('chat-panel'),
  closeBtn: document.getElementById('chat-close'),
  messages: document.getElementById('chat-messages'),
  form: document.getElementById('chat-form'),
  input: document.getElementById('chat-input'),
})
