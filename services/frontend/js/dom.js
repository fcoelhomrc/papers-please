export function esc(s) {
  const div = document.createElement('div')
  div.textContent = s ?? ''
  return div.innerHTML
}

export function el(html) {
  const tpl = document.createElement('template')
  tpl.innerHTML = html.trim()
  return tpl.content.firstElementChild
}
