import { chat } from './api.js'

export function mountChat({ toggleBtn, panel, closeBtn, messages, form, input }) {
  toggleBtn.addEventListener('click', () => panel.classList.toggle('open'))
  closeBtn.addEventListener('click', () => panel.classList.remove('open'))

  form.addEventListener('submit', async (e) => {
    e.preventDefault()
    const text = input.value.trim()
    if (!text) return

    appendMessage(messages, 'user', text)
    input.value = ''
    input.disabled = true

    try {
      const data = await chat(text)
      appendMessage(messages, 'agent', data.reply, data.tool_calls)
    } catch (err) {
      appendMessage(messages, 'error', err.message)
    } finally {
      input.disabled = false
      input.focus()
    }
  })
}

function appendMessage(container, role, text, toolCalls) {
  const div = document.createElement('div')
  div.className = `chat-msg ${role}`
  div.textContent = text
  if (toolCalls?.length) {
    const tools = document.createElement('div')
    tools.className = 'chat-tools'
    tools.textContent = `used: ${toolCalls.join(', ')}`
    div.appendChild(tools)
  }
  container.appendChild(div)
  container.scrollTop = container.scrollHeight
}
