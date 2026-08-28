import * as ScrollArea from '@radix-ui/react-scroll-area'
import { ArrowUp, MessageCircle, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { chat as sendChat } from '../api'

function Bubble({ role, content, toolCalls }) {
  if (role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-accent text-accent-ink px-3.5 py-2 text-[13.5px] leading-relaxed whitespace-pre-wrap">
          {content}
        </div>
      </div>
    )
  }
  const isError = role === 'error'
  return (
    <div className="flex justify-start">
      <div
        className={`max-w-[90%] rounded-2xl rounded-bl-sm px-3.5 py-2 text-[13.5px] leading-relaxed whitespace-pre-wrap ${
          isError
            ? 'bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-900 text-red-700 dark:text-red-400'
            : 'bg-canvas border border-border text-ink'
        }`}
      >
        {content}
        {toolCalls?.length > 0 && (
          <div className="mt-1.5 text-[11px] text-faint">used: {toolCalls.join(', ')}</div>
        )}
      </div>
    </div>
  )
}

export default function Chat() {
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const viewportRef = useRef(null)

  useEffect(() => {
    if (viewportRef.current) {
      viewportRef.current.scrollTop = viewportRef.current.scrollHeight
    }
  }, [messages, open])

  async function handleSend(e) {
    e.preventDefault()
    const text = input.trim()
    if (!text || sending) return

    setMessages((m) => [...m, { role: 'user', content: text }])
    setInput('')
    setSending(true)

    try {
      const data = await sendChat(text)
      setMessages((m) => [...m, { role: 'agent', content: data.reply, toolCalls: data.tool_calls }])
    } catch (err) {
      setMessages((m) => [...m, { role: 'error', content: err.message }])
    } finally {
      setSending(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend(e)
    }
  }

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="fixed bottom-6 right-6 rounded-full bg-accent text-accent-ink shadow-lg p-3.5 hover:opacity-90 transition-opacity"
        aria-label="Open chat"
      >
        <MessageCircle size={20} />
      </button>
    )
  }

  return (
    <div className="fixed bottom-6 right-6 w-[380px] h-[560px] max-h-[80vh] rounded-2xl border border-border bg-surface shadow-2xl flex flex-col overflow-hidden">
      <div className="h-12 flex items-center justify-between px-4 border-b border-border shrink-0">
        <span className="text-[13.5px] font-semibold">Agent</span>
        <button onClick={() => setOpen(false)} className="text-muted hover:text-ink" aria-label="Close chat">
          <X size={16} />
        </button>
      </div>

      <ScrollArea.Root className="flex-1 overflow-hidden">
        <ScrollArea.Viewport ref={viewportRef} className="h-full w-full px-4 py-4">
          {messages.length === 0 ? (
            <p className="text-[13px] text-faint">
              Ask me to fetch papers ("find recent papers on transformer attention") or answer
              questions from what's already in the library ("has anyone studied X?").
            </p>
          ) : (
            <div className="space-y-3">
              {messages.map((m, i) => (
                <Bubble key={i} {...m} />
              ))}
              {sending && (
                <div className="flex justify-start">
                  <div className="rounded-2xl rounded-bl-sm bg-canvas border border-border px-3.5 py-2 text-[13px] text-faint">
                    thinking…
                  </div>
                </div>
              )}
            </div>
          )}
        </ScrollArea.Viewport>
        <ScrollArea.Scrollbar orientation="vertical" className="w-2">
          <ScrollArea.Thumb className="bg-border rounded-full" />
        </ScrollArea.Scrollbar>
      </ScrollArea.Root>

      <form onSubmit={handleSend} className="border-t border-border p-3 flex items-end gap-2 shrink-0">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Message the agent…"
          rows={1}
          className="flex-1 resize-none rounded-xl border border-border bg-canvas px-3 py-2 text-[13.5px] focus:outline-none focus:ring-1 focus:ring-ink/20 max-h-28"
        />
        <button
          type="submit"
          disabled={sending || !input.trim()}
          className="rounded-full bg-accent text-accent-ink p-2 disabled:opacity-40 shrink-0"
          aria-label="Send"
        >
          <ArrowUp size={16} />
        </button>
      </form>
    </div>
  )
}
