import * as ScrollArea from '@radix-ui/react-scroll-area'
import clsx from 'clsx'
import { ArrowUp, RotateCcw, Sparkles, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { useChat } from '../hooks/queries'
import { isAbstention, linkCitations, parseCitationHref } from '../lib/citations.js'
import {
  AbstentionNote,
  EvidenceCards,
  LiveSteps,
  ToolTrace,
} from './AgentEvidence.jsx'
import { Button, Textarea } from './ui.jsx'

const THREAD_KEY = 'pp-chat-thread-id'

const SUGGESTIONS = [
  'Find recent papers on transformer attention',
  'Has anyone studied fall recovery in legged robots?',
  'What does the library say about federated learning privacy?',
]

function newThreadId() {
  return crypto.randomUUID()
}

function getThreadId() {
  let id = localStorage.getItem(THREAD_KEY)
  if (!id) {
    id = newThreadId()
    localStorage.setItem(THREAD_KEY, id)
  }
  return id
}

/* The agent's "[doc 3, p4]" rendered as a clickable superscript.
 *
 * Scrolls to the matching source card rather than opening the PDF directly:
 * the card is where the paper's identity lives, and jumping straight into a
 * PDF viewer skips the step where the reader finds out which paper they are
 * about to read. The card itself is the link to the page.
 */
function Citation({ href, children }) {
  const parsed = parseCitationHref(href)
  if (!parsed) return null

  return (
    <button
      type="button"
      onClick={(e) => {
        const card = e.currentTarget
          .closest('[data-message]')
          ?.querySelector(`#evidence-${parsed.docId}`)
        card?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
        card?.animate(
          [{ borderColor: 'var(--color-accent, currentColor)' }, {}],
          { duration: 1200 },
        )
      }}
      className="mx-px inline-flex h-3.5 min-w-3.5 items-center justify-center rounded bg-accent/15 px-1 align-super text-[10px] font-semibold tabular-nums text-accent transition-colors hover:bg-accent/30"
      title={parsed.page ? `Source ${children} · page ${parsed.page}` : `Source ${children}`}
    >
      {children}
    </button>
  )
}

function Bubble({ role, content, evidence, trace }) {
  if (role === 'user') {
    return (
      <div className="flex justify-end animate-slide-up">
        <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-accent px-3.5 py-2 text-sm leading-relaxed text-accent-ink">
          {content}
        </div>
      </div>
    )
  }

  const isError = role === 'error'

  return (
    <div className="flex justify-start animate-slide-up">
      <div
        data-message
        className={clsx(
          'max-w-[92%] rounded-2xl rounded-bl-md px-3.5 py-2 text-sm leading-relaxed',
          isError
            ? 'border border-danger/25 bg-danger-soft text-danger'
            : 'border border-border bg-canvas text-ink',
        )}
      >
        {isError ? (
          <span className="whitespace-pre-wrap">{content}</span>
        ) : (
          <>
            <div className="prose-chat">
              <ReactMarkdown
                components={{
                  a: ({ href, children }) =>
                    parseCitationHref(href) ? (
                      <Citation href={href}>{children}</Citation>
                    ) : (
                      <a href={href} target="_blank" rel="noreferrer">
                        {children}
                      </a>
                    ),
                }}
              >
                {linkCitations(content, evidence)}
              </ReactMarkdown>
            </div>
            <EvidenceCards evidence={evidence} />
            {isAbstention(trace, evidence) && <AbstentionNote />}
            <ToolTrace trace={trace} />
          </>
        )}
      </div>
    </div>
  )
}

export default function ChatPanel({ open, onClose }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [steps, setSteps] = useState([])
  const [threadId, setThreadId] = useState(getThreadId)
  const viewportRef = useRef(null)
  const inputRef = useRef(null)
  const chat = useChat()

  useEffect(() => {
    if (viewportRef.current) {
      viewportRef.current.scrollTop = viewportRef.current.scrollHeight
    }
  }, [messages, steps, chat.isPending, open])

  useEffect(() => {
    if (open) inputRef.current?.focus()
  }, [open])

  function send(text) {
    const trimmed = text.trim()
    if (!trimmed || chat.isPending) return

    setMessages((m) => [...m, { role: 'user', content: trimmed }])
    setInput('')
    setSteps([])
    chat.mutate(
      // Steps arrive over SSE while the turn runs, so the panel can say what
      // the agent is doing instead of showing an ellipsis until it's over.
      { message: trimmed, threadId, onStep: (s) => setSteps((prev) => [...prev, s]) },
      {
        onSuccess: (data) => {
          setSteps([])
          setMessages((m) => [
            ...m,
            {
              role: 'agent',
              content: data.reply,
              evidence: data.evidence,
              trace: data.trace,
            },
          ])
        },
        onError: (err) => {
          setSteps([])
          setMessages((m) => [...m, { role: 'error', content: err.message }])
        },
      },
    )
  }

  function handleReset() {
    const id = newThreadId()
    localStorage.setItem(THREAD_KEY, id)
    setThreadId(id)
    setMessages([])
    setSteps([])
    chat.reset()
  }

  return (
    <aside
      // One mount at every breakpoint - a full-height docked rail from lg up,
      // an overlay sheet below it. Swapping between a rail and a Dialog would
      // remount and wipe the conversation on resize.
      className={clsx(
        'z-40 flex-col border-border bg-surface',
        'fixed inset-y-0 right-0 w-full max-w-[420px] border-l shadow-panel',
        'lg:static lg:max-w-none lg:shadow-none',
        open ? 'flex lg:w-[400px] lg:shrink-0 animate-slide-in-right lg:animate-none' : 'hidden',
      )}
      aria-label="Agent chat"
    >
      <div className="flex h-14 shrink-0 items-center justify-between border-b border-border px-4">
        <div className="flex items-center gap-2">
          <Sparkles size={15} className="text-accent" />
          <span className="text-sm font-semibold">Agent</span>
        </div>
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="icon" onClick={handleReset} title="New conversation">
            <RotateCcw size={14} />
          </Button>
          <Button variant="ghost" size="icon" onClick={onClose} title="Close chat">
            <X size={15} />
          </Button>
        </div>
      </div>

      <ScrollArea.Root className="flex-1 overflow-hidden">
        <ScrollArea.Viewport ref={viewportRef} className="h-full w-full px-4 py-4">
          {messages.length === 0 ? (
            <div className="space-y-4">
              <p className="text-sm text-muted">
                Ask the agent to fetch new papers, or to answer questions from what's already in
                the library. Answers cite their sources.
              </p>
              <div className="space-y-1.5">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => send(s)}
                    className="block w-full rounded-lg border border-border bg-canvas px-3 py-2 text-left text-sm text-muted transition-colors hover:border-border-strong hover:text-ink"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              {messages.map((m, i) => (
                <Bubble key={i} {...m} />
              ))}
              {chat.isPending && <LiveSteps steps={steps} />}
            </div>
          )}
        </ScrollArea.Viewport>
        <ScrollArea.Scrollbar orientation="vertical" className="w-2 p-0.5">
          <ScrollArea.Thumb className="rounded-full bg-border-strong" />
        </ScrollArea.Scrollbar>
      </ScrollArea.Root>

      <form
        onSubmit={(e) => {
          e.preventDefault()
          send(input)
        }}
        className="flex shrink-0 items-end gap-2 border-t border-border p-3"
      >
        <Textarea
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              send(input)
            }
          }}
          placeholder="Message the agent…"
          rows={1}
          className="max-h-32 min-h-[36px] flex-1 bg-canvas"
        />
        <Button type="submit" variant="primary" size="icon" disabled={!input.trim() || chat.isPending}>
          <ArrowUp size={15} />
        </Button>
      </form>
    </aside>
  )
}
