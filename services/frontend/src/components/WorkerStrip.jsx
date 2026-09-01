import clsx from 'clsx'
import { ChevronDown } from 'lucide-react'
import { useState } from 'react'
import { Badge, Card, Skeleton } from './ui.jsx'
import { useWorkerLogs } from '../hooks/queries'

/* Are the pipeline workers actually running?
 *
 * The Queue page could show a full backlog and a stopped worker at the same
 * time and looked identical in both cases — which is how six stuck
 * downloads got reported as a broken pipeline when the download worker
 * simply was not up. Counters say how much is left; this says whether
 * anything is working on it.
 */

const LABEL = {
  'worker-download': 'download',
  'worker-chunk': 'chunk',
  'worker-embed': 'embed',
}

// The runtime's own vocabulary, plus `missing` for a service with no
// container and `unknown` when the runtime itself can't be reached.
const STATE = {
  running: { tone: 'accent', text: 'running' },
  exited: { tone: 'danger', text: 'stopped' },
  created: { tone: 'warn', text: 'not started' },
  paused: { tone: 'warn', text: 'paused' },
  missing: { tone: 'danger', text: 'not created' },
  unknown: { tone: 'neutral', text: 'unknown' },
}

function WorkerRow({ worker }) {
  const [open, setOpen] = useState(false)
  const logs = useWorkerLogs(worker.service, { enabled: open })
  const state = STATE[worker.state] || { tone: 'neutral', text: worker.state }

  return (
    <div className="border-b border-border last:border-0">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left transition-colors hover:bg-inset/60"
        aria-expanded={open}
      >
        <ChevronDown
          size={12}
          className={clsx('shrink-0 text-faint transition-transform', open && 'rotate-180')}
        />
        <span className="text-sm font-medium">{LABEL[worker.service] || worker.service}</span>
        <Badge tone={state.tone}>{state.text}</Badge>
        <span className="ml-auto truncate text-2xs text-faint" title={worker.status}>
          {worker.status}
        </span>
      </button>

      {open && (
        <div className="px-3 pb-3">
          {logs.isLoading ? (
            <Skeleton className="h-20 w-full" />
          ) : logs.error ? (
            <p className="text-2xs text-danger">{logs.error.message}</p>
          ) : !logs.data?.trim() ? (
            <p className="text-2xs text-faint">No output — this worker has produced no logs.</p>
          ) : (
            // Tail last, scrolled to the bottom: the newest lines are the
            // ones that explain what it is doing now.
            <pre className="max-h-64 overflow-auto rounded-lg bg-inset p-2.5 text-2xs leading-relaxed text-muted">
              {logs.data.trimEnd()}
            </pre>
          )}
        </div>
      )}
    </div>
  )
}

export default function WorkerStrip({ query }) {
  if (query.isLoading) return <Skeleton className="h-24 w-full rounded-xl" />
  if (query.error) return null

  const { workers = [], unavailable } = query.data || {}

  if (unavailable) {
    return (
      <Card className="px-3 py-2.5">
        <p className="text-2xs text-faint">
          Worker status unavailable — no container runtime reachable ({unavailable}). Normal if
          the backend is running outside compose.
        </p>
      </Card>
    )
  }

  const down = workers.filter((w) => w.state !== 'running')

  return (
    <div className="space-y-2">
      {down.length > 0 && (
        <p className="text-xs text-danger">
          {down.length === workers.length
            ? 'No pipeline workers are running — nothing below will move.'
            : `${down.length} worker${down.length !== 1 ? 's are' : ' is'} not running — that part of the pipeline is stalled.`}
        </p>
      )}
      <Card className="overflow-hidden p-0">
        {workers.map((w) => (
          <WorkerRow key={w.service} worker={w} />
        ))}
      </Card>
    </div>
  )
}
