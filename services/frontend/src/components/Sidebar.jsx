import clsx from 'clsx'
import { Download, FileStack, Library, ListChecks, MessageSquare, Search } from 'lucide-react'
import { NavLink } from 'react-router-dom'
import { useStatus } from '../hooks/queries'
import ThemeToggle from './ThemeToggle.jsx'
import { Button } from './ui.jsx'

const NAV = [
  { to: '/', label: 'Search', icon: Search, end: true },
  { to: '/fetch', label: 'Fetch', icon: Download },
  { to: '/documents', label: 'Documents', icon: FileStack },
  { to: '/queue', label: 'Queue', icon: ListChecks },
]

/* Ambient pipeline health, so you don't have to open Queue to notice a
   backlog. Shares the ['status'] query key with the Queue page - TanStack
   Query dedupes them into one request rather than two pollers. */
function PipelinePulse() {
  const { data } = useStatus({ refetchInterval: 30_000 })
  if (!data) return null

  const backlog =
    (data.pending_download ?? 0) +
    (data.objects_by_status?.pending ?? 0) +
    (data.chunks_pending_embed ?? 0)
  const failed = data.objects_by_status?.failed ?? 0

  const tone = failed > 0 ? 'bg-danger' : backlog > 0 ? 'bg-warn' : 'bg-success'
  const label =
    failed > 0
      ? `${failed} failed`
      : backlog > 0
        ? `${backlog} in pipeline`
        : 'Pipeline idle'

  return (
    <NavLink
      to="/queue"
      className="flex items-center gap-2 rounded-lg px-3 py-2 text-xs text-muted transition-colors hover:bg-inset hover:text-ink"
    >
      <span className={clsx('h-1.5 w-1.5 shrink-0 rounded-full', tone)} />
      <span className="truncate">{label}</span>
      <span className="ml-auto tabular-nums text-faint">{data.documents_total}</span>
    </NavLink>
  )
}

export default function SidebarContent({ onNavigate, onToggleChat, chatOpen }) {
  return (
    <div className="flex h-full flex-col bg-surface">
      <div className="flex h-14 shrink-0 items-center gap-2 px-5">
        <Library size={17} className="text-accent" strokeWidth={2} />
        <span className="text-lg font-semibold tracking-tight">Papers Please</span>
      </div>

      <nav className="flex-1 space-y-0.5 px-3 py-2">
        {NAV.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            onClick={onNavigate}
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                isActive
                  ? 'bg-accent-soft text-accent'
                  : 'text-muted hover:bg-inset hover:text-ink',
              )
            }
          >
            <Icon size={16} strokeWidth={2} />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="shrink-0 space-y-3 border-t border-border p-3">
        <PipelinePulse />
        {/* Desktop only - below md the agent lives in the mobile top bar, so
            the drawer doesn't render this. */}
        {onToggleChat && (
          <Button
            variant={chatOpen ? 'primary' : 'secondary'}
            onClick={onToggleChat}
            className="w-full justify-center"
          >
            <MessageSquare size={14} />
            Agent
            <kbd className="ml-1 font-sans text-2xs opacity-60">⌘J</kbd>
          </Button>
        )}
        <div className="flex items-center justify-between px-1">
          <span className="text-2xs text-faint">Theme</span>
          <ThemeToggle />
        </div>
      </div>
    </div>
  )
}
