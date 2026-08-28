import { Download, FileStack, ListChecks, Search } from 'lucide-react'
import { NavLink } from 'react-router-dom'

const NAV = [
  { to: '/', label: 'Search', icon: Search, end: true },
  { to: '/fetch', label: 'Fetch', icon: Download },
  { to: '/documents', label: 'Documents', icon: FileStack },
  { to: '/queue', label: 'Queue', icon: ListChecks },
]

export default function Sidebar() {
  return (
    <aside className="w-56 shrink-0 border-r border-border bg-surface flex flex-col">
      <div className="h-14 flex items-center px-5 border-b border-border">
        <span className="font-semibold tracking-tight text-[15px]">Papers Please</span>
      </div>
      <nav className="flex-1 px-3 py-4 space-y-0.5">
        {NAV.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              `flex items-center gap-2.5 rounded-lg px-3 py-2 text-[13.5px] font-medium transition-colors ${
                isActive
                  ? 'bg-canvas text-ink'
                  : 'text-muted hover:text-ink hover:bg-canvas/60'
              }`
            }
          >
            <Icon size={16} strokeWidth={2} />
            {label}
          </NavLink>
        ))}
      </nav>
    </aside>
  )
}
