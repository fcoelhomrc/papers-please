import { Monitor, Moon, Sun } from 'lucide-react'
import clsx from 'clsx'
import { useTheme } from '../lib/theme.jsx'

const MODES = [
  { value: 'light', icon: Sun, label: 'Light' },
  { value: 'dark', icon: Moon, label: 'Dark' },
  { value: 'system', icon: Monitor, label: 'System' },
]

export default function ThemeToggle() {
  const { mode, setMode } = useTheme()
  return (
    <div className="inline-flex rounded-lg border border-border bg-surface p-0.5">
      {MODES.map(({ value, icon: Icon, label }) => (
        <button
          key={value}
          onClick={() => setMode(value)}
          aria-label={`${label} theme`}
          aria-pressed={mode === value}
          title={label}
          className={clsx(
            'rounded-md p-1.5 transition-colors',
            mode === value ? 'bg-inset text-ink' : 'text-faint hover:text-ink',
          )}
        >
          <Icon size={14} strokeWidth={2} />
        </button>
      ))}
    </div>
  )
}
