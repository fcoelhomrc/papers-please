/* Shared primitives. The previous frontend repeated the same long class
 * strings inline on every page (four different spellings of the same input,
 * three of the same error banner), which is why the visual language drifted.
 * Everything styled lives here or in index.css.
 */
import clsx from 'clsx'
import { AlertCircle, CheckCircle2, Inbox, Loader2 } from 'lucide-react'
import { forwardRef } from 'react'

/* ---------- Button ---------- */

const BUTTON_VARIANTS = {
  primary: 'bg-accent text-accent-ink hover:opacity-90 shadow-subtle',
  secondary: 'bg-surface text-ink border border-border hover:border-border-strong hover:bg-inset',
  ghost: 'text-muted hover:text-ink hover:bg-inset',
  danger: 'bg-danger text-white hover:opacity-90',
}

const BUTTON_SIZES = {
  sm: 'h-7 px-2.5 text-xs gap-1.5 rounded-md',
  md: 'h-9 px-3.5 text-sm gap-2 rounded-lg',
  lg: 'h-10 px-5 text-sm gap-2 rounded-lg',
  icon: 'h-8 w-8 rounded-lg justify-center',
}

/* Exported so elements that must look like a button but cannot be one - an
   <a href> for a download, which <button> can't express - share exactly one
   definition instead of a hand-copied class string. */
export function buttonClass({ variant = 'secondary', size = 'md', className } = {}) {
  return clsx(
    'inline-flex shrink-0 select-none items-center font-medium transition-all',
    'disabled:pointer-events-none disabled:opacity-40',
    BUTTON_VARIANTS[variant],
    BUTTON_SIZES[size],
    className,
  )
}

export const Button = forwardRef(function Button(
  { variant = 'secondary', size = 'md', loading, className, children, disabled, ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      disabled={disabled || loading}
      className={buttonClass({ variant, size, className })}
      {...props}
    >
      {loading && <Loader2 size={14} className="animate-spin" />}
      {children}
    </button>
  )
})

/* ---------- Form controls ---------- */

// Deliberately carries no width: clsx concatenates, it does not resolve
// conflicts, so a `w-full` baked in here would beat a caller's `w-auto`
// whenever Tailwind happens to emit w-full later in the stylesheet (it does).
// Width is the caller's business; only Input/Textarea opt into w-full, where
// it is always what's wanted.
const FIELD_BASE =
  'rounded-lg border border-border bg-surface text-ink placeholder:text-faint transition-colors hover:border-border-strong disabled:opacity-50'

export const Input = forwardRef(function Input({ className, ...props }, ref) {
  return (
    <input ref={ref} className={clsx(FIELD_BASE, 'h-9 w-full px-3 text-sm', className)} {...props} />
  )
})

export const Textarea = forwardRef(function Textarea({ className, ...props }, ref) {
  return (
    <textarea
      ref={ref}
      className={clsx(FIELD_BASE, 'w-full resize-none px-3 py-2 text-sm', className)}
      {...props}
    />
  )
})

export function Select({ className, children, ...props }) {
  return (
    <select className={clsx(FIELD_BASE, 'h-9 cursor-pointer px-2.5 text-sm', className)} {...props}>
      {children}
    </select>
  )
}

export function Field({ label, hint, children }) {
  return (
    <label className="block space-y-1.5">
      <span className="block text-xs font-medium text-muted">{label}</span>
      {children}
      {hint && <span className="block text-2xs text-faint">{hint}</span>}
    </label>
  )
}

export function Checkbox({ label, className, ...props }) {
  return (
    <label
      className={clsx(
        'inline-flex cursor-pointer select-none items-center gap-2 text-sm text-muted hover:text-ink',
        className,
      )}
    >
      <input
        type="checkbox"
        className="h-3.5 w-3.5 cursor-pointer accent-accent"
        {...props}
      />
      {label}
    </label>
  )
}

/* Replaces the Radix Tabs that were being used purely as a two-option
   toggle - no tab panels were ever rendered, so this is the honest control. */
export function SegmentedControl({ value, onChange, options }) {
  return (
    <div role="tablist" className="inline-flex rounded-lg border border-border bg-surface p-0.5">
      {options.map((opt) => (
        <button
          key={opt.value}
          role="tab"
          aria-selected={value === opt.value}
          onClick={() => onChange(opt.value)}
          className={clsx(
            'rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
            value === opt.value
              ? 'bg-inset text-ink shadow-subtle'
              : 'text-muted hover:text-ink',
          )}
        >
          {opt.label}
        </button>
      ))}
    </div>
  )
}

/* ---------- Surfaces ---------- */

export function Card({ className, children, ...props }) {
  return (
    <div
      className={clsx('rounded-xl border border-border bg-surface shadow-card', className)}
      {...props}
    >
      {children}
    </div>
  )
}

export function PageHeader({ title, description, actions }) {
  return (
    <header className="flex items-start justify-between gap-4">
      <div className="min-w-0">
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        {description && <p className="mt-1 max-w-prose text-sm text-muted">{description}</p>}
      </div>
      {actions}
    </header>
  )
}

export function Badge({ tone = 'neutral', className, children }) {
  const tones = {
    neutral: 'bg-inset text-muted',
    accent: 'bg-accent-soft text-accent',
    success: 'bg-success-soft text-success',
    warn: 'bg-warn-soft text-warn',
    danger: 'bg-danger-soft text-danger',
  }
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-2xs font-medium',
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  )
}

/* ---------- States ---------- */

export function Callout({ tone = 'neutral', className, children }) {
  const tones = {
    neutral: 'border-border bg-surface text-muted',
    success: 'border-success/25 bg-success-soft text-success',
    warn: 'border-warn/25 bg-warn-soft text-warn',
    danger: 'border-danger/25 bg-danger-soft text-danger',
  }
  const Icon = tone === 'danger' ? AlertCircle : tone === 'success' ? CheckCircle2 : AlertCircle
  return (
    <div
      role={tone === 'danger' ? 'alert' : undefined}
      className={clsx(
        'flex items-start gap-2.5 rounded-lg border px-3.5 py-2.5 text-sm animate-fade-in',
        tones[tone],
        className,
      )}
    >
      <Icon size={15} className="mt-px shrink-0" />
      <div className="min-w-0">{children}</div>
    </div>
  )
}

export function ErrorState({ error, className }) {
  if (!error) return null
  return (
    <Callout tone="danger" className={className}>
      {error.message || String(error)}
    </Callout>
  )
}

export function EmptyState({ icon: Icon = Inbox, title, description, action }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-border px-6 py-14 text-center animate-fade-in">
      <Icon size={22} className="text-faint" strokeWidth={1.75} />
      <p className="mt-3 text-sm font-medium text-ink">{title}</p>
      {description && <p className="mt-1 max-w-xs text-sm text-muted">{description}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

export function Skeleton({ className }) {
  return <div className={clsx('skeleton', className)} />
}
