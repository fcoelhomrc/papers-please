import { createContext, useContext, useEffect, useState } from 'react'

const STORAGE_KEY = 'pp-theme'
const ThemeContext = createContext(null)

function systemTheme() {
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

function applyTheme(mode) {
  const resolved = mode === 'system' ? systemTheme() : mode
  document.documentElement.classList.toggle('dark', resolved === 'dark')
  document.documentElement.classList.toggle('light', resolved !== 'dark')
  return resolved
}

export function ThemeProvider({ children }) {
  const [mode, setMode] = useState(() => localStorage.getItem(STORAGE_KEY) || 'system')
  const [resolved, setResolved] = useState(() => (mode === 'system' ? systemTheme() : mode))

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, mode)
    setResolved(applyTheme(mode))

    // Only follow the OS while the user is actually on "system" - once they
    // pick light or dark explicitly, an OS change must not override them.
    if (mode !== 'system') return
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const onChange = () => setResolved(applyTheme('system'))
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [mode])

  return (
    <ThemeContext.Provider value={{ mode, setMode, resolved }}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  const ctx = useContext(ThemeContext)
  if (!ctx) throw new Error('useTheme must be used inside ThemeProvider')
  return ctx
}
