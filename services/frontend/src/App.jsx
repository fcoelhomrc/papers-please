import * as Dialog from '@radix-ui/react-dialog'
import { MessageSquare, Menu } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Route, Routes } from 'react-router-dom'
import ChatPanel from './components/ChatPanel.jsx'
import SidebarContent from './components/Sidebar.jsx'
import { Button } from './components/ui.jsx'
import Document from './pages/Document.jsx'
import Documents from './pages/Documents.jsx'
import Fetch from './pages/Fetch.jsx'
import Queue from './pages/Queue.jsx'
import Search from './pages/Search.jsx'

export default function App() {
  const [navOpen, setNavOpen] = useState(false)
  const [chatOpen, setChatOpen] = useState(false)

  useEffect(() => {
    function onKey(e) {
      if (e.key === 'Escape') setChatOpen(false)
      // Cmd/Ctrl-J for the agent, mirroring the usual chat-drawer shortcut.
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'j') {
        e.preventDefault()
        setChatOpen((v) => !v)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  return (
    // h-dvh, not h-screen: on mobile browsers h-screen sits behind the URL
    // bar and the chat composer ends up under the fold.
    <div className="flex h-dvh w-full overflow-hidden bg-canvas text-ink">
      <aside className="hidden w-60 shrink-0 border-r border-border md:block">
        <SidebarContent
          onToggleChat={() => setChatOpen((v) => !v)}
          chatOpen={chatOpen}
        />
      </aside>

      {/* Below md the sidebar becomes a drawer - Radix Dialog for the focus
          trap and scroll lock rather than a hand-rolled overlay. */}
      <Dialog.Root open={navOpen} onOpenChange={setNavOpen}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 z-40 bg-black/40 animate-fade-in md:hidden" />
          <Dialog.Content className="fixed inset-y-0 left-0 z-50 w-64 border-r border-border shadow-panel animate-slide-in-left md:hidden">
            <Dialog.Title className="sr-only">Navigation</Dialog.Title>
            <SidebarContent onNavigate={() => setNavOpen(false)} />
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>

      <div className="flex min-w-0 flex-1 flex-col">
        {/* Mobile only. On desktop the nav is always visible and the agent
            toggle sits in the sidebar footer, so a top bar would just be an
            empty strip with a second "Agent" label next to the open rail. */}
        <header className="flex h-14 shrink-0 items-center gap-2 border-b border-border px-4 md:hidden">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setNavOpen(true)}
            aria-label="Open navigation"
          >
            <Menu size={17} />
          </Button>
          <span className="text-base font-semibold tracking-tight">Papers Please</span>
          <Button
            variant={chatOpen ? 'primary' : 'secondary'}
            size="sm"
            className="ml-auto"
            onClick={() => setChatOpen((v) => !v)}
          >
            <MessageSquare size={14} />
            Agent
          </Button>
        </header>

        <main className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6 lg:px-10">
            <Routes>
              <Route path="/" element={<Search />} />
              <Route path="/fetch" element={<Fetch />} />
              <Route path="/documents" element={<Documents />} />
              <Route path="/documents/:docId" element={<Document />} />
              <Route path="/queue" element={<Queue />} />
            </Routes>
          </div>
        </main>
      </div>

      {/* Backdrop only while the chat is an overlay (below lg); from lg up it
          is a docked rail and must not dim the page behind it. */}
      {chatOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/40 animate-fade-in lg:hidden"
          onClick={() => setChatOpen(false)}
        />
      )}
      <ChatPanel open={chatOpen} onClose={() => setChatOpen(false)} />
    </div>
  )
}
