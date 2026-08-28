import { Route, Routes } from 'react-router-dom'
import Chat from './components/Chat.jsx'
import Sidebar from './components/Sidebar.jsx'
import Documents from './pages/Documents.jsx'
import Fetch from './pages/Fetch.jsx'
import Queue from './pages/Queue.jsx'
import Search from './pages/Search.jsx'

export default function App() {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-canvas text-ink">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-5xl mx-auto px-8 py-8">
          <Routes>
            <Route path="/" element={<Search />} />
            <Route path="/fetch" element={<Fetch />} />
            <Route path="/documents" element={<Documents />} />
            <Route path="/queue" element={<Queue />} />
          </Routes>
        </div>
      </main>
      <Chat />
    </div>
  )
}
