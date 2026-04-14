import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'
import Documents from './pages/Documents'
import Fetch from './pages/Fetch'
import Search from './pages/Search'

function Header() {
  const cls = ({ isActive }) =>
    `text-sm font-medium transition-colors ${
      isActive ? 'text-white' : 'text-slate-400 hover:text-slate-200'
    }`

  return (
    <header className="bg-slate-950 border-b border-slate-800">
      <div className="max-w-5xl mx-auto px-6 h-14 flex items-center gap-8">
        <span className="text-white font-semibold tracking-tight select-none">
          Papers Please
        </span>
        <nav className="flex gap-6">
          <NavLink to="/" end className={cls}>Search</NavLink>
          <NavLink to="/fetch" className={cls}>Fetch</NavLink>
          <NavLink to="/documents" className={cls}>Documents</NavLink>
        </nav>
      </div>
    </header>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50 text-gray-900">
        <Header />
        <main className="max-w-5xl mx-auto px-6 py-10">
          <Routes>
            <Route path="/" element={<Search />} />
            <Route path="/fetch" element={<Fetch />} />
            <Route path="/documents" element={<Documents />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
