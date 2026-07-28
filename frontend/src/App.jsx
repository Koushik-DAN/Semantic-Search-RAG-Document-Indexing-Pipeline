import { useEffect, useState } from 'react'
import { getHealth } from './api.js'
import FileUpload from './components/FileUpload.jsx'
import Chat from './components/Chat.jsx'

export default function App() {
  const [health, setHealth] = useState(null)
  const [healthError, setHealthError] = useState(null)

  async function refreshHealth() {
    try {
      const result = await getHealth()
      setHealth(result)
      setHealthError(null)
    } catch (err) {
      setHealthError(err.message)
    }
  }

  useEffect(() => {
    refreshHealth()
  }, [])

  return (
    <div className="app">
      <header>
        <h1>RAG Document Pipeline</h1>
        <p className="status">
          {healthError && <span className="error">Backend unreachable: {healthError}</span>}
          {health && (
            <>
              Index: {health.num_chunks} chunk(s) · Ollama ({health.ollama_model}):{' '}
              {health.ollama_reachable ? 'reachable' : 'unreachable'}
            </>
          )}
        </p>
      </header>
      <main>
        <FileUpload onIndexed={refreshHealth} />
        <Chat />
      </main>
    </div>
  )
}
