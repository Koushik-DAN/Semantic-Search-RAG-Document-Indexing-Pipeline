import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { query } from '../api.js'

export default function Chat() {
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState([])
  const [busy, setBusy] = useState(false)

  async function handleSend() {
    const q = question.trim()
    if (!q || busy) return
    setBusy(true)
    setQuestion('')
    setMessages((prev) => [...prev, { question: q, answer: null, sources: [], error: null }])

    try {
      const result = await query(q)
      setMessages((prev) => {
        const next = [...prev]
        next[next.length - 1] = { question: q, answer: result.answer, sources: result.sources, error: null }
        return next
      })
    } catch (err) {
      setMessages((prev) => {
        const next = [...prev]
        next[next.length - 1] = { question: q, answer: null, sources: [], error: err.message }
        return next
      })
    } finally {
      setBusy(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <section className="panel">
      <h2>Chat</h2>
      <div className="messages">
        {messages.length === 0 && <p className="hint">Ask a question about your indexed documents.</p>}
        {messages.map((m, i) => (
          <div key={i} className="turn">
            <div className="bubble question">{m.question}</div>
            {m.error && <div className="bubble error">{m.error}</div>}
            {m.answer && (
              <div className="bubble answer">
                <div className="markdown">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.answer}</ReactMarkdown>
                </div>
                {m.sources.length > 0 && (
                  <ul className="sources">
                    {m.sources.map((s) => (
                      <li key={s.chunk_id}>
                        [{s.score.toFixed(3)}] {s.source} ({s.chunk_id})
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
            {!m.answer && !m.error && <div className="bubble answer pending">Thinking…</div>}
          </div>
        ))}
      </div>
      <div className="row">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question…"
          disabled={busy}
        />
        <button onClick={handleSend} disabled={busy || !question.trim()}>
          Send
        </button>
      </div>
    </section>
  )
}
