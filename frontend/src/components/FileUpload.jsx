import { useState } from 'react'
import { fileToBase64, uploadFile } from '../api.js'

export default function FileUpload({ onIndexed }) {
  const [file, setFile] = useState(null)
  const [status, setStatus] = useState(null)
  const [busy, setBusy] = useState(false)

  async function handleUpload() {
    if (!file) return
    setBusy(true)
    setStatus(null)
    try {
      const contentBase64 = await fileToBase64(file)
      const result = await uploadFile(file.name, contentBase64)
      setStatus({
        ok: true,
        text: `Indexed ${result.num_chunks} chunk(s) from ${result.num_documents} document(s).`,
      })
      onIndexed?.(result)
    } catch (err) {
      setStatus({ ok: false, text: err.message })
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="panel">
      <h2>Upload a document</h2>
      <p className="hint">Supported types: .md, .txt, .pdf</p>
      <div className="row">
        <input
          type="file"
          accept=".md,.txt,.pdf"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <button onClick={handleUpload} disabled={!file || busy}>
          {busy ? 'Uploading…' : 'Upload & Index'}
        </button>
      </div>
      {status && <p className={status.ok ? 'message success' : 'message error'}>{status.text}</p>}
    </section>
  )
}
