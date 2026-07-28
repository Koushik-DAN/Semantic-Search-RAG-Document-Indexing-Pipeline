const BASE = '/api'

async function request(path, options) {
  const res = await fetch(`${BASE}${path}`, options)
  const body = await res.json()
  if (!res.ok) {
    throw new Error(body.detail || 'Request failed')
  }
  return body
}

export function getHealth() {
  return request('/health')
}

export function query(question, topK) {
  return request('/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: topK ?? null }),
  })
}

export function uploadFile(filename, contentBase64) {
  return request('/upload', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename, content_base64: contentBase64 }),
  })
}

export async function fileToBase64(file) {
  const buffer = await file.arrayBuffer()
  const bytes = new Uint8Array(buffer)
  let binary = ''
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i])
  }
  return btoa(binary)
}
