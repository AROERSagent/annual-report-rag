/**
 * 与后端通信的封装。
 *
 * 这里有一个很实用、面试也常问的知识点：
 *
 *   为什么不用浏览器自带的 EventSource 来接收 SSE？
 *
 *   因为 EventSource 只能发起 GET 请求，而我们的问答接口需要 POST
 *   提交 JSON（问题可能很长，放 URL 里不合适）。所以改用
 *   fetch + ReadableStream 手动解析 SSE 文本流。
 *
 * SSE 的报文格式很简单，就是纯文本：
 *      event: sources\n
 *      data: [{...}]\n
 *      \n                     ← 空行表示一个事件结束
 */

// 开发时留空即可：Vite 的 proxy 会把 /api 转发到 8000 端口。
// 如果要直连后端（不走代理），在 web/.env 里写 VITE_API_BASE=http://127.0.0.1:8000
const API_BASE = import.meta.env.VITE_API_BASE || ''

export async function fetchInfo() {
  const resp = await fetch(`${API_BASE}/api/info`)
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
  return resp.json()
}

/**
 * 发起流式问答。
 *
 * @param {object}   opts
 * @param {string}   opts.question  用户问题
 * @param {number}   [opts.topK]    检索条数
 * @param {Function} opts.onSources 收到出处列表时调用
 * @param {Function} opts.onDelta   收到一段新文本时调用（会被调用很多次）
 * @param {Function} opts.onError   出错时调用
 * @param {Function} opts.onDone    结束时调用
 */
export async function streamChat({ question, topK, onSources, onDelta, onError, onDone }) {
  const resp = await fetch(`${API_BASE}/api/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: topK ?? null }),
  })

  if (!resp.ok || !resp.body) {
    throw new Error(`HTTP ${resp.status}`)
  }

  const reader = resp.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  // 处理单个事件块
  const handleEvent = (raw) => {
    let event = 'message'
    const dataLines = []

    for (const line of raw.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim()
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
    }

    const data = dataLines.join('\n')
    if (!data) return

    let payload
    try {
      payload = JSON.parse(data)
    } catch {
      payload = data
    }

    if (event === 'sources') onSources?.(payload)
    else if (event === 'delta') onDelta?.(payload.text ?? '')
    else if (event === 'error') onError?.(payload.message ?? '未知错误')
  }

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    buffer = buffer.replace(/\r\n/g, '\n') // 兼容不同服务器的换行

    // 以空行切分事件；最后一段可能不完整，留到下一轮再处理
    let sep
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      handleEvent(buffer.slice(0, sep))
      buffer = buffer.slice(sep + 2)
    }
  }

  onDone?.()
}
