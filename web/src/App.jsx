import { useEffect, useRef, useState } from 'react'
import { fetchInfo, streamChat } from './api'

// 识别答案里的引用标记，例如 [比亚迪 第57页] [贵州茅台 第16页]
const CITATION_RE = /\[([^[\]]+?)\s*第\s*(\d+)\s*页\]/g

/**
 * 把回答文本切成「普通文字」和「引用标签」两种片段。
 * 引用标签是可点击的按钮，点一下会高亮右边对应的来源卡片。
 */
function renderWithCitations(text, onCiteClick) {
  const nodes = []
  let lastIndex = 0
  let match
  CITATION_RE.lastIndex = 0

  while ((match = CITATION_RE.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index))
    }
    const [, company, page] = match
    nodes.push(
      <button
        key={`${match.index}-${page}`}
        type="button"
        className="cite"
        onClick={() => onCiteClick(Number(page))}
        title={`定位到 ${company} 第${page}页`}
      >
        {company} 第{page}页
      </button>,
    )
    lastIndex = match.index + match[0].length
  }

  if (lastIndex < text.length) nodes.push(text.slice(lastIndex))
  return nodes
}

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [info, setInfo] = useState(null)
  const [highlightPage, setHighlightPage] = useState(null)
  const bottomRef = useRef(null)

  // 进页面时拉一次知识库信息（有哪些公司）
  useEffect(() => {
    fetchInfo().then(setInfo).catch(() => setInfo(null))
  }, [])

  // 有新消息就滚到底部
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // 更新最后一条（也就是正在流式输出的那条）助手消息
  const patchLast = (patch) => {
    setMessages((prev) => {
      if (prev.length === 0) return prev
      const next = [...prev]
      const i = next.length - 1
      next[i] = { ...next[i], ...patch }
      return next
    })
  }

  const handleSend = async () => {
    const question = input.trim()
    if (!question || loading) return

    setInput('')
    setLoading(true)
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: question },
      { role: 'assistant', content: '', sources: [], streaming: true },
    ])

    try {
      await streamChat({
        question,
        onSources: (sources) => patchLast({ sources }),
        onDelta: (text) =>
          setMessages((prev) => {
            const next = [...prev]
            const i = next.length - 1
            next[i] = { ...next[i], content: next[i].content + text }
            return next
          }),
        onError: (message) =>
          patchLast({ content: `出错了：${message}`, streaming: false }),
        onDone: () => patchLast({ streaming: false }),
      })
    } catch (err) {
      patchLast({ content: `请求失败：${err.message}`, streaming: false })
    } finally {
      setLoading(false)
      patchLast({ streaming: false })
    }
  }

  const handleKeyDown = (e) => {
    // 回车发送，Shift+回车换行
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const fillQuestion = (company) => {
    setInput(`${company} 2024 年的营业收入是多少？`)
  }

  return (
    <div className="page">
      <header className="header">
        <div>
          <h1>上市公司年报 RAG 问答</h1>
          <p className="subtitle">
            基于真实年报的检索增强问答，每个结论都标注出处页码
          </p>
        </div>
        {info && (
          <div className="info">
            <span className="info-item">{info.chunks} 个文本块</span>
            <span className="info-item">{info.llm_model}</span>
          </div>
        )}
      </header>

      {info?.companies?.length > 0 && (
        <div className="companies">
          <span className="companies-label">知识库已收录：</span>
          {info.companies.map((c) => (
            <button key={c} type="button" className="chip" onClick={() => fillQuestion(c)}>
              {c}
            </button>
          ))}
        </div>
      )}

      <main className="chat">
        {messages.length === 0 && (
          <div className="empty">
            <p>试着问一个能从年报里查到的问题：</p>
            <ul>
              <li>比亚迪 2024 年的营业收入是多少？</li>
              <li>招商银行的不良贷款率是多少？</li>
              <li>贵州茅台的毛利率是多少？</li>
            </ul>
          </div>
        )}

        {messages.map((msg, index) => (
          <div key={index} className={`msg msg-${msg.role}`}>
            <div className="bubble">
              {msg.role === 'assistant' ? (
                <>
                  <div className="answer">
                    {renderWithCitations(msg.content, setHighlightPage)}
                    {msg.streaming && <span className="cursor" />}
                  </div>

                  {msg.sources?.length > 0 && (
                    <div className="sources">
                      <div className="sources-title">参考来源</div>
                      {msg.sources.map((s, i) => (
                        <div
                          key={i}
                          className={`source ${
                            highlightPage === s.page ? 'source-active' : ''
                          }`}
                        >
                          <div className="source-head">
                            <span className="source-company">{s.company}</span>
                            <span className="source-page">第 {s.page} 页</span>
                            <span className="source-distance">
                              距离 {s.distance}
                            </span>
                          </div>
                          <div className="source-preview">{s.preview}…</div>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              ) : (
                <div className="question">{msg.content}</div>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </main>

      <footer className="composer">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入问题，回车发送（Shift+回车换行）"
          rows={2}
          disabled={loading}
        />
        <button type="button" onClick={handleSend} disabled={loading || !input.trim()}>
          {loading ? '生成中…' : '发送'}
        </button>
      </footer>
    </div>
  )
}

export default App
