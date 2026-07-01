import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Navbar from '../components/Navbar'
import Sidebar from '../components/Sidebar'
import ChatBubble from '../components/ChatBubble'
import TypingIndicator from '../components/TypingIndicator'
import PromptChip from '../components/PromptChip'

const PROMPT_CHIPS = [
  'Schemes for farmers in Delhi',
  'I\'m a 25-year-old woman entrepreneur, what help can I get?',
  'PM Kisan eligibility',
]

function EmptyState({ onChipClick }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 20,
        padding: '40px 24px',
      }}
    >
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 8, repeat: Infinity, ease: 'linear' }}
        style={{
          fontFamily: '"Cormorant Garamond", Georgia, serif',
          fontSize: 56,
          color: 'var(--gold)',
          lineHeight: 1,
          userSelect: 'none',
        }}
      >
        ✦
      </motion.div>

      <div
        style={{
          fontFamily: 'Outfit, sans-serif',
          fontWeight: 300,
          fontSize: 16,
          color: 'var(--text-muted)',
          textAlign: 'center',
        }}
      >
        Find government schemes made for you
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', maxWidth: 480 }}>
        {PROMPT_CHIPS.map((chip, i) => (
          <PromptChip key={i} text={chip} onClick={onChipClick} />
        ))}
      </div>
    </motion.div>
  )
}

export default function Chat() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [chats, setChats] = useState([])
  const [activeChatId, setActiveChatId] = useState(null)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isTyping])

  async function sendMessage(text) {
    const userMsg = text || input.trim()
    if (!userMsg) return

    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: userMsg }])
    setIsTyping(true)

    if (messages.length === 0) {
      const newChat = { id: Date.now(), label: userMsg.slice(0, 40) }
      setChats(prev => [newChat, ...prev])
      setActiveChatId(newChat.id)
    }

    // Add an empty assistant bubble that we'll fill as tokens stream in
    setMessages(prev => [...prev, { role: 'assistant', content: '', schemes: [] }])

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg, history: messages }),
      })

      if (!res.ok) throw new Error('API error')

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buf += decoder.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop() // keep any incomplete line for the next chunk

        for (const line of lines) {
          if (!line.startsWith('data: ') || line === 'data: [DONE]') continue
          const evt = JSON.parse(line.slice(6))

          if (evt.type === 'schemes') {
            setMessages(prev => {
              const msgs = [...prev]
              msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], schemes: evt.schemes }
              return msgs
            })
            // keep isTyping true — spinner stays until first text token arrives
          } else if (evt.type === 'token') {
            setIsTyping(false)   // first token → hide spinner, text streams in
            setMessages(prev => {
              const msgs = [...prev]
              const last = msgs[msgs.length - 1]
              msgs[msgs.length - 1] = { ...last, content: last.content + evt.content }
              return msgs
            })
          }
        }
      }
    } catch {
      setMessages(prev => {
        const msgs = [...prev]
        msgs[msgs.length - 1] = {
          role: 'assistant',
          content: 'Sorry, something went wrong. Please make sure the backend is running.',
          schemes: [],
        }
        return msgs
      })
    } finally {
      setIsTyping(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  function handleNewChat() {
    setMessages([])
    setInput('')
    setActiveChatId(null)
    inputRef.current?.focus()
  }

  return (
    <div
      style={{
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        background: 'var(--bg-base)',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div
        style={{
          height: 60,
          flexShrink: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 24px',
          background: 'rgba(250,247,242,0.92)',
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)',
          borderBottom: '1px solid var(--border)',
          zIndex: 10,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ color: 'var(--gold)', fontSize: 18 }}>✦</span>
          <span
            style={{
              fontFamily: '"Cormorant Garamond", Georgia, serif',
              fontWeight: 600,
              fontSize: 18,
              color: 'var(--text-primary)',
            }}
          >
            YojanaAI
          </span>
        </div>

        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.97 }}
          onClick={handleNewChat}
          style={{
            background: 'transparent',
            border: '1px solid var(--border)',
            color: 'var(--text-muted)',
            borderRadius: 8,
            padding: '7px 16px',
            fontSize: 12,
            fontFamily: 'Outfit, sans-serif',
            fontWeight: 400,
            cursor: 'pointer',
            transition: 'border-color 0.2s, color 0.2s',
          }}
          onMouseEnter={e => {
            e.currentTarget.style.borderColor = 'var(--border-bright)'
            e.currentTarget.style.color = 'var(--text-primary)'
          }}
          onMouseLeave={e => {
            e.currentTarget.style.borderColor = 'var(--border)'
            e.currentTarget.style.color = 'var(--text-muted)'
          }}
        >
          New Chat
        </motion.button>
      </div>

      {/* Body */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        <Sidebar
          chats={chats}
          activeId={activeChatId}
          onSelect={id => setActiveChatId(id)}
          isOpen={true}
        />

        {/* Chat area */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {/* Messages */}
          <div
            style={{
              flex: 1,
              overflowY: 'auto',
              padding: '28px 32px',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            {messages.length === 0 ? (
              <EmptyState onChipClick={text => sendMessage(text)} />
            ) : (
              <>
                {messages.map((msg, i) => (
                  <ChatBubble key={i} message={msg} />
                ))}
                <AnimatePresence>
                  {isTyping && <TypingIndicator />}
                </AnimatePresence>
              </>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div
            style={{
              padding: '16px 24px',
              borderTop: '1px solid var(--border)',
              background: 'rgba(250,247,242,0.92)',
              backdropFilter: 'blur(20px)',
              WebkitBackdropFilter: 'blur(20px)',
            }}
          >
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 12,
                background: 'var(--bg-surface-1)',
                border: '1px solid var(--border)',
                borderRadius: 14,
                padding: '6px 6px 6px 18px',
                transition: 'border-color 0.2s, box-shadow 0.2s',
              }}
              onFocusCapture={e => {
                e.currentTarget.style.borderColor = 'var(--gold-dim)'
                e.currentTarget.style.boxShadow = '0 0 0 3px rgba(232,160,69,0.06)'
              }}
              onBlurCapture={e => {
                e.currentTarget.style.borderColor = 'var(--border)'
                e.currentTarget.style.boxShadow = 'none'
              }}
            >
              <input
                ref={inputRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about any scheme or describe your situation..."
                style={{
                  flex: 1,
                  background: 'transparent',
                  border: 'none',
                  outline: 'none',
                  fontFamily: 'Outfit, sans-serif',
                  fontWeight: 300,
                  fontSize: 14,
                  color: 'var(--text-primary)',
                  height: 40,
                }}
              />
              <motion.button
                whileHover={input.trim() ? { scale: 1.05 } : {}}
                whileTap={input.trim() ? { scale: 0.95 } : {}}
                onClick={() => sendMessage()}
                disabled={!input.trim() || isTyping}
                style={{
                  width: 40,
                  height: 40,
                  borderRadius: 10,
                  background: input.trim() ? 'var(--gold)' : 'var(--bg-surface-2)',
                  border: 'none',
                  color: input.trim() ? '#faf7f2' : 'var(--text-faint)',
                  fontSize: 18,
                  cursor: input.trim() && !isTyping ? 'pointer' : 'default',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  transition: 'background 0.2s, color 0.2s',
                  flexShrink: 0,
                }}
              >
                →
              </motion.button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
