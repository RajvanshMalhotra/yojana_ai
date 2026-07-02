import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Navbar from '../components/Navbar'
import Sidebar from '../components/Sidebar'
import ChatBubble from '../components/ChatBubble'
import TypingIndicator from '../components/TypingIndicator'
import PromptChip from '../components/PromptChip'
import MicButton from '../components/MicButton'

const CHIPS = {
  en: [
    'Schemes for farmers in Delhi',
    "I'm a 25-year-old woman entrepreneur, what help can I get?",
    'PM Kisan eligibility',
  ],
  hi: [
    'दिल्ली के किसानों के लिए योजनाएं',
    'मैं 25 साल की महिला उद्यमी हूं, मुझे क्या सहायता मिल सकती है?',
    'पीएम किसान की पात्रता',
  ],
}

function LangButton({ label, sub, onClick }) {
  return (
    <motion.button
      whileHover={{ scale: 1.04 }}
      whileTap={{ scale: 0.97 }}
      onClick={onClick}
      style={{
        background: 'var(--bg-surface-1)',
        border: '1px solid var(--border)',
        borderRadius: 14,
        padding: '14px 28px',
        cursor: 'pointer',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 4,
        minWidth: 130,
        transition: 'border-color 0.2s, box-shadow 0.2s',
      }}
      onMouseEnter={e => {
        e.currentTarget.style.borderColor = 'var(--gold-dim)'
        e.currentTarget.style.boxShadow = '0 0 0 3px rgba(232,160,69,0.06)'
      }}
      onMouseLeave={e => {
        e.currentTarget.style.borderColor = 'var(--border)'
        e.currentTarget.style.boxShadow = 'none'
      }}
    >
      <span style={{ fontFamily: '"Cormorant Garamond", Georgia, serif', fontSize: 22, color: 'var(--text-primary)' }}>
        {label}
      </span>
      <span style={{ fontFamily: 'Outfit, sans-serif', fontWeight: 300, fontSize: 12, color: 'var(--text-muted)' }}>
        {sub}
      </span>
    </motion.button>
  )
}

function EmptyState({ lang, onLangSelect, onChipClick }) {
  const spinner = (
    <motion.div
      animate={{ rotate: 360 }}
      transition={{ duration: 8, repeat: Infinity, ease: 'linear' }}
      style={{ fontFamily: '"Cormorant Garamond", Georgia, serif', fontSize: 56, color: 'var(--gold)', lineHeight: 1, userSelect: 'none' }}
    >
      ✦
    </motion.div>
  )

  return (
    <motion.div
      key={lang || 'pick'}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4 }}
      style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 20, padding: '40px 24px' }}
    >
      {spinner}

      {!lang ? (
        <>
          <div style={{ fontFamily: 'Outfit, sans-serif', fontWeight: 300, fontSize: 16, color: 'var(--text-muted)', textAlign: 'center' }}>
            Choose your preferred language
          </div>
          <div style={{ display: 'flex', gap: 14 }}>
            <LangButton label="English" sub="English" onClick={() => onLangSelect('en')} />
            <LangButton label="हिंदी" sub="Hindi" onClick={() => onLangSelect('hi')} />
          </div>
        </>
      ) : (
        <>
          <div style={{ fontFamily: 'Outfit, sans-serif', fontWeight: 300, fontSize: 16, color: 'var(--text-muted)', textAlign: 'center' }}>
            {lang === 'hi' ? 'अपने लिए सरकारी योजनाएं खोजें' : 'Find government schemes made for you'}
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', maxWidth: 480 }}>
            {CHIPS[lang].map((chip, i) => (
              <PromptChip key={i} text={chip} onClick={onChipClick} />
            ))}
          </div>
        </>
      )}
    </motion.div>
  )
}

export default function Chat() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [chats, setChats] = useState([])       // [{id, label, messages}]
  const [activeChatId, setActiveChatId] = useState(null)
  const [lang, setLang] = useState(null)        // 'en' | 'hi' | null (not chosen yet)
  const [voiceAudioState, setVoiceAudioState] = useState(null) // 'playing' | 'paused' | null
  const messagesEndRef = useRef(null)
  const inputRef      = useRef(null)
  const voiceAudioRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isTyping])

  // Keep active chat's stored messages in sync
  useEffect(() => {
    if (!activeChatId || messages.length === 0) return
    setChats(prev => prev.map(c =>
      c.id === activeChatId ? { ...c, messages } : c
    ))
  }, [messages, activeChatId])

  async function sendMessage(text, { isVoice = false } = {}) {
    const userMsg = text || input.trim()
    if (!userMsg) return

    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: userMsg }])
    setIsTyping(true)

    if (messages.length === 0) {
      const newChat = { id: Date.now(), label: userMsg.slice(0, 40), messages: [] }
      setChats(prev => [newChat, ...prev])
      setActiveChatId(newChat.id)
    }

    // Add an empty assistant bubble that we'll fill as tokens stream in
    const startTime = Date.now()
    setMessages(prev => [...prev, { role: 'assistant', content: '', schemes: [] }])

    let fullAnswer = ''

    // Sentence-streaming TTS:
    //   First chunk fires at 80 chars so audio starts with minimal lag.
    //   Subsequent chunks fire at 200 chars so each plays long enough for the
    //   next Rumik request to resolve before it's needed (gap-free playback).
    let ttsBuf      = ''   // within-sentence token buffer
    let ttsAccum    = ''   // complete-sentence accumulator
    let ttsFirstSent = false  // lower threshold for the very first TTS call
    const ttsQueue = []
    let ttsPlaying = false

    // Each entry = { blob: Blob|null, promise: Promise<Blob> }
    // Blob is cached as soon as the fetch resolves so advanceTTS plays instantly with zero wait.
    function advanceTTS() {
      if (ttsQueue.length === 0) { ttsPlaying = false; setVoiceAudioState('paused'); return }
      ttsPlaying = true
      const entry = ttsQueue.shift()

      const playBlob = (blob) => {
        if (!blob) { advanceTTS(); return }
        if (voiceAudioRef.current) {
          voiceAudioRef.current.pause()
          if (voiceAudioRef.current._url) URL.revokeObjectURL(voiceAudioRef.current._url)
        }
        const url = URL.createObjectURL(blob)
        const audio = new Audio(url)
        audio._url = url
        voiceAudioRef.current = audio
        setVoiceAudioState('playing')
        audio.onended = () => advanceTTS()
        audio.onerror  = () => { URL.revokeObjectURL(url); advanceTTS() }
        audio.play()
      }

      // If blob already cached (resolved while previous sentence was playing) → zero gap
      if (entry.blob) {
        playBlob(entry.blob)
      } else {
        entry.promise.then(playBlob).catch(() => advanceTTS())
      }
    }

    function enqueueTTS(sentence) {
      if (!sentence.trim()) return
      const entry = { blob: null, promise: null }
      entry.promise = fetch('/api/voice/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: sentence.trim(), lang: lang || 'en' }),
      })
        .then(r => r.ok ? r.blob() : Promise.reject(r.status))
        .then(blob => { entry.blob = blob; return blob })
        .catch(() => null)
      ttsQueue.push(entry)
      if (!ttsPlaying) advanceTTS()
    }

    // Split text longer than 250 chars at word boundaries before enqueueing.
    // Prevents sending multi-thousand-char chunks to Rumik (causes timeouts).
    function enqueueTTSText(text) {
      const MAX = 250
      let t = text.trim()
      while (t.length > MAX) {
        let split = t.lastIndexOf(' ', MAX)
        if (split < 0) split = MAX
        enqueueTTS(t.slice(0, split).trim())
        t = t.slice(split).trim()
      }
      if (t) enqueueTTS(t)
    }

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg, history: messages, lang }),
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
          if (!line.startsWith('data: ')) continue
          if (line === 'data: [DONE]') {
            const latencyMs = Date.now() - startTime
            setMessages(prev => {
              const msgs = [...prev]
              msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], latencyMs }
              return msgs
            })
            continue
          }
          const evt = JSON.parse(line.slice(6))

          if (evt.type === 'schemes') {
            setMessages(prev => {
              const msgs = [...prev]
              msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], schemes: evt.schemes }
              return msgs
            })
          } else if (evt.type === 'token') {
            setIsTyping(false)
            fullAnswer += evt.content
            if (isVoice) {
              ttsBuf += evt.content
              // Drain all complete sentences from ttsBuf (while loop handles
              // multiple sentence-endings arriving in a single token batch)
              let sm
              while ((sm = /^([\s\S]*?[.!?])\s+([\s\S]*)$/.exec(ttsBuf))) {
                ttsAccum += (ttsAccum ? ' ' : '') + sm[1]
                ttsBuf = sm[2]
              }
              const threshold = ttsFirstSent ? 200 : 80
              if (ttsAccum.length >= threshold) {
                enqueueTTSText(ttsAccum)
                ttsAccum = ''
                ttsFirstSent = true
              } else if (ttsBuf.length > 300) {
                // Safety: no sentence boundary found but buffer is growing — flush raw
                enqueueTTSText(ttsBuf.trim())
                ttsBuf = ''
                ttsFirstSent = true
              }
            }
            setMessages(prev => {
              const msgs = [...prev]
              const last = msgs[msgs.length - 1]
              msgs[msgs.length - 1] = { ...last, content: last.content + evt.content }
              return msgs
            })
          } else if (evt.type === 'stats') {
            setMessages(prev => {
              const msgs = [...prev]
              msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], stats: evt }
              return msgs
            })
          }
        }
      }

      // Flush any remaining accumulated sentences + partial sentence, split into
      // ≤250-char chunks so Rumik doesn't timeout on a large final flush.
      if (isVoice) {
        const remaining = (ttsAccum + (ttsBuf.trim() ? ' ' + ttsBuf.trim() : '')).trim()
        if (remaining) enqueueTTSText(remaining)
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

  function handleSelectChat(id) {
    const chat = chats.find(c => c.id === id)
    if (chat) {
      setActiveChatId(id)
      setMessages(chat.messages)
      setIsTyping(false)
    }
  }

  function handleNewChat() {
    setMessages([])
    setInput('')
    setActiveChatId(null)
    setLang(null)
    inputRef.current?.focus()
  }

  function handleVoiceStopped() {
    setIsTyping(true)
  }

  function handleVoiceTranscript(transcript) {
    sendMessage(transcript, { isVoice: true })
  }

  function handleStopAudio() {
    voiceAudioRef.current?.pause()
    setVoiceAudioState('paused')
  }

  function handleReplayAudio() {
    const audio = voiceAudioRef.current
    if (!audio) return
    audio.currentTime = 0
    audio.play()
    setVoiceAudioState('playing')
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
          onSelect={handleSelectChat}
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
              <EmptyState lang={lang} onLangSelect={setLang} onChipClick={text => sendMessage(text)} />
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

          {/* Voice audio controls */}
          {voiceAudioState && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              padding: '6px 24px',
              background: 'var(--bg-surface-1)',
              borderTop: '1px solid var(--border)',
            }}>
              <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'Outfit, sans-serif', fontWeight: 300 }}>
                {voiceAudioState === 'playing' ? 'Playing response…' : 'Audio ready'}
              </span>
              {voiceAudioState === 'playing' ? (
                <button onClick={handleStopAudio} style={{
                  background: 'var(--bg-surface-2)', border: '1px solid var(--border)',
                  borderRadius: 8, padding: '3px 10px', cursor: 'pointer',
                  fontSize: 12, color: 'var(--text-primary)', fontFamily: 'Outfit, sans-serif',
                }}>■ Stop</button>
              ) : (
                <button onClick={handleReplayAudio} style={{
                  background: 'var(--bg-surface-2)', border: '1px solid var(--border)',
                  borderRadius: 8, padding: '3px 10px', cursor: 'pointer',
                  fontSize: 12, color: 'var(--text-primary)', fontFamily: 'Outfit, sans-serif',
                }}>↺ Replay</button>
              )}
            </div>
          )}

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
              <MicButton
                onTranscript={handleVoiceTranscript}
                onStopped={handleVoiceStopped}
                disabled={isTyping}
                lang={lang}
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
