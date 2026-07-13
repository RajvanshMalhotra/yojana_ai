import { useState, useRef, useCallback, useEffect } from 'react'
import { motion } from 'framer-motion'

export default function MicButton({ onTranscript, onStopped, disabled, lang, large = false }) {
  const [recording,  setRecording]  = useState(false)
  const [processing, setProcessing] = useState(false)
  const recorderRef = useRef(null)
  const chunksRef   = useRef([])

  const start = useCallback(async () => {
    if (disabled || processing || recording) return
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      chunksRef.current = []
      const rec = new MediaRecorder(stream)
      rec.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data) }

      rec.onstop = async () => {
        stream.getTracks().forEach(t => t.stop())
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        setProcessing(true)
        try {
          const langParam = lang ? `?lang=${lang}` : ''
          const res = await fetch(`/api/voice/stt${langParam}`, {
            method: 'POST',
            headers: { 'Content-Type': 'audio/webm' },
            body: blob,
          })
          if (!res.ok) throw new Error(`stt ${res.status}`)
          const { transcript } = await res.json()
          if (transcript) onTranscript?.(transcript)
        } catch (err) {
          console.error('stt error:', err)
        } finally {
          setProcessing(false)
        }
      }

      recorderRef.current = rec
      rec.start()
      setRecording(true)
    } catch (err) {
      console.error('mic access error:', err)
    }
  }, [disabled, processing, recording, lang, onTranscript])

  const stop = useCallback(() => {
    const rec = recorderRef.current
    if (rec && rec.state !== 'inactive') {
      rec.stop()
      recorderRef.current = null
    }
    setRecording(false)
    onStopped?.()
  }, [onStopped])

  useEffect(() => () => { recorderRef.current?.stop() }, [])

  const handleClick = useCallback(() => {
    if (disabled || processing) return
    recording ? stop() : start()
  }, [disabled, processing, recording, start, stop])

  const sz = large ? 64 : 40
  const radius = large ? 20 : 10
  const iconSz = large ? 24 : 15

  return (
    <motion.button
      onClick={handleClick}
      disabled={disabled || processing}
      animate={recording ? { scale: [1, 1.08, 1] } : {}}
      transition={recording ? { repeat: Infinity, duration: 0.9 } : {}}
      title={recording ? 'Click to stop' : processing ? 'Transcribing…' : 'Click to speak'}
      style={{
        width: sz,
        height: sz,
        borderRadius: radius,
        border: large && !recording ? '1px solid var(--border)' : 'none',
        background: recording ? 'var(--gold)' : processing ? 'var(--bg-surface-1)' : 'var(--bg-surface-2)',
        color: recording ? '#faf7f2' : processing ? 'var(--gold-dim)' : 'var(--text-muted)',
        cursor: disabled || processing ? 'default' : 'pointer',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: iconSz,
        flexShrink: 0,
        transition: 'background 0.2s, color 0.2s',
        boxShadow: large && recording ? '0 0 0 6px rgba(232,160,69,0.15)' : 'none',
      }}
    >
      {processing ? '…' : recording ? '■' : '🎙'}
    </motion.button>
  )
}
