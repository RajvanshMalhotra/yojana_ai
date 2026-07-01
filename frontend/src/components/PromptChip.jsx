import { motion } from 'framer-motion'

export default function PromptChip({ text, onClick }) {
  return (
    <motion.button
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.97 }}
      onClick={() => onClick(text)}
      style={{
        background: 'transparent',
        border: '1px solid var(--border)',
        borderRadius: 20,
        padding: '8px 16px',
        fontFamily: 'Outfit, sans-serif',
        fontWeight: 400,
        fontSize: 13,
        color: 'var(--text-muted)',
        cursor: 'pointer',
        transition: 'border-color 0.2s, color 0.2s, background 0.2s',
      }}
      onMouseEnter={e => {
        e.currentTarget.style.borderColor = 'var(--gold-dim)'
        e.currentTarget.style.color = 'var(--text-primary)'
        e.currentTarget.style.background = 'rgba(232,160,69,0.06)'
      }}
      onMouseLeave={e => {
        e.currentTarget.style.borderColor = 'var(--border)'
        e.currentTarget.style.color = 'var(--text-muted)'
        e.currentTarget.style.background = 'transparent'
      }}
    >
      {text}
    </motion.button>
  )
}
