import { motion } from 'framer-motion'

export default function TypingIndicator() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 4 }}
      transition={{ duration: 0.25 }}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 5,
        padding: '12px 18px',
        background: 'var(--bg-surface-1)',
        border: '1px solid var(--border)',
        borderRadius: '20px 20px 20px 4px',
        width: 'fit-content',
        marginBottom: 16,
      }}
    >
      {[0, 1, 2].map(i => (
        <motion.span
          key={i}
          animate={{ y: [0, -6, 0] }}
          transition={{
            duration: 0.7,
            delay: i * 0.15,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
          style={{
            display: 'block',
            width: 7,
            height: 7,
            borderRadius: '50%',
            background: 'var(--gold)',
            opacity: 0.85,
          }}
        />
      ))}
    </motion.div>
  )
}
