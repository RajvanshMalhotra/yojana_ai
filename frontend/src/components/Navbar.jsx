import { Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'

export default function Navbar() {
  const { pathname } = useLocation()
  const isChat = pathname === '/chat'

  return (
    <motion.nav
      initial={{ y: -64, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        zIndex: 100,
        height: 60,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 32px',
        background: 'rgba(250,247,242,0.88)',
        backdropFilter: 'blur(20px)',
        WebkitBackdropFilter: 'blur(20px)',
        borderBottom: '1px solid var(--border)',
      }}
    >
      <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: 10, textDecoration: 'none' }}>
        <span style={{ color: 'var(--gold)', fontSize: 20, lineHeight: 1 }}>✦</span>
        <span
          style={{
            fontFamily: '"Cormorant Garamond", Georgia, serif',
            fontWeight: 600,
            fontSize: 20,
            color: 'var(--text-primary)',
            letterSpacing: '-0.01em',
          }}
        >
          YojanaAI
        </span>
      </Link>

      <Link to="/chat">
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.97 }}
          style={{
            background: 'transparent',
            border: '1px solid var(--gold)',
            color: 'var(--gold)',
            borderRadius: 8,
            padding: '8px 20px',
            fontSize: 13,
            fontFamily: 'Outfit, sans-serif',
            fontWeight: 500,
            letterSpacing: '0.04em',
            cursor: 'pointer',
            transition: 'background 0.2s, color 0.2s',
          }}
          onMouseEnter={e => {
            e.currentTarget.style.background = 'var(--gold)'
            e.currentTarget.style.color = '#faf7f2'
          }}
          onMouseLeave={e => {
            e.currentTarget.style.background = 'transparent'
            e.currentTarget.style.color = 'var(--gold)'
          }}
        >
          {isChat ? 'New Chat' : 'Open App →'}
        </motion.button>
      </Link>
    </motion.nav>
  )
}
