import { motion, AnimatePresence } from 'framer-motion'

export default function Sidebar({ chats, activeId, onSelect, isOpen, onToggle }) {
  return (
    <>
      {/* Desktop sidebar */}
      <div
        style={{
          width: 200,
          flexShrink: 0,
          borderRight: '1px solid var(--border)',
          background: 'var(--bg-surface-1)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            padding: '12px 16px',
            borderBottom: '1px solid var(--border)',
            fontFamily: 'Outfit, sans-serif',
            fontWeight: 500,
            fontSize: 11,
            letterSpacing: '0.12em',
            textTransform: 'uppercase',
            color: 'var(--text-faint)',
          }}
        >
          Previous Chats
        </div>

        <div style={{ flex: 1, overflow: 'auto', padding: '8px 0' }}>
          {chats.length === 0 ? (
            <div
              style={{
                padding: '16px',
                fontFamily: 'Outfit, sans-serif',
                fontWeight: 300,
                fontSize: 12,
                color: 'var(--text-faint)',
                fontStyle: 'italic',
              }}
            >
              No history yet
            </div>
          ) : (
            chats.map(chat => (
              <button
                key={chat.id}
                onClick={() => onSelect(chat.id)}
                style={{
                  width: '100%',
                  background: 'transparent',
                  border: 'none',
                  borderLeft: chat.id === activeId ? '2px solid var(--gold)' : '2px solid transparent',
                  padding: '9px 16px',
                  textAlign: 'left',
                  cursor: 'pointer',
                  fontFamily: 'Outfit, sans-serif',
                  fontWeight: 300,
                  fontSize: 12,
                  color: chat.id === activeId ? 'var(--text-primary)' : 'var(--text-muted)',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  transition: 'color 0.15s, background 0.15s',
                }}
                onMouseEnter={e => {
                  if (chat.id !== activeId) {
                    e.currentTarget.style.background = 'rgba(255,255,255,0.03)'
                    e.currentTarget.style.color = 'var(--text-primary)'
                  }
                }}
                onMouseLeave={e => {
                  if (chat.id !== activeId) {
                    e.currentTarget.style.background = 'transparent'
                    e.currentTarget.style.color = 'var(--text-muted)'
                  }
                }}
              >
                {chat.label}
              </button>
            ))
          )}
        </div>
      </div>
    </>
  )
}
