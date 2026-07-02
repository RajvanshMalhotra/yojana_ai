import { motion } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import SchemeCard from './SchemeCard'

export default function ChatBubble({ message }) {
  const isUser = message.role === 'user'

  // Don't render placeholder bubbles that have no content yet
  if (!isUser && !message.content && (!message.schemes || message.schemes.length === 0)) {
    return null
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 16, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: isUser ? 'flex-end' : 'flex-start',
        marginBottom: 16,
      }}
    >
      <div
        style={{
          maxWidth: isUser ? '72%' : '80%',
          padding: '12px 18px',
          borderRadius: isUser ? '20px 20px 4px 20px' : '20px 20px 20px 4px',
          background: isUser
            ? 'linear-gradient(135deg, #e8a045, #c96a3a)'
            : 'var(--bg-surface-1)',
          border: isUser ? 'none' : '1px solid var(--border)',
          fontFamily: 'Outfit, sans-serif',
          fontWeight: 300,
          fontSize: 14,
          lineHeight: 1.7,
          color: isUser ? '#faf7f2' : 'var(--text-primary)',
        }}
      >
        {isUser ? (
          message.content
        ) : (
          <ReactMarkdown
            components={{
              p: ({ children }) => <span style={{ display: 'block', marginBottom: 4 }}>{children}</span>,
              strong: ({ children }) => <strong style={{ fontWeight: 600 }}>{children}</strong>,
              a: ({ href, children }) => (
                <a href={href} target="_blank" rel="noopener noreferrer"
                  style={{ color: 'var(--gold)', textDecoration: 'underline' }}>
                  {children}
                </a>
              ),
            }}
          >
            {message.content}
          </ReactMarkdown>
        )}
      </div>

      {/* Latency / stats badge */}
      {!isUser && message.stats && (
        <div style={{
          marginTop: 6,
          display: 'flex',
          flexWrap: 'wrap',
          gap: '4px 10px',
          fontFamily: 'Outfit, sans-serif',
          fontWeight: 300,
          fontSize: 11,
          color: 'var(--text-faint)',
        }}>
          <span title="Estimated prompt tokens">↑ {message.stats.prompt_tokens} tok</span>
          <span title="Completion tokens">↓ {message.stats.completion_tokens} tok</span>
          <span title="Time to first token">TTFT {message.stats.ttft}s</span>
          <span title="Retrieval time">ret {message.stats.retrieval_s}s</span>
          <span title="Generation time">gen {message.stats.gen_s}s</span>
          <span title="Total wall time" style={{ color: 'var(--text-muted)' }}>
            total {message.stats.total_s}s
          </span>
        </div>
      )}

      {/* Scheme cards below assistant message */}
      {!isUser && message.schemes && message.schemes.length > 0 && (
        <div style={{ width: '80%', marginTop: 4 }}>
          {message.schemes.map((scheme, i) => (
            <SchemeCard key={i} scheme={scheme} index={i} />
          ))}
        </div>
      )}
    </motion.div>
  )
}
