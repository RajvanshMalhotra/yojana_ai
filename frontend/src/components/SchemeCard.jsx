import { motion } from 'framer-motion'

const BADGE_COLORS = {
  farmer:      { color: '#e8a045', bg: 'rgba(232,160,69,0.12)' },
  women:       { color: '#b07cc6', bg: 'rgba(176,124,198,0.12)' },
  health:      { color: '#5aabdb', bg: 'rgba(90,171,219,0.12)' },
  youth:       { color: '#d4b93c', bg: 'rgba(212,185,60,0.12)' },
  education:   { color: '#56b87a', bg: 'rgba(86,184,122,0.12)' },
  income_support: { color: '#e8a045', bg: 'rgba(232,160,69,0.12)' },
  business:    { color: '#e07b5a', bg: 'rgba(224,123,90,0.12)' },
  default:     { color: '#6b6878', bg: 'rgba(107,104,120,0.12)' },
}

function Badge({ label }) {
  const key = label.toLowerCase().replace(/\s+/g, '_')
  const { color, bg } = BADGE_COLORS[key] || BADGE_COLORS.default

  return (
    <span
      style={{
        display: 'inline-block',
        background: bg,
        color,
        fontFamily: 'Outfit, sans-serif',
        fontWeight: 500,
        fontSize: 10,
        letterSpacing: '0.12em',
        textTransform: 'uppercase',
        padding: '3px 8px',
        borderRadius: 4,
      }}
    >
      {label}
    </span>
  )
}

export default function SchemeCard({ scheme, index = 0 }) {
  const categories = typeof scheme.categories === 'string'
    ? scheme.categories.split(',')
    : (scheme.categories || [])

  return (
    <motion.div
      initial={{ opacity: 0, x: 24 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.45, delay: index * 0.08, ease: [0.22, 1, 0.36, 1] }}
      style={{
        background: 'var(--bg-surface-2)',
        border: '1px solid var(--border)',
        borderRadius: 14,
        padding: '18px 20px',
        marginTop: 10,
        transition: 'border-color 0.2s, box-shadow 0.2s, transform 0.2s',
        cursor: 'default',
      }}
      onMouseEnter={e => {
        e.currentTarget.style.borderColor = 'var(--gold-dim)'
        e.currentTarget.style.boxShadow = '0 0 20px rgba(232,160,69,0.1)'
        e.currentTarget.style.transform = 'translateY(-2px)'
      }}
      onMouseLeave={e => {
        e.currentTarget.style.borderColor = 'var(--border)'
        e.currentTarget.style.boxShadow = 'none'
        e.currentTarget.style.transform = 'translateY(0)'
      }}
    >
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12, marginBottom: 10 }}>
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginBottom: 8 }}>
            {categories.slice(0, 2).map((cat, i) => (
              <Badge key={i} label={cat.trim()} />
            ))}
            {scheme.level && (
              <Badge label={scheme.level === 'delhi' ? 'Delhi' : 'Central'} />
            )}
          </div>
          <div
            style={{
              fontFamily: 'Outfit, sans-serif',
              fontWeight: 500,
              fontSize: 15,
              color: 'var(--text-primary)',
              lineHeight: 1.3,
            }}
          >
            {scheme.name}
          </div>
          {scheme.ministry && (
            <div
              style={{
                fontFamily: 'Outfit, sans-serif',
                fontWeight: 300,
                fontSize: 12,
                color: 'var(--text-muted)',
                marginTop: 3,
              }}
            >
              {scheme.ministry}
            </div>
          )}
        </div>
      </div>

      {/* Benefit line */}
      {scheme.benefits && (
        <div
          style={{
            fontFamily: 'Outfit, sans-serif',
            fontWeight: 400,
            fontSize: 13,
            color: 'var(--gold)',
            marginBottom: 8,
          }}
        >
          {scheme.benefits}
        </div>
      )}

      {/* Eligibility snippet */}
      {scheme.eligibility_snippet && (
        <p
          style={{
            fontFamily: 'Outfit, sans-serif',
            fontWeight: 300,
            fontSize: 13,
            color: 'var(--text-muted)',
            lineHeight: 1.65,
            marginBottom: 12,
            display: '-webkit-box',
            WebkitLineClamp: 2,
            WebkitBoxOrient: 'vertical',
            overflow: 'hidden',
          }}
        >
          {scheme.eligibility_snippet}
        </p>
      )}

      {/* Divider + link */}
      <div
        style={{
          borderTop: '1px solid var(--border)',
          paddingTop: 10,
          display: 'flex',
          justifyContent: 'flex-end',
        }}
      >
        {scheme.source_url ? (
          <a
            href={scheme.source_url}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              fontFamily: 'Outfit, sans-serif',
              fontWeight: 500,
              fontSize: 12,
              color: 'var(--gold)',
              textDecoration: 'none',
              letterSpacing: '0.03em',
              transition: 'opacity 0.2s',
            }}
            onMouseEnter={e => {
              e.currentTarget.style.textDecoration = 'underline'
              e.currentTarget.style.opacity = '0.8'
            }}
            onMouseLeave={e => {
              e.currentTarget.style.textDecoration = 'none'
              e.currentTarget.style.opacity = '1'
            }}
          >
            View Scheme →
          </a>
        ) : null}
      </div>
    </motion.div>
  )
}
