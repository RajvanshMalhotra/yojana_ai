import { motion } from 'framer-motion'
import useInView from '../hooks/useInView'

const FEATURES = [
  {
    icon: '◎',
    title: 'Natural language queries',
    desc: 'Ask in plain Hindi or English. No filters, no forms.',
    span: 'row',
  },
  {
    icon: '⌖',
    title: 'Searches 1,200+ schemes',
    desc: 'Comprehensive coverage of central government welfare programmes.',
    span: null,
  },
  {
    icon: '◈',
    title: 'Hybrid retrieval',
    desc: 'Semantic search + keyword ranking for precision results.',
    span: null,
  },
  {
    icon: '◉',
    title: 'Central schemes',
    desc: 'PM Kisan, Ayushman Bharat, Mudra, and hundreds more.',
    span: null,
  },
  {
    icon: '◐',
    title: 'Delhi state schemes',
    desc: 'Ladli, pension schemes, Mukhyamantri Tirth Yatra, and more.',
    span: null,
  },
  {
    icon: '◍',
    title: 'Instant answers with sources',
    desc: 'Every response cites the official scheme URL so you can apply directly.',
    span: 'full',
  },
]

function Card({ feature, index, inView }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 50 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.6, delay: index * 0.1, ease: [0.22, 1, 0.36, 1] }}
      style={{
        background: 'var(--bg-surface-1)',
        border: '1px solid var(--border)',
        borderRadius: 16,
        padding: '28px 28px 24px',
        position: 'relative',
        overflow: 'hidden',
        cursor: 'default',
        transition: 'border-color 0.3s, box-shadow 0.3s',
        gridRow: feature.span === 'row' ? 'span 2' : 'span 1',
        gridColumn: feature.span === 'full' ? '1 / -1' : 'span 1',
        display: 'flex',
        flexDirection: 'column',
        gap: feature.span === 'full' ? 0 : 16,
      }}
      onMouseEnter={e => {
        e.currentTarget.style.borderColor = 'var(--border-bright)'
        e.currentTarget.style.boxShadow = '0 0 32px rgba(232,160,69,0.08)'
        const shimmer = e.currentTarget.querySelector('.shimmer')
        if (shimmer) shimmer.style.animation = 'shimmer 0.7s ease forwards'
      }}
      onMouseLeave={e => {
        e.currentTarget.style.borderColor = 'var(--border)'
        e.currentTarget.style.boxShadow = 'none'
        const shimmer = e.currentTarget.querySelector('.shimmer')
        if (shimmer) shimmer.style.animation = 'none'
      }}
    >
      {/* Shimmer overlay */}
      <div
        className="shimmer"
        style={{
          position: 'absolute',
          inset: 0,
          background: 'linear-gradient(90deg, transparent, rgba(232,160,69,0.05), transparent)',
          transform: 'translateX(-100%)',
          pointerEvents: 'none',
        }}
      />

      {feature.span === 'full' ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
          <span style={{ color: 'var(--gold)', fontSize: 22, opacity: 0.9 }}>{feature.icon}</span>
          <div>
            <div style={{ fontFamily: 'Outfit, sans-serif', fontWeight: 500, fontSize: 15, color: 'var(--text-primary)', marginBottom: 4 }}>
              {feature.title}
            </div>
            <div style={{ fontFamily: 'Outfit, sans-serif', fontWeight: 300, fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.6 }}>
              {feature.desc}
            </div>
          </div>
        </div>
      ) : (
        <>
          <span style={{ color: 'var(--gold)', fontSize: 22, opacity: 0.9 }}>{feature.icon}</span>
          <div>
            <div style={{ fontFamily: 'Outfit, sans-serif', fontWeight: 500, fontSize: 15, color: 'var(--text-primary)', marginBottom: 8 }}>
              {feature.title}
            </div>
            <div style={{ fontFamily: 'Outfit, sans-serif', fontWeight: 300, fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.7 }}>
              {feature.desc}
            </div>
          </div>
        </>
      )}
    </motion.div>
  )
}

export default function BentoGrid() {
  const [ref, inView] = useInView()

  return (
    <section style={{ padding: '100px 60px', maxWidth: 1100, margin: '0 auto' }}>
      <div style={{ marginBottom: 56 }}>
        <h2
          style={{
            fontFamily: '"Cormorant Garamond", Georgia, serif',
            fontWeight: 600,
            fontSize: 48,
            color: 'var(--text-primary)',
            lineHeight: 1.05,
            marginBottom: 0,
            display: 'inline-block',
          }}
        >
          Why YojanaAI?
        </h2>
        <div
          style={{
            width: 40,
            height: 1,
            background: 'var(--gold)',
            marginTop: 12,
          }}
        />
      </div>

      <div
        ref={ref}
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gridAutoRows: 'auto',
          gap: 12,
        }}
      >
        {FEATURES.map((f, i) => (
          <Card key={i} feature={f} index={i} inView={inView} />
        ))}
      </div>
    </section>
  )
}
