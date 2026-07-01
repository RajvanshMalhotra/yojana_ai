import { motion } from 'framer-motion'
import useInView from '../hooks/useInView'

const STEPS = [
  {
    num: '01',
    title: 'Tell us who you are',
    desc: 'Describe yourself in plain language — your state, occupation, age, or situation. No forms to fill.',
  },
  {
    num: '02',
    title: 'We search 1,200+ schemes',
    desc: 'Our AI searches across central and Delhi government schemes and ranks the most relevant ones for you.',
  },
  {
    num: '03',
    title: 'Get matched, instantly',
    desc: 'Receive clear answers with eligibility details, benefits, and direct links to apply on the official portal.',
  },
]

export default function HowItWorks() {
  const [ref, inView] = useInView()

  return (
    <section style={{ padding: '80px 60px 100px', maxWidth: 1100, margin: '0 auto' }}>
      <div style={{ marginBottom: 56 }}>
        <h2
          style={{
            fontFamily: '"Cormorant Garamond", Georgia, serif',
            fontWeight: 600,
            fontSize: 48,
            color: 'var(--text-primary)',
            display: 'inline-block',
          }}
        >
          How it works
        </h2>
        <div style={{ width: 40, height: 1, background: 'var(--gold)', marginTop: 12 }} />
      </div>

      <div
        ref={ref}
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: 24,
          position: 'relative',
        }}
      >
        {/* Connector line */}
        <div
          style={{
            position: 'absolute',
            top: 60,
            left: '16%',
            right: '16%',
            height: 1,
            borderTop: '1px dashed rgba(232,160,69,0.3)',
            pointerEvents: 'none',
          }}
        />

        {STEPS.map((step, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 40 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.7, delay: i * 0.12, ease: [0.22, 1, 0.36, 1] }}
            style={{ position: 'relative' }}
          >
            {/* Big decorative number */}
            <div
              style={{
                position: 'absolute',
                top: -28,
                left: -4,
                fontFamily: '"Cormorant Garamond", Georgia, serif',
                fontWeight: 600,
                fontSize: 120,
                color: 'var(--text-faint)',
                lineHeight: 1,
                userSelect: 'none',
                pointerEvents: 'none',
                zIndex: 0,
              }}
            >
              {step.num}
            </div>

            <div
              style={{
                background: 'var(--bg-surface-1)',
                border: '1px solid var(--border)',
                borderRadius: 16,
                padding: '80px 28px 28px',
                position: 'relative',
                zIndex: 1,
                transition: 'border-color 0.25s',
              }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--border-bright)' }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)' }}
            >
              <h3
                style={{
                  fontFamily: 'Outfit, sans-serif',
                  fontWeight: 500,
                  fontSize: 16,
                  color: 'var(--text-primary)',
                  marginBottom: 10,
                }}
              >
                {step.title}
              </h3>
              <p
                style={{
                  fontFamily: 'Outfit, sans-serif',
                  fontWeight: 300,
                  fontSize: 14,
                  color: 'var(--text-muted)',
                  lineHeight: 1.7,
                }}
              >
                {step.desc}
              </p>
            </div>
          </motion.div>
        ))}
      </div>
    </section>
  )
}
