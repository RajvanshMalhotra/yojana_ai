import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import Navbar from '../components/Navbar'
import VerticalMarquee from '../components/VerticalMarquee'
import HorizontalMarquee from '../components/HorizontalMarquee'
import BentoGrid from '../components/BentoGrid'
import HowItWorks from '../components/HowItWorks'
import CTASection from '../components/CTASection'
import useInView from '../hooks/useInView'

const HEADLINE_WORDS = ['Find', 'the', 'schemes', 'that', 'belong', 'to', 'you.']

function StatsStrip() {
  const [ref, inView] = useInView()
  const stats = [
    { value: '1,200+', label: 'Schemes indexed' },
    { value: '28', label: 'States covered' },
    { value: '100%', label: 'Free to use' },
  ]

  return (
    <div
      ref={ref}
      style={{
        display: 'flex',
        justifyContent: 'center',
        gap: 0,
        padding: '60px 60px',
        borderTop: '1px solid var(--border)',
        maxWidth: 800,
        margin: '0 auto',
      }}
    >
      {stats.map((s, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5, delay: i * 0.1, ease: [0.22, 1, 0.36, 1] }}
          style={{
            flex: 1,
            textAlign: 'center',
            padding: '0 32px',
            borderRight: i < 2 ? '1px solid var(--border)' : 'none',
          }}
        >
          <div
            style={{
              fontFamily: '"Cormorant Garamond", Georgia, serif',
              fontWeight: 600,
              fontSize: 44,
              color: 'var(--gold)',
              lineHeight: 1,
              marginBottom: 8,
            }}
          >
            {s.value}
          </div>
          <div
            style={{
              fontFamily: 'Outfit, sans-serif',
              fontWeight: 300,
              fontSize: 13,
              color: 'var(--text-muted)',
              letterSpacing: '0.04em',
            }}
          >
            {s.label}
          </div>
        </motion.div>
      ))}
    </div>
  )
}

export default function Landing() {
  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg-base)' }}>
      <Navbar />

      {/* Hero */}
      <section
        style={{
          height: '100vh',
          display: 'flex',
          flexDirection: 'column',
          paddingTop: 60,
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        {/* Ambient orb */}
        <div
          style={{
            position: 'absolute',
            top: '20%',
            right: '15%',
            width: 400,
            height: 400,
            borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(232,160,69,0.18) 0%, transparent 70%)',
            filter: 'blur(60px)',
            pointerEvents: 'none',
          }}
        />
        <div
          style={{
            position: 'absolute',
            bottom: '25%',
            left: '25%',
            width: 280,
            height: 280,
            borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(201,106,58,0.1) 0%, transparent 70%)',
            filter: 'blur(80px)',
            pointerEvents: 'none',
          }}
        />

        {/* Two-column layout */}
        <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
          <VerticalMarquee />

          {/* Right column */}
          <div
            style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              padding: '0 80px 0 60px',
            }}
          >
            {/* Headline */}
            <h1
              style={{
                fontFamily: '"Cormorant Garamond", Georgia, serif',
                fontWeight: 600,
                fontSize: 'clamp(56px, 7vw, 96px)',
                lineHeight: 0.95,
                color: 'var(--text-primary)',
                marginBottom: 28,
                letterSpacing: '-0.02em',
              }}
            >
              {HEADLINE_WORDS.map((word, i) => (
                <motion.span
                  key={i}
                  initial={{ opacity: 0, y: 24 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{
                    duration: 0.5,
                    delay: 0.1 + i * 0.08,
                    ease: [0.22, 1, 0.36, 1],
                  }}
                  style={{ display: 'inline-block', marginRight: '0.25em' }}
                >
                  {word}
                </motion.span>
              ))}
            </h1>

            {/* Subtitle */}
            <motion.p
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.75, ease: [0.22, 1, 0.36, 1] }}
              style={{
                fontFamily: 'Outfit, sans-serif',
                fontWeight: 300,
                fontSize: 18,
                color: 'var(--text-muted)',
                lineHeight: 1.7,
                marginBottom: 44,
                maxWidth: 460,
              }}
            >
              Over a thousand central &amp; Delhi government schemes.
              Ask in plain Hindi or English.
            </motion.p>

            {/* CTAs */}
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.9, ease: [0.22, 1, 0.36, 1] }}
              style={{ display: 'flex', gap: 16, alignItems: 'center' }}
            >
              <Link to="/chat">
                <motion.button
                  whileHover={{ scale: 1.02, background: 'var(--gold)', color: '#faf7f2' }}
                  whileTap={{ scale: 0.97 }}
                  style={{
                    background: 'transparent',
                    border: '1px solid var(--gold)',
                    color: 'var(--gold)',
                    borderRadius: 10,
                    padding: '14px 32px',
                    fontSize: 15,
                    fontFamily: 'Outfit, sans-serif',
                    fontWeight: 500,
                    cursor: 'pointer',
                    animation: 'breathe 3s ease-in-out infinite',
                    transition: 'background 0.2s, color 0.2s',
                  }}
                >
                  Ask Now !
                </motion.button>
              </Link>

              <a
                href="#how-it-works"
                style={{
                  fontFamily: 'Outfit, sans-serif',
                  fontWeight: 400,
                  fontSize: 14,
                  color: 'var(--text-muted)',
                  textDecoration: 'none',
                  letterSpacing: '0.02em',
                  transition: 'color 0.2s',
                }}
                onMouseEnter={e => { e.currentTarget.style.color = 'var(--text-primary)' }}
                onMouseLeave={e => { e.currentTarget.style.color = 'var(--text-muted)' }}
              >
                How it works ↓
              </a>
            </motion.div>

            {/* Footnote */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5, delay: 1.1 }}
              style={{
                marginTop: 52,
                display: 'flex',
                alignItems: 'center',
                gap: 16,
              }}
            >
              <div style={{ width: 32, height: 1, background: 'var(--border)' }} />
              <span
                style={{
                  fontFamily: 'Outfit, sans-serif',
                  fontWeight: 300,
                  fontSize: 12,
                  color: 'var(--text-faint)',
                  letterSpacing: '0.08em',
                }}
              >
                1,200+ schemes · Central &amp; Delhi
              </span>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Horizontal ticker */}
      <HorizontalMarquee />

      {/* Stats */}
      <StatsStrip />

      {/* Bento grid */}
      <div style={{ borderTop: '1px solid var(--border)' }}>
        <BentoGrid />
      </div>

      {/* How it works */}
      <div id="how-it-works" style={{ borderTop: '1px solid var(--border)' }}>
        <HowItWorks />
      </div>

      {/* CTA */}
      <CTASection />

      {/* Footer strip */}
      <div
        style={{
          padding: '20px 60px',
          borderTop: '1px solid var(--border)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ color: 'var(--gold)', fontSize: 14 }}>✦</span>
          <span
            style={{
              fontFamily: '"Cormorant Garamond", Georgia, serif',
              fontWeight: 600,
              fontSize: 16,
              color: 'var(--text-muted)',
            }}
          >
            YojanaAI
          </span>
        </span>
        <span
          style={{
            fontFamily: 'Outfit, sans-serif',
            fontWeight: 300,
            fontSize: 12,
            color: 'var(--text-faint)',
          }}
        >
          Central &amp; Delhi government schemes · Always up to date
        </span>
      </div>
    </div>
  )
}
