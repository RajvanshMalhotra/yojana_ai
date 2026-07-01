import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import useInView from '../hooks/useInView'

export default function CTASection() {
  const [ref, inView] = useInView()

  return (
    <section
      ref={ref}
      style={{
        background: 'var(--gold)',
        padding: '100px 60px',
        textAlign: 'center',
      }}
    >
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
      >
        <h2
          style={{
            fontFamily: '"Cormorant Garamond", Georgia, serif',
            fontWeight: 600,
            fontSize: 'clamp(40px, 6vw, 68px)',
            color: '#1a1610',
            lineHeight: 1.05,
            marginBottom: 16,
          }}
        >
          Don't miss what's yours.
        </h2>
        <p
          style={{
            fontFamily: 'Outfit, sans-serif',
            fontWeight: 300,
            fontSize: 18,
            color: 'rgba(26,22,16,0.65)',
            marginBottom: 44,
            maxWidth: 480,
            margin: '0 auto 44px',
          }}
        >
          Most Indians never find out they qualify for schemes that could change their lives.
        </p>

        <Link to="/chat">
          <motion.button
            whileHover={{ scale: 1.03, backgroundColor: '#1a1610', color: '#faf7f2' }}
            whileTap={{ scale: 0.97 }}
            style={{
              background: '#1a1610',
              color: '#faf7f2',
              border: 'none',
              borderRadius: 12,
              padding: '16px 40px',
              fontSize: 15,
              fontFamily: 'Outfit, sans-serif',
              fontWeight: 500,
              letterSpacing: '0.04em',
              cursor: 'pointer',
              transition: 'background 0.2s, color 0.2s',
            }}
          >
            Start for free →
          </motion.button>
        </Link>
      </motion.div>
    </section>
  )
}
