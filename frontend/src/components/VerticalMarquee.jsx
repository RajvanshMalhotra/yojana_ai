const SCHEMES = [
  'PM Kisan', 'Ayushman Bharat', 'Ladli Yojana', 'PM Awas Yojana',
  'Jan Dhan', 'Mudra Loan', 'Beti Bachao', 'PM POSHAN', 'PM Ujjwala',
  'Sukanya Samriddhi', 'PM Matru Vandana', 'PMEGP', 'PMSBY', 'PMJJBY',
  'Atal Pension', 'National Scholarship', 'PM SVANidhi', 'PMKVY',
  'Standup India', 'PM CARES', 'Antyodaya Anna', 'MGNREGS', 'PMGSY',
  'Digital India', 'Skill India', 'Startup India', 'Make in India',
]

export default function VerticalMarquee() {
  const doubled = [...SCHEMES, ...SCHEMES]

  return (
    <div
      style={{
        width: 180,
        height: '100%',
        overflow: 'hidden',
        borderRight: '1px solid var(--border)',
        flexShrink: 0,
        position: 'relative',
      }}
    >
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 20,
          animation: 'scrollUp 36s linear infinite',
          paddingTop: 20,
        }}
      >
        {doubled.map((name, i) => (
          <div
            key={i}
            style={{
              fontFamily: 'Outfit, sans-serif',
              fontWeight: 500,
              fontSize: 11,
              letterSpacing: '0.14em',
              textTransform: 'uppercase',
              color: i % 4 === 0 ? 'rgba(232,160,69,0.65)' : 'var(--text-faint)',
              padding: '2px 20px',
              cursor: 'default',
              transition: 'color 0.2s, letter-spacing 0.2s',
              userSelect: 'none',
            }}
            onMouseEnter={e => {
              e.currentTarget.style.color = 'var(--gold)'
              e.currentTarget.style.letterSpacing = '0.18em'
            }}
            onMouseLeave={e => {
              e.currentTarget.style.color = i % 4 === 0 ? 'rgba(232,160,69,0.65)' : 'var(--text-faint)'
              e.currentTarget.style.letterSpacing = '0.14em'
            }}
          >
            {name}
          </div>
        ))}
      </div>
    </div>
  )
}
