const ITEMS = [
  'PM Kisan', 'Ayushman Bharat', 'Ladli Yojana', 'PM Awas Yojana',
  'Jan Dhan', 'Mudra Loan', 'Beti Bachao', 'Sukanya Samriddhi',
  'PM Ujjwala', 'PM Matru Vandana', 'PMEGP', 'Atal Pension',
  'National Scholarship', 'PM SVANidhi', 'PMKVY', 'Standup India',
  'Skill India', 'Startup India', 'MGNREGS', 'Antyodaya Anna',
]

export default function HorizontalMarquee() {
  const doubled = [...ITEMS, ...ITEMS]

  return (
    <div
      style={{
        height: 48,
        background: 'var(--bg-surface-1)',
        borderTop: '1px solid var(--border)',
        borderBottom: '1px solid var(--border)',
        overflow: 'hidden',
        display: 'flex',
        alignItems: 'center',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 0,
          animation: 'scrollLeft 40s linear infinite',
          whiteSpace: 'nowrap',
          willChange: 'transform',
        }}
      >
        {doubled.map((item, i) => (
          <span key={i} style={{ display: 'flex', alignItems: 'center' }}>
            <span
              style={{
                fontFamily: 'Outfit, sans-serif',
                fontWeight: 500,
                fontSize: 11,
                letterSpacing: '0.15em',
                textTransform: 'uppercase',
                color: 'var(--gold)',
                padding: '0 24px',
              }}
            >
              {item}
            </span>
            <span style={{ color: 'var(--border-bright)', fontSize: 12 }}>·</span>
          </span>
        ))}
      </div>
    </div>
  )
}
