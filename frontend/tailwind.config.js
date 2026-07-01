/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['"Cormorant Garamond"', 'Georgia', 'serif'],
        sans: ['Outfit', 'system-ui', 'sans-serif'],
      },
      colors: {
        base: '#08080c',
        surface1: '#0f0f16',
        surface2: '#17171f',
        border: '#252532',
        'border-bright': '#38384a',
        gold: '#e8a045',
        'gold-dim': '#a86c26',
        terra: '#c96a3a',
        cream: '#f5f0e8',
        muted: '#6b6878',
        faint: '#3a3845',
      },
    },
  },
  plugins: [],
}
