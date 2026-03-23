/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        'ann-bg': '#0f172a',
        'ann-card': '#111827',
        'ann-cyan': '#22d3ee',
        'ann-indigo': '#6366f1',
        'ann-highlight': '#38bdf8',
        'ann-text': '#e5e7eb',
      },
      fontFamily: {
        display: ['"Plus Jakarta Sans"', '"Manrope"', 'sans-serif'],
        body: ['"Manrope"', 'sans-serif'],
      },
      boxShadow: {
        panel: '0 30px 80px rgba(2, 6, 23, 0.55)',
        glow: '0 0 0 1px rgba(56, 189, 248, 0.25), 0 0 35px rgba(34, 211, 238, 0.22)',
        soft: '0 10px 28px rgba(15, 23, 42, 0.44)',
      },
      backgroundImage: {
        'hero-gradient': 'linear-gradient(90deg, rgba(99,102,241,0.6) 0%, rgba(124,58,237,0.48) 45%, rgba(34,211,238,0.66) 100%)',
        'neural-grid': 'linear-gradient(rgba(148,163,184,0.14) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,0.14) 1px, transparent 1px)',
      },
      animation: {
        float: 'float 8s ease-in-out infinite',
        pulseGlow: 'pulseGlow 4s ease-in-out infinite',
        shimmer: 'shimmer 7s linear infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        pulseGlow: {
          '0%, 100%': { opacity: '0.75', transform: 'scale(1)' },
          '50%': { opacity: '1', transform: 'scale(1.06)' },
        },
        shimmer: {
          from: { backgroundPosition: '0% 50%' },
          to: { backgroundPosition: '200% 50%' },
        },
      },
    },
  },
  plugins: [],
};