/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif']
      },
      colors: {
        'card-dark': '#020617'
      },
      boxShadow: {
        'glow-blue': '0 0 40px rgba(56, 189, 248, 0.4)'
      }
    }
  },
  plugins: []
};
