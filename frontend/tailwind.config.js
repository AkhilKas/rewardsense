/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: "var(--color-primary, #2563eb)",
          light: "var(--color-primary-light, #dbeafe)",
        },
        secondary: "var(--color-secondary, #1e40af)",
        surface: "var(--color-surface, #f8fafc)",
        card: "var(--color-card, #ffffff)",
        border: "var(--color-border, #e2e8f0)",
      },
    },
  },
  plugins: [],
};