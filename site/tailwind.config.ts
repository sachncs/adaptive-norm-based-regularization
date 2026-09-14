/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "Inter var",
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "Geist Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "Consolas",
          "monospace",
        ],
        display: [
          "Inter var",
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "sans-serif",
        ],
      },
      colors: {
        ink: {
          50: "#F7F8FA",
          100: "#EEF0F4",
          200: "#DBDFE6",
          300: "#B8BFCB",
          400: "#8C95A6",
          500: "#5F6878",
          600: "#3F4856",
          700: "#2A323F",
          800: "#1A2029",
          850: "#131820",
          900: "#0C1018",
          950: "#070A10",
        },
        brand: {
          50: "#EAF1FB",
          100: "#D2E2F6",
          200: "#A4C4ED",
          300: "#6FA0DF",
          400: "#3F7CD0",
          500: "#2563B9",
          600: "#1B4F96",
          700: "#163E78",
          800: "#13315F",
          900: "#0E2447",
        },
        accent: {
          50: "#F1F4FF",
          100: "#E1E7FF",
          200: "#B7C5FF",
          300: "#8A9FFF",
          400: "#6477FF",
          500: "#4656F5",
          600: "#3438D8",
          700: "#2727AD",
          800: "#1E1E8A",
          900: "#171870",
        },
      },
      letterSpacing: {
        tightest: "-0.04em",
        "tight-2": "-0.025em",
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(255,255,255,0.05), 0 30px 80px -20px rgba(70,86,245,0.35)",
        soft: "0 1px 0 0 rgba(255,255,255,0.06) inset, 0 1px 2px rgba(0,0,0,0.3)",
        card: "0 1px 0 0 rgba(255,255,255,0.04) inset, 0 30px 60px -30px rgba(0,0,0,0.6)",
      },
      backgroundImage: {
        "grid-fade":
          "radial-gradient(circle at 50% 0%, rgba(70,86,245,0.18), transparent 60%)",
        "noise":
          "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0.045 0'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>\")",
        "shine":
          "linear-gradient(180deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0) 100%)",
      },
      keyframes: {
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(12px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-6px)" },
        },
        "pulse-soft": {
          "0%, 100%": { opacity: "0.6" },
          "50%": { opacity: "1" },
        },
      },
      animation: {
        "fade-up": "fade-up 0.7s cubic-bezier(0.22,1,0.36,1) both",
        shimmer: "shimmer 8s linear infinite",
        float: "float 6s ease-in-out infinite",
        "pulse-soft": "pulse-soft 3s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
