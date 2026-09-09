/** Design tokens lifted from mu_team's Project_ABA (frontend/assets/css/pyarg.css :root), so
 *  this app reads as the same product family while running on a different stack. */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        page: "#f3eee7",
        section: "#efe7db",
        panel: "#e8e0d4",
        elevated: "#faf7f2",
        edge: "#d6cabc",
        ink: "#2e241d",
        muted: "#6e6257",
        positive: "#5e6b47",
        negative: "#8a3434",
        accent: "#a8947e",
        navactive: "#837461",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
      },
      borderRadius: { card: "18px", panel: "14px", chip: "10px" },
      boxShadow: {
        soft: "0 10px 26px rgba(64, 45, 27, 0.08)",
        frame: "0 18px 34px rgba(64, 45, 27, 0.12)",
      },
    },
  },
  plugins: [],
};
