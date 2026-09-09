import { NavLink } from "react-router-dom";

// Centered white nav bar, matching mu_team's .top-nav / .nav-link treatment.
const LINKS = [
  { to: "/", label: "Home", end: true },
  { to: "/review", label: "Review" },
  { to: "/graph", label: "Graph Studio" },
];

export default function TopNav() {
  return (
    <header className="h-[72px] bg-white/95 border-b border-black/[0.06] flex items-center justify-center px-8 shrink-0">
      <nav className="w-full max-w-[1280px] flex items-center justify-center gap-14">
        {LINKS.map(({ to, label, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              `text-sm font-semibold py-2.5 transition-opacity ${
                isActive ? "text-navactive opacity-100" : "text-[#111] opacity-90 hover:opacity-65"
              }`
            }
          >
            {label}
          </NavLink>
        ))}
      </nav>
    </header>
  );
}
