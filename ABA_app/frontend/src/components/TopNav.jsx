import { useEffect, useState } from "react";
import { NavLink } from "react-router-dom";

import { currentHotel, fetchHotels } from "../api.js";

// Centered white nav bar, matching mu_team's .top-nav / .nav-link treatment.
const LINKS = [
  { to: "/", label: "Home", end: true },
  { to: "/review", label: "Review Analysis" },
  { to: "/graph", label: "Graph Studio" },
  { to: "/about", label: "About" },
];

/** Which hotel's data the whole site is showing.
 *
 *  Switching hotel navigates rather than setting React state. Every page keeps its own caches
 *  keyed by topic and view, none of which know about the hotel, so a soft switch would leave
 *  one hotel's graph on screen under the other's name. A real navigation drops all of it, and
 *  because the choice lives in the URL the new page loads the right data immediately. The
 *  topic and focus are deliberately dropped: they rarely name a topic that exists in both. */
function HotelPicker() {
  const [hotels, setHotels] = useState([]);
  const selected = currentHotel();

  useEffect(() => {
    let alive = true;
    fetchHotels()
      .then((d) => alive && setHotels(d.hotels || []))
      .catch(() => {});   // the picker is an addition; the site works without it
    return () => {
      alive = false;
    };
  }, []);

  if (hotels.length < 2) return null;   // nothing to choose between

  const active = hotels.find((h) => h.key === selected) || hotels.find((h) => h.default) || hotels[0];

  function choose(key) {
    if (key === active.key) return;
    const fallback = hotels.find((h) => h.default)?.key;
    const params = new URLSearchParams();
    if (key !== fallback) params.set("hotel", key);
    const qs = params.toString();
    window.location.assign(`${window.location.pathname}${qs ? `?${qs}` : ""}`);
  }

  return (
    <label className="absolute right-0 flex items-center gap-2 text-[12px] text-muted"
           title={active.blurb}>
      <span className="hidden sm:inline">Hotel</span>
      <select
        className="field !py-1 !text-[12px] max-w-[190px]"
        value={active.key}
        onChange={(e) => choose(e.target.value)}
      >
        {hotels.map((h) => (
          <option key={h.key} value={h.key}>{h.label}</option>
        ))}
      </select>
    </label>
  );
}

export default function TopNav() {
  return (
    <header className="h-[72px] bg-white/95 border-b border-black/[0.06] flex items-center justify-center px-8 shrink-0">
      <nav className="relative w-full max-w-[1280px] flex items-center justify-center gap-14">
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
        <HotelPicker />
      </nav>
    </header>
  );
}
