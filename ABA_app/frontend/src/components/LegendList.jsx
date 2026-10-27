import LegendIcon from "./LegendIcon.jsx";
import { LEGENDS } from "../graphElements.js";

/** The legend for one view. Entries tagged with a contrary source only appear when that source
 *  actually drew something in `data` (no "LLM" entry on a fully human-voted topic, and back). */
export default function LegendList({ view, data, compact = false }) {
  const sources = new Set(Object.values(data?.contrary_source || {}));
  return (
    <ul className={`m-0 p-0 list-none ${compact ? "space-y-1" : "space-y-1.5"}`}>
      {LEGENDS[view].filter(([, , , only]) => !only || sources.has(only)).map(([label, color, shape]) => (
        <li key={label} className={`flex items-center gap-2 ${compact ? "text-[11px]" : "text-[12px]"}`}>
          <LegendIcon shape={shape} color={color} />
          <span className="text-muted">{label}</span>
        </li>
      ))}
    </ul>
  );
}
