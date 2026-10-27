// Donut gauge ported from mu_team's .stats-gauge SVG (two arcs on a shared circle, rotated
// -90deg so the split starts at 12 o'clock). Circumference for r=78 is 2*pi*78 = 490.1.
const R = 78;
const CIRCUMFERENCE = 2 * Math.PI * R;

export default function StatGauge({ positivePct, total, label = "Total spans" }) {
  const pos = (positivePct / 100) * CIRCUMFERENCE;
  const neg = CIRCUMFERENCE - pos;

  return (
    <div className="relative w-[190px] h-[190px] mx-auto">
      <svg viewBox="0 0 240 240" className="w-full h-full block" aria-hidden="true">
        <circle
          cx="120" cy="120" r={R} fill="none" strokeWidth="18"
          className="stroke-black/10"
          style={{ transform: "rotate(-90deg)", transformOrigin: "120px 120px" }}
        />
        <circle
          cx="120" cy="120" r={R} fill="none" strokeWidth="18" stroke="#5e6b47"
          strokeDasharray={`${pos} ${neg}`}
          style={{ transform: "rotate(-90deg)", transformOrigin: "120px 120px" }}
        />
        <circle
          cx="120" cy="120" r={R} fill="none" strokeWidth="18" stroke="#8a3434"
          strokeDasharray={`${neg} ${pos}`} strokeDashoffset={-pos}
          style={{ transform: "rotate(-90deg)", transformOrigin: "120px 120px" }}
        />
      </svg>
      <div className="absolute inset-0 grid place-content-center text-center gap-0.5">
        <div className="text-[11px] font-bold text-ink/55">{label}</div>
        <div className="text-lg font-extrabold tracking-[0.2px]">{total.toLocaleString()}</div>
      </div>
    </div>
  );
}

export function GaugeLegend({ positivePct, negativePct }) {
  return (
    <div className="flex justify-center gap-6 pt-1 flex-wrap">
      {[
        ["POSITIVE", positivePct, "bg-positive"],
        ["NEGATIVE", negativePct, "bg-negative"],
      ].map(([name, pct, dot]) => (
        <div key={name} className="flex items-center gap-2 text-[11px]">
          <span className={`w-2.5 h-2.5 rounded-full inline-block ${dot}`} />
          <span className="tracking-[0.5px] text-ink/55 font-extrabold">{name}</span>
          <span className="font-extrabold text-ink/85">{pct}%</span>
        </div>
      ))}
    </div>
  );
}
