// One supporting literal's graph, opened from the Review page (like mu_team's per-literal
// graphs): the literal, its assumption, the claim it backs, what attacks it and what it attacks.
// Loaded lazily with Cytoscape, so the Review page itself stays light.

import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import GraphCanvas from "./GraphCanvas.jsx";
import LegendList from "./LegendList.jsx";
import { fetchGraph, hotelLink } from "../api.js";
import { CONTRARY_SETS, VIEWS, graphFacts, humanAtom } from "../graphElements.js";

/** The one figure this popup is about: how many literals contradict the one on screen. */
function AttacksCard({ count, predicted, predictedBy }) {
  const none = count === 0;
  return (
    <div
      className={`rounded-panel border p-4 ${
        none ? "border-edge bg-elevated" : "border-negative/25 bg-negative/[0.06]"
      }`}
    >
      <div className="flex items-center gap-2">
        <span
          aria-hidden="true"
          className={`grid place-items-center w-6 h-6 rounded-full shrink-0 ${
            none ? "bg-black/[0.06] text-muted" : "bg-negative/15 text-negative"
          }`}
        >
          <svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor"
               strokeWidth="1.8" strokeLinecap="round">
            <path d="M3.5 3.5 12.5 12.5M12.5 3.5 3.5 12.5" />
          </svg>
        </span>
        <span className="text-[11px] font-bold uppercase tracking-wide text-muted">Attacks</span>
      </div>

      <div className={`mt-2 text-[34px] font-extrabold leading-none tabular-nums ${
        none ? "text-muted" : "text-negative"
      }`}>
        {count.toLocaleString()}
      </div>

      <p className="m-0 mt-2 text-[11.5px] leading-relaxed text-muted">
        {none
          ? "Nothing in this topic contradicts it, so its assumption stands."
          : `${count === 1 ? "One literal contradicts" : `${count.toLocaleString()} literals contradict`} it — each dashed arrow in the picture.`}
      </p>

      {predicted > 0 && (
        <p className="m-0 mt-2 pt-2 border-t border-negative/15 text-[11px] leading-relaxed text-[#4f6d9a]">
          {predicted === count ? "All" : predicted.toLocaleString()} of them {predicted === 1 ? "was" : "were"}{" "}
          predicted by {predictedBy} and checked by nobody.
        </p>
      )}
    </div>
  );
}

export default function LiteralGraphModal({ aspect, topic, literal, sentiment, source, onClose }) {
  const navigate = useNavigate();
  const canvasRef = useRef(null);
  const [view, setView] = useState("new_aba");
  // One literal's neighbourhood stays small, so both sets are readable here: what attacks its
  // assumption, and which assumptions it attacks. A human-voted topic only has the first set —
  // the annotators voted on Contrary(P)Body(N) alone, and nothing is predicted for it.
  const [contraries, setContraries] = useState(source === "human" ? "pn" : "both");
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError(null);
    // The focus goes to the server too: very large topics only send the focused part of the
    // reduced and AA views.
    fetchGraph(aspect, null, literal, contraries)
      .then((d) => !cancelled && setData(d))
      .catch((e) => !cancelled && setError(e.message));
    return () => { cancelled = true; };
  }, [aspect, literal, contraries]);

  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [onClose]);

  const facts = useMemo(() => (data ? graphFacts(data, view, literal, "") : null), [data, view, literal]);
  const attackers = facts ? facts.edges.conflict : 0;

  return (
    <div
      className="fixed inset-0 z-50 bg-black/35 grid place-items-center p-4"
      onMouseDown={(e) => e.target === e.currentTarget && onClose()}
      role="dialog"
      aria-modal="true"
      aria-label={`Graph for ${literal}`}
    >
      <div className="card bg-page w-[min(1180px,100%)] max-h-[calc(100vh-2rem)] overflow-y-auto p-5 shadow-frame">
        <header className="flex items-start justify-between gap-4 flex-wrap">
          <div className="min-w-0">
            <p className="eyebrow m-0">{topic} · {sentiment} review literal</p>
            <h2 className="m-0 mt-1 text-[22px] font-extrabold break-words">{literal}</h2>
            <p className="m-0 mt-1 text-[12px] text-muted">
              What “{humanAtom(literal)}” backs up, the assumption it rests on, and what attacks it.
            </p>
          </div>
          <div className="flex gap-2 shrink-0">
            <button
              className="btn"
              // merge=0 so Graph Studio opens on the same unmerged picture this modal drew;
              // without it the reader lands on the Loose default and the node count changes.
              onClick={() => navigate(hotelLink(`/graph?aspect=${encodeURIComponent(aspect)}&view=${view}&focus=${encodeURIComponent(literal)}&merge=0${contraries !== "pn" ? `&contraries=${contraries}` : ""}`))}
            >
              Open in Graph Studio
            </button>
            <button className="btn" onClick={onClose} aria-label="Close">Close</button>
          </div>
        </header>

        <div className="mt-4 flex items-center justify-between gap-3 flex-wrap">
          <div className="flex gap-1.5 flex-wrap">
            {VIEWS.map((v) => (
              <button key={v.key} className={`btn !py-1.5 ${view === v.key ? "btn-active" : ""}`}
                      onClick={() => setView(v.key)}>
                {v.label}
              </button>
            ))}
          </div>
          <div className="inline-flex rounded-chip border border-edge bg-elevated p-0.5" role="radiogroup" aria-label="Contrary set">
            {CONTRARY_SETS.filter((c) => !data?.contrary_sets || data.contrary_sets[c.key]).map((c) => (
              <button
                key={c.key}
                role="radio"
                aria-checked={contraries === c.key}
                title={c.hint}
                className={`px-2 py-1 rounded-[8px] text-[11px] font-semibold transition-colors ${
                  contraries === c.key ? "bg-accent text-white" : "text-ink hover:bg-[#f7f2ea]"}`}
                onClick={() => setContraries(c.key)}
              >
                {c.label}
              </button>
            ))}
          </div>
          <div className="flex gap-1.5">
            <button className="btn !py-1.5" onClick={() => canvasRef.current?.zoomOut()} title="Zoom out">−</button>
            <button className="btn !py-1.5" onClick={() => canvasRef.current?.zoomIn()} title="Zoom in">+</button>
            <button className="btn !py-1.5" onClick={() => canvasRef.current?.fit()}>Fit</button>
          </div>
        </div>

        <div className="mt-3 grid grid-cols-1 lg:grid-cols-[1fr_260px] gap-4">
          <div className="h-[min(58vh,560px)] border border-edge rounded-panel overflow-hidden relative">
            {!data && !error && (
              <div className="absolute inset-0 grid place-content-center text-[13px] text-muted z-10">Loading…</div>
            )}
            {error && (
              <div className="absolute inset-0 grid place-content-center p-6 text-center text-[13px] text-negative z-10">
                {error}
              </div>
            )}
            <GraphCanvas ref={canvasRef} data={data} view={view} focus={literal} head="" />
          </div>

          <aside className="flex flex-col gap-3 min-w-0">
            {facts && (
              <AttacksCard count={attackers} predicted={facts.edges.predicted}
                           predictedBy={facts.predictedBy} />
            )}
            {data && data.contrary_sets && !data.contrary_sets.pn && !data.contrary_sets.np && (
              <p className="m-0 text-[11px] leading-relaxed text-muted">
                This topic was only mined with one sentiment, so nothing in it can contradict this
                literal.
              </p>
            )}
            <div className="card p-3">
              <p className="eyebrow m-0 mb-2">Legend</p>
              <LegendList view={view} data={data} compact />
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}
