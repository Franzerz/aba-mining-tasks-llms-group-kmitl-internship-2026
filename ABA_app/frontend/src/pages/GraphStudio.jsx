import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import Frame, { PageHead } from "../components/Frame.jsx";
import GraphCanvas from "../components/GraphCanvas.jsx";
import LegendList from "../components/LegendList.jsx";
import LlmSummaryBody, { useLlmSummary } from "../components/LlmSummary.jsx";
import {
  CONTRARY_SETS, DEFAULT_CONTRARY_SET, DEFAULT_MERGE, DEFAULT_VIEW, MERGE_LEVELS, VIEWS,
  acceptedAssumptions, normaliseMerge,
  backsHead, focusOptions,
  graphFacts,
} from "../graphElements.js";
import { fetchAspects, fetchGraph } from "../api.js";

// Past this many nodes + edges Cytoscape stalls the tab and the picture is an unreadable blot
// anyway. The largest human-voted view (staff, reduced: ~10k) stays under it; the LLM-only
// topics' full views do not, and ask for a focus instead.
const MAX_DRAW_ELEMENTS = 12000;
const DEFAULT_ASPECT = "staff";

function SummaryCard({ label, value }) {
  return (
    <article className="card p-4 min-h-[84px] flex flex-col justify-center">
      <span className="text-xs text-muted mb-1.5">{label}</span>
      <strong className="text-[clamp(16px,1.1vw,22px)] leading-tight break-words">
        {typeof value === "number" ? value.toLocaleString() : value}
      </strong>
    </article>
  );
}

function SidebarCard({ title, hint, children }) {
  return (
    <section className="card p-4">
      <h3 className="m-0 mb-1 text-[13px] font-extrabold uppercase tracking-wide">{title}</h3>
      {hint && <p className="text-[11px] text-muted mt-0 mb-2.5 leading-relaxed">{hint}</p>}
      {children}
    </section>
  );
}

/** Flip one contrary set on or off, keeping at least one on.
 *
 *  State stays the same three-valued key the rest of the app and the API use ("pn" | "np" |
 *  "both"); only the control is two switches. */
function toggleSet(current, key) {
  const other = key === "pn" ? "np" : "pn";
  if (current === "both") return other;      // turning this one off leaves the other
  if (current === key) return current;       // the last one on: refuse to turn it off
  return "both";                             // the other was on alone; add this one
}

/** The accepted assumptions of the whole framework, reported the way py_arg's own ABA
 *  visualiser reports them: one flat list of assumption names under the selected semantics.
 *
 *  Deliberately NOT filtered by the head claim on screen. Acceptance is a property of the
 *  framework, not of whichever side of it is currently being drawn, so narrowing it to one claim
 *  would give a number that disagrees with the extension the server computed. Clicking an
 *  assumption focuses the graph on the literal it rests on.
 */
function AcceptedAssumptionsPanel({ data, view, head, onFocus }) {
  const result = useMemo(() => acceptedAssumptions(data, head), [data, head]);
  const literalOf = useMemo(() => {
    const map = new Map();
    const roles = (data && data.new_aba && data.new_aba.atom_roles) || {};
    for (const rule of (data && data.new_aba && data.new_aba.rules) || []) {
      const lits = rule.body.filter(
        (b) => roles[b] === "proposition_pos" || roles[b] === "proposition_neg");
      for (const a of rule.body) {
        if (String(roles[a] || "").startsWith("assumption") && !map.has(a) && lits.length) {
          map.set(a, lits[0]);
        }
      }
    }
    return map;
  }, [data]);

  if (!result) return null;
  const { accepted, rejected, total, semantics } = result;
  // Name them the way the picture on screen names them: the Simple view merges a literal with
  // its assumption and shows the literal, the ABA view draws the assumption as its own node and
  // shows its real no_evident_not_ / have_evident_ name. The set of accepted assumptions is the
  // same either way -- only the label changes.
  const simple = view === "simple";
  // Sort by the text on the chip. acceptedAssumptions orders by the raw atom, which in the
  // Simple view is the invisible prefix: the reader would see every have_evident_ point first,
  // then every no_evident_not_ one, each alphabetical, and the list would look unsorted.
  const shown = (atom) => (simple ? literalOf.get(atom) || atom : atom);
  const byShown = (a, b) => shown(a).localeCompare(shown(b));

  const Chip = ({ atom }) => (
    <button
      type="button"
      onClick={() => onFocus(literalOf.get(atom) || atom)}
      title={simple ? `${atom} — show it in the graph` : "Show it in the graph"}
      className="text-[11px] font-mono px-1.5 py-1 rounded border border-edge bg-elevated
                 hover:bg-panel break-all text-left"
    >
      {simple ? literalOf.get(atom) || atom : atom}
    </button>
  );

  return (
    <SidebarCard
      title={`Accepted assumption(s) (${semantics || "—"})`}
      hint={`${accepted.length} of ${total} assumptions behind ${head || "this topic"} are `
            + "accepted under the selected semantics and contrary set. Click one to show it in "
            + "the graph."}
    >
      <div className="flex items-baseline justify-between gap-2 mb-1.5">
        <span className="text-[12px] font-bold text-positive">Accepted</span>
        <span className="text-[11px] text-muted tabular-nums">{accepted.length} of {total}</span>
      </div>
      <div className="h-1.5 rounded-full bg-black/[0.07] overflow-hidden mb-2">
        <div className="h-full bg-positive"
             style={{ width: `${total ? (accepted.length / total) * 100 : 0}%` }} />
      </div>

      {accepted.length === 0 ? (
        <p className="text-[12px] text-muted m-0">
          Nothing is accepted here: every assumption has a contrary some guest asserted.
        </p>
      ) : (
        <div className="flex flex-wrap gap-1 max-h-56 overflow-y-auto">
          {[...accepted].sort(byShown).map((a) => <Chip key={a} atom={a} />)}
        </div>
      )}

      {rejected.length > 0 && (
        <details className="mt-2">
          <summary className="cursor-pointer text-[11px] text-muted">
            {rejected.length} not accepted — something in the data contradicts them
          </summary>
          <div className="flex flex-wrap gap-1 mt-1.5 max-h-56 overflow-y-auto opacity-70">
            {[...rejected].sort(byShown).map((a) => <Chip key={a} atom={a} />)}
          </div>
        </details>
      )}
    </SidebarCard>
  );
}

/** What the chosen contrary set leaves out, so a placeholder-only side is never a surprise. */
function ContrarySetNote({ data }) {
  const sets = data.contrary_sets || {};
  let text = null;
  if (!sets.pn && !sets.np) {
    text = "This topic was only mined with one sentiment, so there is no pair to ask about: nothing " +
      "in it can contradict anything, and every assumption keeps its default contrary.";
  } else if (data.contraries === "pn") {
    text = sets.np
      ? "Showing Contrary(P)Body(N) only: the complaint side keeps its default contraries. Pick " +
        "Contrary(N)Body(P) or Both sets to add what attacks the complaints."
      : "Contrary(N)Body(P) is not available for this topic yet, so the complaint side only has default contraries.";
  } else if (data.contraries === "np") {
    text = "Showing Contrary(N)Body(P) only: the praise side keeps its default contraries." +
      (data.contrary_source?.np === "llm"
        ? " Nobody has voted on this side, so every contrary in it is an LLM prediction."
        : " Each pair here was answered once, on the Contrary(P)Body(N) sheet; this side asks the " +
          "same pair the other way round and carries the same vote.");
  }
  return text ? <p className="m-0 text-[11px] text-muted leading-relaxed">{text}</p> : null;
}

function SourceChip({ source, label }) {
  const llm = source === "llm";
  return (
    <span
      className={`shrink-0 rounded-full px-2 py-0.5 border font-semibold ${
        llm ? "border-[#4f6d9a]/40 text-[#4f6d9a] bg-[#4f6d9a]/10" : "border-positive/40 text-positive bg-positive/10"
      }`}
    >
      {llm ? `LLM: ${label}` : "Human-voted"}
    </span>
  );
}

export default function GraphStudio() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [aspects, setAspects] = useState([]);
  const [status, setStatus] = useState(null);
  const [aspect, setAspect] = useState(searchParams.get("aspect") || "");
  // null = "whatever the server defaults to". Storing the server's answer back here would
  // re-trigger the fetch effect and recompute the whole framework a second time on first load
  // (seconds of wasted work on staff/facility), so the response's semantics is read off `data`
  // for display instead.
  const [semantics, setSemantics] = useState(null);
  // The AA view is gone — the work is about ABA, and the Dung instantiation only restated it.
  const urlView = searchParams.get("view");
  const [view, setView] = useState(VIEWS.some((v) => v.key === urlView) ? urlView : DEFAULT_VIEW);
  // 0 = every point on its own node. Otherwise near-identical wordings share one (see
  // mergeSimilar): topics repeat themselves, and the picture is mostly the same point rephrased.
  const [merge, setMerge] = useState(() => normaliseMerge(searchParams.get("merge")));
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  // Empty = whole framework. Otherwise the canvas is cut down to that atom's neighbourhood,
  // which is how mu_team's studio works: one graph per supporting literal, not the whole set.
  const [focus, setFocus] = useState(searchParams.get("focus") || "");
  // One claim at a time: good_X or bad_X. Drawing both at once buried the picture in edges, so
  // the framework is always shown from one claim's side (good_X until the reader picks the other).
  const [head, setHead] = useState(searchParams.get("head") || "");
  // Contrary(P)Body(N) alone by default: with both sets the two kinds of attack run in opposite
  // directions over the same literals and the picture turns into a criss-cross.
  // Both sets by default: the two directions of attack are both part of the framework, and
  // showing one alone hides half of what the votes say. A topic that has only one of them falls
  // back to it below.
  const [contraries, setContraries] = useState(searchParams.get("contraries") || DEFAULT_CONTRARY_SET);
  // An assumption nothing attacks still has a contrary (the contrary function is total), but that
  // default atom is derived by nothing and attacks nothing, so it is never drawn.
  const showPlaceholders = false;
  const canvasRef = useRef(null);
  const sectionRef = useRef(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [legendOpen, setLegendOpen] = useState(true);
  const [summaryOpen, setSummaryOpen] = useState(true);

  // Tallied from the same build + filters the canvas draws with, so the summary always
  // describes the picture on screen rather than the whole framework behind it.
  // Until the reader picks a side, the graph shows the topic's good_X claim.
  const currentHead = head || data?.heads?.[0] || "";
  const facts = useMemo(
    () => graphFacts(data, view, focus, currentHead, { placeholders: showPlaceholders, merge }),
    [data, view, focus, currentHead, showPlaceholders, merge],
  );
  const llm = useLlmSummary(facts);
  const needsFocus = Boolean(data?.partial && view !== "new_aba" && !focus);
  const tooLarge = needsFocus || Boolean(facts && facts.counts.elements > MAX_DRAW_ELEMENTS);

  // The URL owns the merge level, so Back and Forward move it. Reading it only in the useState
  // initializer left the dropdown and the canvas on the old level while the URL showed the new.
  useEffect(() => {
    const fromUrl = normaliseMerge(searchParams.get("merge"));
    if (fromUrl !== merge) setMerge(fromUrl);
  }, [searchParams]);   // eslint-disable-line react-hooks/exhaustive-deps

  // Escape exits fullscreen, and the page behind it shouldn't scroll while it's up.
  useEffect(() => {
    if (!isFullscreen) return;
    const onKey = (e) => {
      if (e.key === "Escape") setIsFullscreen(false);
    };
    window.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [isFullscreen]);

  // Real browser fullscreen (like F11), on top of the fixed-overlay panel above — that overlay
  // is what actually fills the screen if this API is unsupported or denied (e.g. embedded in an
  // iframe without the `fullscreen` permission). The button only ever requests/exits; whether
  // it's actually active is read back off the browser via fullscreenchange, so a native exit
  // (Esc, F11, the browser's own "exit fullscreen" bar) also drops the overlay in step.
  useEffect(() => {
    const onFullscreenChange = () => {
      if (!document.fullscreenElement) setIsFullscreen(false);
    };
    document.addEventListener("fullscreenchange", onFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", onFullscreenChange);
  }, []);

  function toggleFullscreen() {
    if (isFullscreen) {
      setIsFullscreen(false);
      if (document.fullscreenElement) document.exitFullscreen?.().catch(() => {});
    } else {
      setIsFullscreen(true);
      sectionRef.current?.requestFullscreen?.().catch(() => {});
    }
  }

  // Cytoscape caches its container's pixel size, so it needs telling once the panel has
  // actually reflowed to its new (fullscreen or not) dimensions.
  useEffect(() => {
    const id = requestAnimationFrame(() => canvasRef.current?.resize());
    return () => cancelAnimationFrame(id);
  }, [isFullscreen]);

  useEffect(() => {
    fetchAspects()
      .then((r) => {
        setAspects(r.aspects);
        setStatus(r.status || null);
        // Staff is the default: the largest topic with human-voted contraries.
        setAspect((cur) => cur || (r.aspects.includes(DEFAULT_ASPECT) ? DEFAULT_ASPECT : r.aspects[0]) || "");
      })
      .catch((e) => setError(e.message));
  }, []);

  // Very large frameworks come back "partial": their reduced/AA views are fetched per focus.
  const serverFocus = data?.partial && data.aspect === aspect ? focus : "";
  useEffect(() => {
    if (!aspect) return;
    setLoading(true);
    setError(null);
    fetchGraph(aspect, semantics, serverFocus, contraries)
      .then((d) => setData(d))
      .catch((e) => {
        setError(e.message);
        setData(null);
      })
      .finally(() => setLoading(false));
  }, [aspect, semantics, serverFocus, contraries]);

  // A link carrying ?contraries=np|both to a topic that has no such data (every human-voted one)
  // falls back, so the selector always shows the set that is actually drawn.
  useEffect(() => {
    if (data?.contrary_sets && !data.contrary_sets[contraries]) {
      // Prefer both, then whichever single set exists, so the selector always shows what is drawn.
      const fallback = ["both", "pn", "np"].find((k) => data.contrary_sets[k]);
      if (fallback && fallback !== contraries) changeContraries(fallback);
    }
  }, [data]);   // eslint-disable-line react-hooks/exhaustive-deps

  // A link can name a literal from either side — the Review page's Graph button does — while the
  // head claim defaults to good_X. The claim follows the literal, so it is actually on screen.
  useEffect(() => {
    if (!data || !focus || backsHead(data, focus, currentHead)) return;
    const match = (data.heads || []).find((h) => backsHead(data, focus, h));
    if (match) {
      setHead(match);
      syncParams(aspect, view, focus, match);
    }
  }, [data]);   // eslint-disable-line react-hooks/exhaustive-deps

  // Keep aspect+view in the URL so a particular view is linkable (and survives a refresh).
  function syncParams(nextAspect, nextView, nextFocus = focus, nextHead = head, nextContraries = contraries) {
    const params = {};
    if (nextAspect) params.aspect = nextAspect;
    if (nextView && nextView !== DEFAULT_VIEW) params.view = nextView;
    if (merge !== DEFAULT_MERGE) params.merge = String(merge);
    if (nextFocus) params.focus = nextFocus;
    if (nextHead) params.head = nextHead;
    if (nextContraries && nextContraries !== DEFAULT_CONTRARY_SET) params.contraries = nextContraries;
    setSearchParams(params);
  }

  function changeContraries(next) {
    setContraries(next);
    // Each set attacks one side: Contrary(P)Body(N) the praise assumptions, Contrary(N)Body(P) the
    // complaint ones. Point the head claim at the side the chosen set touches, or the reader gets
    // a picture of nothing but support edges. "Both sets" leaves the claim alone.
    const want = next === "pn" ? "good_" : next === "np" ? "bad_" : null;
    const side = want && (data?.heads || []).find((h) => h.startsWith(want));
    const nextHead = side || head;
    if (side && side !== head) setHead(side);
    const keepFocus = backsHead(data, focus, nextHead) ? focus : "";
    if (keepFocus !== focus) setFocus(keepFocus);
    syncParams(aspect, view, keepFocus, nextHead, next);
  }

  function changeHead(next) {
    setHead(next);
    // The picker only offers what backs the chosen claim, so a focus from the other side goes
    // with it rather than staying selected while nothing of it is drawn.
    const keep = backsHead(data, focus, next);
    if (!keep) setFocus("");
    syncParams(aspect, view, keep ? focus : "", next);
  }

  function changeFocus(next) {
    setFocus(next);
    syncParams(aspect, view, next);
  }

  function changeAspect(next) {
    setAspect(next);
    setFocus("");                 // atoms are aspect-specific; a stale focus would show nothing
    setHead("");
    // Human-voted topics only have Contrary(P)Body(N). A set carried over from the previous
    // topic would leave the selector with nothing highlighted, so it falls back to that set.
    const keep = contraries === "pn" || status?.[next]?.np;
    setContraries(keep ? contraries : "pn");
    syncParams(next, view, "", "", keep ? contraries : "pn");
  }

  function changeView(next) {
    setView(next);
    syncParams(aspect, next);
  }

  function changeMerge(next) {
    setMerge(next);
    const params = Object.fromEntries(searchParams.entries());
    if (next !== DEFAULT_MERGE) params.merge = String(next);
    else delete params.merge;
    setSearchParams(params);
  }

  const activeView = VIEWS.find((v) => v.key === view);
  const aspectStatus = (a) => status?.[a]?.pn?.source;

  return (
    <Frame>
      <PageHead
        eyebrow="Task 4 — framework & semantics"
        title="ABA Graph Studio"
        subtitle="One canonical pipeline: the multi-valued contrary framework (C*), the standard
          framework the Translation Rule reduces it to (C), and the Dung instantiation py_arg
          evaluates — all built from the same object, so the picture always matches the evaluation."
        actions={
          <>
            <select className="field" value={aspect} onChange={(e) => changeAspect(e.target.value)}>
              {aspects.map((a) => (
                <option key={a} value={a}>
                  {a}{aspectStatus(a) === "llm" ? " (LLM contraries)" : status?.[a]?.no_pairs ? " (no pairs)" : ""}
                </option>
              ))}
            </select>
            <select
              className="field"
              value={semantics ?? data?.semantics ?? ""}
              onChange={(e) => setSemantics(e.target.value)}
              disabled={!data}
            >
              {data?.available_semantics.map((s) => (
                <option key={s} value={s}>
                  {s}{data.fast_semantics.includes(s) ? "" : " (slow)"}
                </option>
              ))}
            </select>
          </>
        }
      />

      {error && (
        <div className="mt-5 card-elevated p-4 text-negative text-[13px] leading-relaxed">{error}</div>
      )}

      {data && (
        <section className="mt-5 grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">
          <SummaryCard label="Aspect" value={data.aspect} />
          <SummaryCard label="Assumptions (C*)" value={data.new_aba.assumptions.length} />
          <SummaryCard label="Clones (C)" value={data.sizes?.clones ?? data.aba.assumptions.length} />
          <SummaryCard label="AA arguments" value={data.sizes?.aa_arguments ?? data.aa.arguments.length} />
          <SummaryCard label="Defeats" value={data.sizes?.aa_defeats ?? data.aa.defeats.length} />
          <SummaryCard label="Semantics" value={data.semantics} />
        </section>
      )}

      <section className="mt-4 grid grid-cols-1 xl:grid-cols-[1fr_minmax(0,340px)] gap-4">
        <div className="flex flex-col gap-4 min-w-0">
        <section
          ref={sectionRef}
          className={
            isFullscreen
              ? "card p-5 min-w-0 fixed inset-0 z-50 rounded-none flex flex-col overflow-y-auto"
              : "card p-5 min-w-0"
          }
        >
          <div className="flex items-center justify-between gap-3 flex-wrap mb-3">
            <div className="flex gap-1.5 flex-wrap">
              {VIEWS.map((v) => (
                <button
                  key={v.key}
                  className={`btn ${view === v.key ? "btn-active" : ""}`}
                  onClick={() => changeView(v.key)}
                >
                  {v.label}
                </button>
              ))}
              <label className="ml-3 flex items-center gap-1.5 text-[12px] text-muted"
                     title="Counts points whose wording is nearly the same as one node, with a badge for how many were folded in. This compares spelling, not meaning: it groups easy_check-in with easy_self_check-in, never with quick_check-in.">
                Count similar
                <select className="field !py-1 !text-[12px]" value={merge}
                        onChange={(e) => changeMerge(Number(e.target.value))} disabled={!data}>
                  {MERGE_LEVELS.map((m) => (
                    <option key={m.value} value={m.value}>{m.label}</option>
                  ))}
                </select>
              </label>
            </div>
            <div className="flex gap-1.5">
              <button className="btn" onClick={() => canvasRef.current?.zoomOut()} title="Zoom out">−</button>
              <button className="btn" onClick={() => canvasRef.current?.zoomIn()} title="Zoom in">+</button>
              <button className="btn" onClick={() => canvasRef.current?.fit()} title="Fit to viewport">Fit</button>
              <button className="btn" onClick={() => canvasRef.current?.relayout()} title="Re-run the layout (it is randomized, so this can untangle a crowded spot)">Re-layout</button>
              <button
                className="btn"
                onClick={toggleFullscreen}
                title={isFullscreen ? "Exit fullscreen (Esc)" : "Expand the graph viewer to fill the screen"}
              >
                {isFullscreen ? "Exit fullscreen" : "Fullscreen"}
              </button>
            </div>
          </div>

          <p className="text-[12px] text-muted mt-0 mb-3 leading-relaxed">{activeView?.blurb}</p>

          {/* One graph per supporting literal / contrary, as mu_team's studio does, instead of
              only ever showing the whole framework. */}
          <div className="flex items-center gap-2 flex-wrap mb-2">
            <span className="text-[12px] text-muted">Contrary set</span>
            {/* Two independent toggles rather than three exclusive buttons: each set is a
                direction of attack that is either in the picture or not, and both at once is the
                honest default. The last one on cannot be switched off -- a framework with no
                contraries at all has nothing to show. */}
            <div className="inline-flex rounded-chip border border-edge bg-elevated p-0.5" aria-label="Contrary set">
              {CONTRARY_SETS.filter((c) => c.key !== "both")
                .filter((c) => !data?.contrary_sets || data.contrary_sets[c.key])
                .map((c) => {
                  const on = contraries === c.key || contraries === "both";
                  const last = on && contraries !== "both";
                  return (
                    <button
                      key={c.key}
                      role="switch"
                      aria-checked={on}
                      disabled={last}
                      className={`px-2.5 py-1 rounded-[8px] text-[12px] font-semibold transition-colors ${
                        on ? "bg-accent text-white" : "text-ink hover:bg-[#f7f2ea]"
                      } ${last ? "cursor-default" : ""}`}
                      title={last ? `${c.hint} — the only set left on` : c.hint}
                      onClick={() => changeContraries(toggleSet(contraries, c.key))}
                    >
                      {c.label}
                    </button>
                  );
                })}
            </div>
            <span className="text-[11px] text-muted">
              {CONTRARY_SETS.find((c) => c.key === contraries)?.hint}
            </span>
          </div>

          <div className="flex items-center gap-2 flex-wrap mb-3">
            <label className="text-[12px] text-muted" htmlFor="head-select">Head claim</label>
            <select id="head-select" className="field" value={currentHead}
                    onChange={(e) => changeHead(e.target.value)} disabled={!data}>
              {(data?.heads || []).map((h) => <option key={h} value={h}>{h}</option>)}
            </select>

            <label className="text-[12px] text-muted ml-2" htmlFor="focus-select">Focus on</label>
            <select id="focus-select" className="field min-w-0 max-w-full sm:max-w-[420px]"
                    value={focus} onChange={(e) => changeFocus(e.target.value)} disabled={!data}>
              <option value="">Whole framework</option>
              {focusOptions(data, currentHead).map((group) => (
                <optgroup key={group.group} label={`${group.group} (${group.atoms.length})`}>
                  {group.atoms.map((atom) => <option key={atom} value={atom}>{atom}</option>)}
                </optgroup>
              ))}
            </select>
            {focus && (
              <button className="btn" onClick={() => changeFocus("")}>Clear focus</button>
            )}
          </div>

          <div
            className={
              isFullscreen
                ? "flex-1 min-h-0 border border-edge rounded-panel overflow-hidden relative"
                : "h-[560px] border border-edge rounded-panel overflow-hidden relative"
            }
          >
            {loading && (
              <div className="absolute inset-0 grid place-content-center text-[13px] text-muted bg-elevated/70 z-10">
                Computing…
              </div>
            )}
            <GraphCanvas ref={canvasRef} data={data} view={view} focus={focus} head={currentHead}
                         merge={merge} placeholders={showPlaceholders} suppressed={tooLarge}
                         onNodeSelect={setSelectedNode} />
            {tooLarge && !loading && (
              <div className="absolute inset-0 grid place-content-center p-6 text-center z-10">
                <div className="card-elevated max-w-[460px] p-5">
                  <p className="m-0 font-bold text-[14px]">
                    {needsFocus
                      ? "This framework is too large to draw whole"
                      : `This view has ${facts.counts.elements.toLocaleString()} nodes and links`}
                  </p>
                  <p className="mt-2 mb-0 text-[12px] text-muted leading-relaxed">
                    That is too many to draw readably. Pick a literal under <b>Focus on</b> to see
                    one part of it{view !== "new_aba" ? ", or switch to the New ABA view" : ""}. The
                    summary, extensions and pair lists still cover the whole framework.
                  </p>
                </div>
              </div>
            )}

            {/* The sidebar is outside this fixed-overlay section, so its cards disappear in
                fullscreen — float the two that matter while reading the graph over the canvas
                instead, in one column so they can never overlap each other. */}
            {isFullscreen && (
              <div className="absolute top-3 right-3 z-20 w-[270px] max-h-[calc(100%-1.5rem)]
                              flex flex-col gap-2 overflow-y-auto">
                <div className="card p-3">
                  <div className="flex items-center justify-between gap-3">
                    <span className="eyebrow">What this graph says</span>
                    <button
                      className="btn !px-2 !py-1 text-[11px]"
                      onClick={() => setSummaryOpen((o) => !o)}
                      title={summaryOpen ? "Collapse summary" : "Expand summary"}
                    >
                      {summaryOpen ? "Hide" : "Show"}
                    </button>
                  </div>
                  {summaryOpen && (
                    <div className="mt-2">
                      <LlmSummaryBody llm={llm} facts={facts} compact />
                    </div>
                  )}
                </div>

                <div className="card p-3">
                  <div className="flex items-center justify-between gap-3">
                    <span className="eyebrow">Legend</span>
                    <button
                      className="btn !px-2 !py-1 text-[11px]"
                      onClick={() => setLegendOpen((o) => !o)}
                      title={legendOpen ? "Collapse legend" : "Expand legend"}
                    >
                      {legendOpen ? "Hide" : "Show"}
                    </button>
                  </div>
                  {legendOpen && (
                    <div className="mt-2">
                      <LegendList view={view} data={data} />
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {selectedNode ? (
            <div className="mt-3 card-elevated p-3 text-[12px] leading-relaxed break-words">
              <span className="eyebrow">Selected node</span>
              <div className="font-semibold mt-1">{selectedNode.full}</div>
              {selectedNode.premise && (
                <div className="text-muted mt-1">Premise: {selectedNode.premise}</div>
              )}
            </div>
          ) : (
            <p className="mt-3 text-[11px] text-muted m-0">
              Labels are shortened at <code>#</code>. Click any node to see its full clone name.
            </p>
          )}
        </section>

        {/* Full width under the graph rather than squeezed into the sidebar: this is prose meant
            to be read, and the sidebar's width made it a scrolling wall of small text. */}
        <section className="card p-5" aria-label="Plain-language explanation">
          <LlmSummaryBody llm={llm} facts={facts} />
        </section>
        </div>

        <aside className="flex flex-col gap-4 min-w-0">
          <SidebarCard title="Legend">
            <LegendList view={view} data={data} />
          </SidebarCard>

          <AcceptedAssumptionsPanel data={data} view={view} head={currentHead}
                                    onFocus={changeFocus} />

          <SidebarCard
            title={`Extensions (${data?.semantics || "—"})`}
            hint={data?.engine && data.engine !== "py_arg"
              ? "Assumption sets accepted under the selected semantics, on the reduced framework. This framework is too large for py_arg's own loops, so an indexed re-implementation computed it (checked identical to py_arg on every human-voted topic)."
              : "Assumption sets accepted under the selected semantics — computed by py_arg on the reduced framework."}
          >
            {!data ? (
              <p className="text-[12px] text-muted m-0">—</p>
            ) : data.extensions.length === 0 ? (
              <p className="text-[12px] text-muted m-0">No extensions.</p>
            ) : (
              <div className="max-h-56 overflow-y-auto space-y-2">
                {data.extensions.map((ext, i) => (
                  <details key={i} className="card-elevated p-2">
                    <summary className="cursor-pointer text-[12px] font-semibold">
                      #{i + 1} — {ext.length} assumption(s)
                    </summary>
                    <div className="mt-1.5 flex flex-wrap gap-1">
                      {ext.map((a) => (
                        <span key={a} className="text-[10px] bg-panel border border-edge rounded px-1.5 py-0.5 break-all">
                          {a}
                        </span>
                      ))}
                    </div>
                  </details>
                ))}
              </div>
            )}
          </SidebarCard>

          <SidebarCard
            title="Where the contraries come from"
            hint={'Every "Yes" became a contrary in C*. Human votes are Task 3 gold; LLM answers are predictions nobody has checked.'}
          >
            <div className="space-y-2 mb-3">
              {(data?.pair_stats || []).map((s) => (
                <div key={s.category} className="card-elevated px-2.5 py-2 text-[11px]">
                  <div className="flex justify-between gap-2 items-center">
                    <span className="font-semibold">{s.category}</span>
                    <SourceChip source={s.source} label={s.label} />
                  </div>
                  <div className="text-muted mt-1">
                    {s.total.toLocaleString()} pairs · <span className="text-positive font-semibold">{s.yes.toLocaleString()} Yes</span>
                    {" · "}{s.no.toLocaleString()} No
                  </div>
                </div>
              ))}
              {data && <ContrarySetNote data={data} />}
            </div>
            <div className="max-h-72 overflow-y-auto space-y-1.5">
              {data?.pairs.map((p) => (
                <div key={p.id} className="text-[11px] border-b border-edge/60 pb-1.5">
                  <div className="flex justify-between gap-2">
                    <span className="font-semibold break-words">{p.assumption}</span>
                    <span
                      className={`shrink-0 font-bold ${
                        p.vote === "Yes" ? "text-positive" : p.vote === "No" ? "text-negative" : "text-muted"
                      }`}
                      title={p.source === "llm" ? "LLM prediction" : "Human vote"}
                    >
                      {p.vote ?? "?"}{p.source === "llm" ? " (LLM)" : ""}
                    </span>
                  </div>
                  <div className="text-muted break-words">attacked by {p.b}?</div>
                </div>
              ))}
              {data?.pairs.length > 0 && (
                <p className="text-[11px] text-muted m-0 pt-1">
                  A sample of {data.pairs.length} pairs, "Yes" answers first.
                </p>
              )}
            </div>
          </SidebarCard>
        </aside>
      </section>
    </Frame>
  );
}
