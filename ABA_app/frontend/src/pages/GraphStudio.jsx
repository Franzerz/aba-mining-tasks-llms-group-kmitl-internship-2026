import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import Frame, { PageHead } from "../components/Frame.jsx";
import GraphCanvas from "../components/GraphCanvas.jsx";
import LegendIcon from "../components/LegendIcon.jsx";
import { LEGENDS, VIEWS, focusOptions } from "../graphElements.js";
import { fetchAspects, fetchGraph } from "../api.js";

function SummaryCard({ label, value }) {
  return (
    <article className="card p-4 min-h-[84px] flex flex-col justify-center">
      <span className="text-xs text-muted mb-1.5">{label}</span>
      <strong className="text-[clamp(16px,1.1vw,22px)] leading-tight break-words">{value}</strong>
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

export default function GraphStudio() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [aspects, setAspects] = useState([]);
  const [aspect, setAspect] = useState(searchParams.get("aspect") || "");
  // null = "whatever the server defaults to". Storing the server's answer back here would
  // re-trigger the fetch effect and recompute the whole framework a second time on first load
  // (seconds of wasted work on staff/facility), so the response's semantics is read off `data`
  // for display instead.
  const [semantics, setSemantics] = useState(null);
  const [view, setView] = useState(searchParams.get("view") || "aba");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  // Empty = whole framework. Otherwise the canvas is cut down to that atom's neighbourhood,
  // which is how mu_team's studio works: one graph per supporting literal, not the whole set.
  const [focus, setFocus] = useState(searchParams.get("focus") || "");
  // Empty = both claims. Otherwise only one side of the framework (good_X or bad_X).
  const [head, setHead] = useState(searchParams.get("head") || "");
  const canvasRef = useRef(null);

  useEffect(() => {
    fetchAspects()
      .then((r) => {
        setAspects(r.aspects);
        setAspect((cur) => cur || r.aspects[0] || "");
      })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!aspect) return;
    setLoading(true);
    setError(null);
    fetchGraph(aspect, semantics)
      .then((d) => setData(d))
      .catch((e) => {
        setError(e.message);
        setData(null);
      })
      .finally(() => setLoading(false));
  }, [aspect, semantics]);

  // Keep aspect+view in the URL so a particular view is linkable (and survives a refresh).
  function syncParams(nextAspect, nextView, nextFocus = focus, nextHead = head) {
    const params = {};
    if (nextAspect) params.aspect = nextAspect;
    if (nextView && nextView !== "aba") params.view = nextView;
    if (nextFocus) params.focus = nextFocus;
    if (nextHead) params.head = nextHead;
    setSearchParams(params);
  }

  function changeHead(next) {
    setHead(next);
    syncParams(aspect, view, focus, next);
  }

  function changeFocus(next) {
    setFocus(next);
    syncParams(aspect, view, next);
  }

  function changeAspect(next) {
    setAspect(next);
    setFocus("");                 // atoms are aspect-specific; a stale focus would show nothing
    setHead("");
    syncParams(next, view, "", "");
  }

  function changeView(next) {
    setView(next);
    syncParams(aspect, next);
  }

  const activeView = VIEWS.find((v) => v.key === view);
  const yesPairs = data ? data.pairs.filter((p) => p.vote === "Yes").length : 0;

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
                <option key={a} value={a}>{a}</option>
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
          <SummaryCard label="Clones (C)" value={data.aba.assumptions.length} />
          <SummaryCard label="AA arguments" value={data.aa.arguments.length} />
          <SummaryCard label="Defeats" value={data.aa.defeats.length} />
          <SummaryCard label="Semantics" value={data.semantics} />
        </section>
      )}

      <section className="mt-4 grid grid-cols-1 xl:grid-cols-[1fr_minmax(0,340px)] gap-4">
        <section className="card p-5 min-w-0">
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
            </div>
            <div className="flex gap-1.5">
              <button className="btn" onClick={() => canvasRef.current?.zoomOut()} title="Zoom out">−</button>
              <button className="btn" onClick={() => canvasRef.current?.zoomIn()} title="Zoom in">+</button>
              <button className="btn" onClick={() => canvasRef.current?.fit()} title="Fit to viewport">Fit</button>
              <button className="btn" onClick={() => canvasRef.current?.relayout()} title="Re-run the layout (it is randomized, so this can untangle a crowded spot)">Re-layout</button>
            </div>
          </div>

          <p className="text-[12px] text-muted mt-0 mb-3 leading-relaxed">{activeView?.blurb}</p>

          {/* One graph per supporting literal / contrary, as mu_team's studio does, instead of
              only ever showing the whole framework. */}
          <div className="flex items-center gap-2 flex-wrap mb-3">
            <label className="text-[12px] text-muted" htmlFor="head-select">Head claim</label>
            <select id="head-select" className="field" value={head}
                    onChange={(e) => changeHead(e.target.value)} disabled={!data}>
              <option value="">Both claims</option>
              {(data?.heads || []).map((h) => <option key={h} value={h}>{h}</option>)}
            </select>

            <label className="text-[12px] text-muted ml-2" htmlFor="focus-select">Focus on</label>
            <select id="focus-select" className="field min-w-0 max-w-full sm:max-w-[420px]"
                    value={focus} onChange={(e) => changeFocus(e.target.value)} disabled={!data}>
              <option value="">Whole framework</option>
              {focusOptions(data).map((group) => (
                <optgroup key={group.group} label={`${group.group} (${group.atoms.length})`}>
                  {group.atoms.map((atom) => <option key={atom} value={atom}>{atom}</option>)}
                </optgroup>
              ))}
            </select>
            {focus && (
              <button className="btn" onClick={() => changeFocus("")}>Clear focus</button>
            )}
          </div>

          <div className="h-[560px] border border-edge rounded-panel overflow-hidden relative">
            {loading && (
              <div className="absolute inset-0 grid place-content-center text-[13px] text-muted bg-elevated/70 z-10">
                Computing…
              </div>
            )}
            <GraphCanvas ref={canvasRef} data={data} view={view} focus={focus} head={head}
                         onNodeSelect={setSelectedNode} />
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

        <aside className="flex flex-col gap-4 min-w-0">
          <SidebarCard title="Legend">
            <ul className="m-0 p-0 list-none space-y-1.5">
              {LEGENDS[view].map(([label, color, shape]) => (
                <li key={label} className="flex items-center gap-2 text-[12px]">
                  <LegendIcon shape={shape} color={color} />
                  <span className="text-muted">{label}</span>
                </li>
              ))}
            </ul>
          </SidebarCard>

          <SidebarCard
            title={`Extensions (${data?.semantics || "—"})`}
            hint="Assumption sets accepted under the selected semantics — computed by py_arg on the reduced framework."
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
            title="Human-voted pairs"
            hint={`Real Task 3 gold annotations. ${yesPairs} "Yes" vote(s) became additional contraries in C*.`}
          >
            <div className="max-h-72 overflow-y-auto space-y-1.5">
              {data?.pairs.slice(0, 200).map((p) => (
                <div key={p.id} className="text-[11px] border-b border-edge/60 pb-1.5">
                  <div className="flex justify-between gap-2">
                    <span className="font-semibold break-words">{p.a}</span>
                    <span
                      className={`shrink-0 font-bold ${
                        p.vote === "Yes" ? "text-positive" : p.vote === "No" ? "text-negative" : "text-muted"
                      }`}
                    >
                      {p.vote ?? "?"}
                    </span>
                  </div>
                  <div className="text-muted break-words">vs {p.b}</div>
                </div>
              ))}
              {data && data.pairs.length > 200 && (
                <p className="text-[11px] text-muted m-0 pt-1">
                  Showing first 200 of {data.pairs.length.toLocaleString()} pairs.
                </p>
              )}
            </div>
          </SidebarCard>
        </aside>
      </section>
    </Frame>
  );
}
