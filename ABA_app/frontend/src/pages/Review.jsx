import { Suspense, lazy, useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import Frame, { PageHead } from "../components/Frame.jsx";
import { fetchReviewData, fetchTopicRatios, hotelLink } from "../api.js";

// Cytoscape is only needed once a graph is opened, so it stays out of this page's bundle.
const LiteralGraphModal = lazy(() => import("../components/LiteralGraphModal.jsx"));

// Mirrors mu_team's review_category page: a topic-card grid, then dual positive/negative
// panels with Supporting / Count / Contrary / Detail columns — but reading the Task 1 gold CSV
// through FastAPI instead of MySQL.

function TopicCard({ topic, active, onClick }) {
  // Every topic has Task 1 review data. What differs is whether it has contraries yet — human
  // votes (Task 3 gold) or, for topics nobody voted on, a local model's predictions — and only
  // then can it open a framework in the Graph Studio. Predicted is not the same as voted, so the
  // chip says which.
  const hasFramework = Boolean(topic.aspect);
  const predicted = topic.contrary_source === "llm";
  const noPairs = topic.contrary_source === "none";
  // Same basis as the Home page: reviews that praised vs. reviews that complained.
  const sides = topic.positive_reviews + topic.negative_reviews;
  const positivePct = sides ? Math.round((topic.positive_reviews / sides) * 100) : 0;
  return (
    <button
      type="button"
      disabled={!hasFramework}
      aria-disabled={!hasFramework}
      onClick={hasFramework ? onClick : undefined}
      title={hasFramework
        ? `${topic.topic}: ${topic.reviews} reviews (${topic.total} opinions) - ABA framework available` +
          (predicted ? " (contraries voted by an LLM, not by a person)"
            : noPairs ? " (only one sentiment was mined, so there is no pair to vote on)"
            : " (contraries voted by humans)")
        : `${topic.topic}: ${topic.reviews} reviews (${topic.total} opinions) - no contraries yet, so no framework`}
      className={`text-left rounded-panel border p-4 transition-colors ${
        active
          ? "border-accent bg-elevated shadow-soft"
          : hasFramework
            ? "border-edge bg-panel hover:bg-elevated"
            : "border-dashed border-edge/70 bg-panel/40 opacity-60 cursor-not-allowed"
      }`}
    >
      <div className="flex items-baseline justify-between gap-2">
        <span className={`font-bold text-[15px] ${hasFramework ? "" : "text-muted"}`}>
          {topic.topic}
        </span>
        <span className="text-[11px] text-muted tabular-nums">{topic.reviews} reviews</span>
      </div>

      <div className={`mt-2.5 h-2 rounded-full overflow-hidden bg-black/5 flex ${hasFramework ? "" : "opacity-45"}`}>
        <div className="bg-positive h-full" style={{ width: `${positivePct}%` }} />
        <div className="bg-negative h-full" style={{ width: `${100 - positivePct}%` }} />
      </div>

      <div className="mt-2 flex items-center justify-between gap-2 text-[11px]">
        <span className="text-muted">{positivePct}% positive</span>
        {hasFramework ? (
          predicted ? (
            <span className="rounded-full bg-[#4f6d9a]/15 text-[#4f6d9a] font-semibold px-2 py-0.5">
              Voted by LLM
            </span>
          ) : noPairs ? (
            <span className="rounded-full bg-black/5 text-muted font-semibold px-2 py-0.5">
              Unnecessary
            </span>
          ) : (
            <span className="rounded-full bg-positive/15 text-positive font-semibold px-2 py-0.5">
              Voted by humans
            </span>
          )
        ) : (
          <span className="rounded-full border border-edge text-muted px-2 py-0.5">
            no contraries yet
          </span>
        )}
      </div>
    </button>
  );
}


function GraphButton({ row, onOpen }) {
  const enabled = row.in_framework;
  return (
    <button
      type="button"
      className={`inline-flex items-center gap-1 rounded-chip border px-2 py-1 text-[11px] font-semibold
                  transition-colors ${enabled
                    ? "border-edge bg-elevated text-ink hover:bg-[#f7f2ea] hover:border-[#c9baa8]"
                    : "border-transparent text-muted/60 cursor-not-allowed"}`}
      disabled={!enabled}
      title={enabled
        ? `Show the graph around ${row.supporting}`
        : "This literal is not in the topic's framework (the voted Task 3 set left it out)"}
      onClick={(e) => {
        // The button sits inside the row's <summary>: don't also expand the row.
        e.preventDefault();
        e.stopPropagation();
        if (enabled) onOpen(row.supporting);
      }}
    >
      <svg width="12" height="12" viewBox="0 0 16 16" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.6">
        <circle cx="3" cy="8" r="2" /><circle cx="13" cy="3.5" r="2" /><circle cx="13" cy="12.5" r="2" />
        <path d="M5 7.2 11 4.3M5 8.8l6 2.9" />
      </svg>
      Graph
    </button>
  );
}

// Header and rows share one track definition, so the count can never drift off its column.
const COLUMNS = "grid grid-cols-[minmax(0,1fr)_3.5rem_5.5rem] gap-3";

function SentimentPanel({ title, tone, rows, loading, query, onOpenGraph, hasCounts }) {
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return rows;
    return rows.filter((r) => r.supporting.toLowerCase().includes(q));
  }, [rows, query]);

  return (
    <section className="card p-4 min-w-0">
      <div className="flex items-center justify-between mb-3">
        <h3 className={`m-0 text-[15px] font-extrabold ${tone}`}>{title}</h3>
        <span className="text-[11px] text-muted">{filtered.length} literal(s)</span>
      </div>

      {loading ? (
        <p className="text-[13px] text-muted p-3 m-0">Loading…</p>
      ) : filtered.length === 0 ? (
        <p className="text-[13px] text-muted p-3 m-0">No literals for this topic and sentiment.</p>
      ) : (
        // The header lives inside the scroller: outside it, the scrollbar's width would shift the
        // rows left and the numbers would sit off their column.
        <div className="max-h-[460px] overflow-y-auto">
          <div className={`${COLUMNS} sticky top-0 z-10 bg-panel px-2 pb-2 pt-1 border-b border-edge
                           text-[11px] font-bold uppercase tracking-wide text-muted`}>
            <div>Supporting</div>
            <div className="text-right">Count</div>
            <div className="text-center">Graph</div>
          </div>
          <div className="divide-y divide-edge/60">
          {filtered.map((row) => (
            <details key={row.supporting} className="group px-2 py-2">
              <summary className={`${COLUMNS} items-center cursor-pointer list-none`}>
                <span className="text-[12px] font-semibold break-words">{row.supporting}</span>
                <span className="text-[12px] text-muted tabular-nums text-right"
                      title={hasCounts ? undefined : "This hotel's literals come from its vote files, which do not record which reviews produced them."}>
                  {hasCounts ? row.count : "-"}
                </span>
                <GraphButton row={row} onOpen={onOpenGraph} />
              </summary>
              <div className="mt-2 pl-1 border-l-2 border-edge">
                <p className="text-[11px] text-muted m-0 mb-1">
                  Head claim: <code className="text-ink">{row.head || "-"}</code>
                  {row.contrary && <> · assumption: <code className="text-ink">{row.contrary}</code></>}
                </p>
                <ul className="m-0 pl-4 text-[12px] text-muted space-y-1">
                  {row.details.map((d, i) => (
                    <li key={i}>“{d}”</li>
                  ))}
                </ul>
              </div>
            </details>
          ))}
          </div>
        </div>
      )}
    </section>
  );
}

export default function Review() {
  const navigate = useNavigate();
  // ?topic=Room&graph=clean_room&side=Positive — the chosen topic and an open graph are linkable.
  const [params, setParams] = useSearchParams();
  const [topics, setTopics] = useState([]);
  const [selected, setSelected] = useState(null);
  const [positive, setPositive] = useState([]);
  const [negative, setNegative] = useState([]);
  // False for a hotel whose literals come from its vote files: those name a literal but not the
  // reviews it was mined from, so there is no occurrence to count.
  const [hasCounts, setHasCounts] = useState(true);
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState("");
  const [error, setError] = useState(null);
  const graphFor = params.get("graph")
    ? { literal: params.get("graph"), sentiment: params.get("side") || "Positive" }
    : null;
  const updateParams = (changes) => {
    const next = new URLSearchParams(params);
    for (const [k, v] of Object.entries(changes)) {
      if (v) next.set(k, v);
      else next.delete(k);
    }
    setParams(next, { replace: true });
  };
  const setGraphFor = (g) => updateParams({ graph: g?.literal, side: g?.sentiment });
  const choose = (topic) => {
    setSelected(topic);
    updateParams({ topic, graph: null, side: null });
  };

  useEffect(() => {
    fetchTopicRatios()
      .then((t) => {
        setTopics(t.topics);
        // Default to a topic that can actually be opened — the list is sorted by size, and
        // the biggest (Room) has no Task 3 annotation, so it would land on a disabled card.
        const wanted = t.topics.find((x) => x.topic === params.get("topic") && x.aspect);
        const firstUsable = wanted || t.topics.find((x) => x.aspect) || null;
        if (firstUsable) setSelected(firstUsable.topic);
      })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!selected) return;
    setLoading(true);
    Promise.all([fetchReviewData(selected, "Positive"), fetchReviewData(selected, "Negative")])
      .then(([p, n]) => {
        setPositive(p.rows);
        setNegative(n.rows);
        setHasCounts(p.has_counts !== false);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [selected]);

  const selectedTopic = topics.find((t) => t.topic === selected);

  return (
    <Frame>
      <PageHead
        eyebrow="Task 1 gold data"
        title="Review Analysis"
        subtitle="Choose an aspect to see the body literals mined from real reviews and how often
          each appears. Open a row for its head claim, assumption and source reviews, or press
          Graph to see what it supports and what attacks it."
        actions={
          selectedTopic?.aspect ? (
            <button className="btn btn-active" onClick={() => navigate(hotelLink(`/graph?aspect=${selectedTopic.aspect}`))}>
              Open {selectedTopic.topic} in Graph Studio
            </button>
          ) : (
            <span className="text-[12px] text-muted max-w-[220px] text-right">
              {selectedTopic ? `${selectedTopic.topic} has no Task 3 gold data yet` : ""}
            </span>
          )
        }
      />

      {error && <div className="mt-6 card-elevated p-4 text-negative text-sm">{error}</div>}

      <section className="mt-6 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        {topics.map((t) => (
          <TopicCard key={t.topic} topic={t} active={t.topic === selected} onClick={() => t.aspect && choose(t.topic)} />
        ))}
      </section>

      {selected && (
        <section className="mt-5">
          <div className="flex items-center justify-between gap-4 mb-3 flex-wrap">
            <h2 className="m-0 text-lg font-extrabold">{selected}</h2>
            <input
              className="field w-full sm:w-80"
              placeholder="Search a literal"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            <SentimentPanel title="Positive Review" tone="text-positive" rows={positive} loading={loading} hasCounts={hasCounts}
                            query={query} onOpenGraph={(literal) => setGraphFor({ literal, sentiment: "Positive" })} />
            <SentimentPanel title="Negative Review" tone="text-negative" rows={negative} loading={loading} hasCounts={hasCounts}
                            query={query} onOpenGraph={(literal) => setGraphFor({ literal, sentiment: "Negative" })} />
          </div>
        </section>
      )}

      {graphFor && selectedTopic?.aspect && (
        <Suspense fallback={<div className="fixed inset-0 z-50 bg-black/35 grid place-content-center text-white">Loading graph…</div>}>
          <LiteralGraphModal
            aspect={selectedTopic.aspect}
            topic={selectedTopic.topic}
            literal={graphFor.literal}
            sentiment={graphFor.sentiment}
            source={selectedTopic.contrary_source}
            onClose={() => setGraphFor(null)}
          />
        </Suspense>
      )}
    </Frame>
  );
}
