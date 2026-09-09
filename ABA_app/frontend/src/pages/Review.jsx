import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import Frame, { PageHead } from "../components/Frame.jsx";
import { fetchReviewData, fetchTopicRatios } from "../api.js";

// Mirrors mu_team's review_category page: a topic-card grid, then dual positive/negative
// panels with Supporting / Count / Contrary / Detail columns — but reading the Task 1 gold CSV
// through FastAPI instead of MySQL.

function TopicCard({ topic, active, onClick }) {
  // Every topic has Task 1 review data, so all cards stay browsable. What differs is whether
  // Task 3 annotated it — only then can it open a framework in the Graph Studio. That is a
  // real distinction, so it is shown as a chip and a muted card rather than a footnote that
  // reads the same as everything else.
  const hasFramework = Boolean(topic.aspect);
  return (
    <button
      type="button"
      disabled={!hasFramework}
      aria-disabled={!hasFramework}
      onClick={hasFramework ? onClick : undefined}
      title={hasFramework
        ? `${topic.topic}: ${topic.total} mined spans — ABA framework available`
        : `${topic.topic}: ${topic.total} mined spans — no Task 3 annotation, so no framework yet`}
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
        <span className="text-[11px] text-muted tabular-nums">{topic.total}</span>
      </div>

      <div className={`mt-2.5 h-2 rounded-full overflow-hidden bg-black/5 flex ${hasFramework ? "" : "opacity-45"}`}>
        <div className="bg-positive h-full" style={{ width: `${topic.positive_pct}%` }} />
        <div className="bg-negative h-full" style={{ width: `${topic.negative_pct}%` }} />
      </div>

      <div className="mt-2 flex items-center justify-between gap-2 text-[11px]">
        <span className="text-muted">{topic.positive_pct}% pos</span>
        {hasFramework ? (
          <span className="rounded-full bg-positive/15 text-positive font-semibold px-2 py-0.5">
            ABA framework
          </span>
        ) : (
          <span className="rounded-full border border-edge text-muted px-2 py-0.5">
            not annotated
          </span>
        )}
      </div>
    </button>
  );
}


function SentimentPanel({ title, tone, rows, loading, query }) {
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return rows;
    return rows.filter(
      (r) => r.supporting.toLowerCase().includes(q) || (r.contrary || "").toLowerCase().includes(q)
    );
  }, [rows, query]);

  return (
    <section className="card p-4 min-w-0">
      <div className="flex items-center justify-between mb-3">
        <h3 className={`m-0 text-[15px] font-extrabold ${tone}`}>{title}</h3>
        <span className="text-[11px] text-muted">{filtered.length} literal(s)</span>
      </div>

      <div className="grid grid-cols-[1fr_auto_1fr] gap-2 px-2 pb-2 text-[11px] font-bold uppercase tracking-wide text-muted border-b border-edge">
        <div>Supporting</div>
        <div className="text-center">Count</div>
        <div>Contrary</div>
      </div>

      {loading ? (
        <p className="text-[13px] text-muted p-3 m-0">Loading…</p>
      ) : filtered.length === 0 ? (
        <p className="text-[13px] text-muted p-3 m-0">No literals for this topic and sentiment.</p>
      ) : (
        <div className="max-h-[460px] overflow-y-auto divide-y divide-edge/60">
          {filtered.map((row) => (
            <details key={row.supporting} className="group px-2 py-2">
              <summary className="grid grid-cols-[1fr_auto_1fr] gap-2 items-center cursor-pointer list-none">
                <span className="text-[12px] font-semibold break-words">{row.supporting}</span>
                <span className="text-[12px] text-muted tabular-nums text-center px-2">{row.count}</span>
                <span className="text-[12px] text-muted break-words">{row.contrary || "—"}</span>
              </summary>
              <div className="mt-2 pl-1 border-l-2 border-edge">
                <p className="text-[11px] text-muted m-0 mb-1">
                  Head claim: <code className="text-ink">{row.head || "—"}</code>
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
      )}
    </section>
  );
}

export default function Review() {
  const navigate = useNavigate();
  const [topics, setTopics] = useState([]);
  const [selected, setSelected] = useState(null);
  const [positive, setPositive] = useState([]);
  const [negative, setNegative] = useState([]);
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState("");
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchTopicRatios()
      .then((t) => {
        setTopics(t.topics);
        // Default to a topic that can actually be opened — the list is sorted by size, and
        // the biggest (Room) has no Task 3 annotation, so it would land on a disabled card.
        const firstUsable = t.topics.find((x) => x.aspect) || null;
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
        subtitle="Choose an aspect to see the body literals mined from real reviews, how often each
          appears, and the contrary literal derived for it."
        actions={
          selectedTopic?.aspect ? (
            <button className="btn btn-active" onClick={() => navigate(`/graph?aspect=${selectedTopic.aspect}`)}>
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
          <TopicCard key={t.topic} topic={t} active={t.topic === selected} onClick={() => t.aspect && setSelected(t.topic)} />
        ))}
      </section>

      {selected && (
        <section className="mt-5">
          <div className="flex items-center justify-between gap-4 mb-3 flex-wrap">
            <h2 className="m-0 text-lg font-extrabold">{selected}</h2>
            <input
              className="field w-full sm:w-80"
              placeholder="Search supporting or contrary literal"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            <SentimentPanel title="Positive Review" tone="text-positive" rows={positive} loading={loading} query={query} />
            <SentimentPanel title="Negative Review" tone="text-negative" rows={negative} loading={loading} query={query} />
          </div>
        </section>
      )}
    </Frame>
  );
}
