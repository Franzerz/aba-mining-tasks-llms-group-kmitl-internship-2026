import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import Frame, { PageHead } from "../components/Frame.jsx";
import StatGauge, { GaugeLegend } from "../components/StatGauge.jsx";
import { fetchDatasetSummary, fetchTopicRatios } from "../api.js";

// Every figure on this page is counted from the Task 1 gold CSV at request time — nothing is
// hardcoded. If the gold file changes, these numbers change with it.
const PIPELINE = [
  { task: "Task 1", title: "Topic & sentiment mining", body: "Mines the aspect, the supporting span, and its polarity from each hotel review." },
  { task: "Task 2", title: "Body-literal generation", body: "Turns each mined span into body literals plus their auto-derived contrary literals." },
  { task: "Task 3", title: "Contrary classification", body: "Decides, for a pair of literals, whether one is genuinely contrary to the other — the attack relation." },
  { task: "Task 4", title: "ABA framework & graph", body: "Assembles the framework, applies the Translation Rule, and evaluates it with py_arg." },
];

function StatCard({ label, value, hint }) {
  return (
    <article className="card p-4 min-h-[96px] flex flex-col justify-center">
      <span className="text-xs text-muted mb-2">{label}</span>
      <strong className="text-[clamp(18px,1.2vw,24px)] leading-tight break-words">{value}</strong>
      {hint && <span className="text-[11px] text-muted mt-1">{hint}</span>}
    </article>
  );
}

export default function Home() {
  const [summary, setSummary] = useState(null);
  const [topics, setTopics] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([fetchDatasetSummary(), fetchTopicRatios()])
      .then(([s, t]) => {
        setSummary(s);
        setTopics(t.topics);
      })
      .catch((e) => setError(e.message));
  }, []);

  // Largest topic, so each bar can be drawn as a share of it rather than always full width.
  const maxTotal = topics.reduce((m, t) => Math.max(m, t.total), 0);

  return (
    <Frame>
      <PageHead
        eyebrow="Assumption-Based Argumentation"
        title="ABA Mining Studio"
        subtitle="Mining argumentation frameworks from hotel reviews — from raw review text through
          topic and literal extraction to a formally evaluated ABA/AA graph."
        actions={
          <>
            <Link className="btn" to="/review">Browse reviews</Link>
            <Link className="btn btn-active" to="/graph">Open Graph Studio</Link>
          </>
        }
      />

      {error && (
        <div className="mt-6 card-elevated p-4 text-negative text-sm">Could not load dataset: {error}</div>
      )}

      {summary && (
        <>
          <section className="mt-6 grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-3">
            <StatCard
              label="Mined spans"
              value={summary.rows.toLocaleString()}
              hint={
                summary.raw_rows
                  ? `of ${summary.raw_rows.toLocaleString()} rows — ${summary.excluded_rows} off-topic excluded`
                  : undefined
              }
            />
            <StatCard label="Topics covered" value={summary.topics} />
            <StatCard label="Distinct body literals" value={summary.body_literals.toLocaleString()} />
            <StatCard
              label="Topics with a framework"
              value={summary.topics_with_framework.length}
              hint={summary.topics_with_framework.join(", ")}
            />
            <StatCard label="Positive / negative" value={`${summary.positive_pct}% / ${summary.negative_pct}%`} />
          </section>

          <section className="mt-4 grid grid-cols-1 lg:grid-cols-[minmax(0,360px)_1fr] gap-4">
            <article className="card p-5">
              <p className="eyebrow m-0">Statistics</p>
              <h2 className="mt-1 mb-3 text-base font-extrabold">Sentiment split across the gold set</h2>
              <div className="h-px bg-black/10 mb-4" />
              <StatGauge positivePct={summary.positive_pct} total={summary.rows} />
              <GaugeLegend positivePct={summary.positive_pct} negativePct={summary.negative_pct} />
              <p className="text-[11px] text-muted mt-3 leading-relaxed">
                {summary.positive.toLocaleString()} positive and {summary.negative.toLocaleString()} negative
                spans. Off-topic rows are excluded.
              </p>
            </article>

            <article className="card p-5">
              <p className="eyebrow m-0">Dataset breakdown</p>
              <h2 className="mt-1 mb-1 text-base font-extrabold">Mined spans per topic</h2>
              <p className="text-[11px] text-muted mt-0 mb-2 leading-relaxed">
                One span = one mined (review &times; topic &times; sentiment) row, so a single review can
                contribute several. Bar length is the topic's share of the {(summary.rows || 0).toLocaleString()} spans.
              </p>
              <div className="h-px bg-black/10 mb-3" />

              {/* Column headings — the bare number column previously had no label at all. */}
              <div className="flex items-center gap-3 pb-1.5 mb-1 border-b border-edge/60
                              text-[10px] font-bold uppercase tracking-wide text-muted">
                <span className="w-28 shrink-0">Topic</span>
                <span className="flex-1">Positive / negative</span>
                <span className="w-14 text-right">Spans</span>
                <span className="w-28">ABA framework</span>
              </div>

              <div className="space-y-2.5">
                {topics.map((t) => {
                  // The bar is scaled by the topic's SHARE of the dataset, with the pos/neg
                  // split inside it. Previously every bar was full width, so Booking-issue
                  // (3 spans) looked exactly as substantial as Room (602).
                  const share = maxTotal ? (t.total / maxTotal) * 100 : 0;
                  return (
                    <div key={t.topic} className="flex items-center gap-3">
                      <span className="w-28 shrink-0 text-[13px] font-semibold">{t.topic}</span>
                      <div className="flex-1 h-4 rounded-full bg-black/5 overflow-hidden">
                        <div className="h-full rounded-full overflow-hidden flex"
                             style={{ width: `${Math.max(share, 1.5)}%` }}
                             title={`${t.total} spans — ${t.positive} positive, ${t.negative} negative`}>
                          <div className="bg-positive h-full" style={{ width: `${t.positive_pct}%` }} />
                          <div className="bg-negative h-full" style={{ width: `${t.negative_pct}%` }} />
                        </div>
                      </div>
                      <span className="w-14 text-right text-[12px] text-muted tabular-nums">{t.total}</span>
                      <span className="w-28 text-[11px]">
                        {t.aspect ? (
                          <span className="text-positive font-semibold">available</span>
                        ) : (
                          <span className="text-muted">not annotated</span>
                        )}
                      </span>
                    </div>
                  );
                })}
              </div>
            </article>
          </section>

          <section className="mt-4 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
            {PIPELINE.map((p) => (
              <article key={p.task} className="card p-5">
                <p className="eyebrow m-0">{p.task}</p>
                <h3 className="mt-1 mb-2 text-[15px] font-extrabold">{p.title}</h3>
                <p className="text-[13px] text-muted leading-relaxed m-0">{p.body}</p>
              </article>
            ))}
          </section>
        </>
      )}
    </Frame>
  );
}
