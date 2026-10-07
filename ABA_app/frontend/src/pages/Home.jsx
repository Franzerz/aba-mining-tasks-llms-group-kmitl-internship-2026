import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import Frame, { PageHead } from "../components/Frame.jsx";
import StatGauge, { GaugeLegend } from "../components/StatGauge.jsx";
import { fetchDatasetSummary, fetchTopicRatios, hotelLink } from "../api.js";

// Every figure on this page is counted from the Task 1 gold CSV at request time — nothing is
// hardcoded. If the gold file changes, these numbers change with it.
//
// Wording: a "raw sub opinion" is one review's praise or complaint about one topic (one row of
// the gold file, which the pipeline calls a span). The topic statistics count REVIEWS instead,
// which is what a reader expects "how many guests talked about the room" to mean.
const PIPELINE = [
  { task: "Task 1", title: "Topic & sentiment mining", body: "Finds which topics each hotel review talks about, the words it uses, and whether it praises or complains." },
  { task: "Task 2", title: "Body-literal generation", body: "Turns each of those raw sub opinions into short literals (e.g. clean_room) and the assumptions behind them." },
  { task: "Task 3", title: "Contrary classification", body: "Decides, for a pair of literals, whether one is evidence against the other — the attack relation." },
  { task: "Task 4", title: "ABA framework & graph", body: "Assembles the framework, applies the Translation Rule, and evaluates it with py_arg." },
];

const pct = (n, d) => (d ? Math.round((n / d) * 100) : 0);

/** "IDs 1–785 (no rows for 520, 563) · 744 give a raw sub opinion" */
function reviewHint(summary) {
  const parts = [];
  if (summary.review_id_range) {
    const [first, last] = summary.review_id_range;
    const missing = summary.missing_review_ids || [];
    parts.push(`IDs ${first}–${last}` + (missing.length ? ` (no rows for ${missing.join(", ")})` : ""));
  }
  if (summary.reviews_with_opinion != null) {
    parts.push(`${summary.reviews_with_opinion.toLocaleString()} give a raw sub opinion`);
  }
  return parts.join(" · ");
}

/** The small "i" in a card's corner: the long version of what the figure counts, on hover or
 *  keyboard focus. CSS-only, so it costs nothing until someone points at it. */
function InfoDot({ label, children }) {
  return (
    <span className="group absolute top-2.5 right-2.5">
      <button
        type="button"
        aria-label={`What "${label}" counts`}
        className="grid place-items-center w-[18px] h-[18px] rounded-full border border-edge
                   text-[11px] font-bold italic leading-none text-muted transition-colors
                   hover:bg-accent hover:text-white hover:border-accent
                   focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/40"
      >
        i
      </button>
      <span
        role="tooltip"
        className="pointer-events-none absolute right-0 top-[calc(100%+8px)] z-30 w-[250px]
                   card-elevated px-3 py-2.5 shadow-frame text-left font-normal normal-case
                   text-[11.5px] leading-relaxed text-ink/80
                   opacity-0 translate-y-1 transition-all duration-150
                   group-hover:opacity-100 group-hover:translate-y-0
                   group-focus-within:opacity-100 group-focus-within:translate-y-0"
      >
        {children}
        <span
          aria-hidden="true"
          className="absolute right-[5px] -top-[6px] w-3 h-3 rotate-45
                     bg-elevated border-l border-t border-edge"
        />
      </span>
    </span>
  );
}

function StatCard({ label, value, hint, info }) {
  return (
    <article className="card p-4 min-h-[96px] flex flex-col justify-center relative">
      {info && <InfoDot label={label}>{info}</InfoDot>}
      <span className={`text-xs text-muted mb-2 ${info ? "pr-6" : ""}`}>{label}</span>
      <strong className="text-[clamp(18px,1.2vw,24px)] leading-tight break-words">{value}</strong>
      {hint && <span className="text-[11px] text-muted mt-1 leading-snug">{hint}</span>}
    </article>
  );
}

const VOTED_BY_CHIPS = {
  human: ["Humans", "bg-positive/10 text-positive border-positive/30", "Contraries voted by annotators (Task 3 gold)"],
  llm: ["LLM", "bg-[#4f6d9a]/10 text-[#4f6d9a] border-[#4f6d9a]/30", "Contraries predicted by a local LLM, not checked by a person"],
  none: ["Unnecessary", "bg-black/5 text-muted border-edge", "Only one sentiment was mined, so there is no pair to vote on"],
};

function VotedByChip({ topic }) {
  const chip = topic.aspect ? VOTED_BY_CHIPS[topic.contrary_source] : null;
  if (!chip) return <span className="text-[11px] text-muted">—</span>;
  const [label, cls, title] = chip;
  return (
    <Link
      to={hotelLink(`/graph?aspect=${encodeURIComponent(topic.aspect)}`)}
      title={`${title} — open in Graph Studio`}
      className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10.5px] font-bold
                  hover:brightness-95 transition ${cls}`}
    >
      {label}
    </Link>
  );
}

/** Hover / focus card for one topic's bar, following the pointer along the bar. */
function BarTooltip({ topic, x, width, allReviews }) {
  const opinions = topic.positive_reviews + topic.negative_reviews;
  const left = Math.min(Math.max(x, 120), Math.max(width - 120, 120));
  return (
    <div
      role="tooltip"
      className="absolute bottom-[calc(100%+10px)] z-20 w-[240px] -translate-x-1/2 pointer-events-none
                 card-elevated px-3.5 py-3 shadow-frame"
      style={{ left }}
    >
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[13px] font-extrabold">{topic.topic}</span>
        <span className="text-[11px] text-muted">{topic.reviews.toLocaleString()} reviews</span>
      </div>
      <div className="mt-2 space-y-1.5 text-[12px]">
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-positive shrink-0" />
          <span className="flex-1">Positive</span>
          <strong className="tabular-nums">{topic.positive_reviews.toLocaleString()}</strong>
          <span className="w-9 text-right text-muted tabular-nums">{pct(topic.positive_reviews, opinions)}%</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-negative shrink-0" />
          <span className="flex-1">Negative</span>
          <strong className="tabular-nums">{topic.negative_reviews.toLocaleString()}</strong>
          <span className="w-9 text-right text-muted tabular-nums">{pct(topic.negative_reviews, opinions)}%</span>
        </div>
      </div>
      <p className="m-0 mt-2 pt-2 border-t border-edge text-[10.5px] text-muted leading-snug">
        Mentioned in {topic.reviews_pct}% of all {allReviews.toLocaleString()} reviews
        {topic.both_reviews > 0 && ` · ${topic.both_reviews.toLocaleString()} of them both praised and complained`}
      </p>
      <span
        aria-hidden="true"
        className="absolute left-1/2 -bottom-[6px] w-3 h-3 -translate-x-1/2 rotate-45
                   bg-elevated border-r border-b border-edge"
      />
    </div>
  );
}

function TopicRow({ topic, rank, maxReviews, allReviews, ready }) {
  const [tip, setTip] = useState(null);   // { x, width } while hovered / focused
  const opinions = topic.positive_reviews + topic.negative_reviews;
  const length = maxReviews ? Math.max((topic.reviews / maxReviews) * 100, 2) : 0;
  const positiveShare = pct(topic.positive_reviews, opinions);

  const track = (e) => {
    const box = e.currentTarget.getBoundingClientRect();
    setTip({ x: e.clientX - box.left, width: box.width });
  };

  return (
    <div className="group grid grid-cols-[20px_minmax(96px,150px)_1fr_76px] sm:grid-cols-[20px_minmax(110px,160px)_1fr_84px_92px_78px]
                    items-center gap-x-3 px-2 py-2 rounded-chip hover:bg-elevated/70 transition-colors">
      <span className="text-[11px] text-muted tabular-nums text-right">{rank}</span>

      <div className="min-w-0">
        <div className="text-[13px] font-bold truncate">{topic.topic}</div>
        <div className="text-[10.5px] text-muted">{topic.reviews_pct}% of reviews</div>
      </div>

      <div
        className="relative outline-none"
        tabIndex={0}
        aria-label={`${topic.topic}: ${topic.reviews} reviews, ${topic.positive_reviews} positive, ${topic.negative_reviews} negative`}
        onMouseMove={track}
        onMouseLeave={() => setTip(null)}
        onFocus={(e) => {
          const box = e.currentTarget.getBoundingClientRect();
          setTip({ x: (box.width * length) / 200, width: box.width });
        }}
        onBlur={() => setTip(null)}
      >
        <div className="h-[18px] rounded-full bg-black/[0.06] overflow-hidden cursor-default
                        group-focus-within:ring-2 group-focus-within:ring-accent/40">
          <div
            className="h-full flex rounded-full overflow-hidden transition-[width] duration-700 ease-out"
            style={{ width: ready ? `${length}%` : "0%" }}
          >
            <div className={`h-full bg-positive transition-opacity ${tip ? "opacity-100" : "opacity-90"}`}
                 style={{ width: `${positiveShare}%` }} />
            <div className={`h-full bg-negative transition-opacity ${tip ? "opacity-100" : "opacity-90"}`}
                 style={{ width: `${100 - positiveShare}%` }} />
          </div>
        </div>
        {tip && <BarTooltip topic={topic} x={tip.x} width={tip.width} allReviews={allReviews} />}
      </div>

      <div className="text-right">
        <div className="text-[14px] font-extrabold tabular-nums leading-tight">{topic.reviews.toLocaleString()}</div>
        <div className="text-[10px] text-muted">reviews</div>
      </div>

      <div className="hidden sm:block text-right">
        <div className={`text-[13px] font-bold tabular-nums leading-tight ${positiveShare >= 50 ? "text-positive" : "text-negative"}`}>
          {positiveShare >= 50 ? `${positiveShare}%` : `${100 - positiveShare}%`}
        </div>
        <div className="text-[10px] text-muted">{positiveShare >= 50 ? "positive" : "negative"}</div>
      </div>

      <div className="hidden sm:flex justify-end">
        <VotedByChip topic={topic} />
      </div>
    </div>
  );
}

function TopicStatistics({ topics, allReviews }) {
  const [ready, setReady] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setReady(true));   // bars grow in once
    return () => cancelAnimationFrame(id);
  }, []);

  const sorted = [...topics].sort((a, b) => b.reviews - a.reviews);
  const maxReviews = sorted.reduce((m, t) => Math.max(m, t.reviews), 0);
  const top = sorted[0];

  return (
    <article className="card p-5">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <p className="eyebrow m-0">Statistics</p>
          <h2 className="mt-1 mb-1 text-base font-extrabold">What guests talk about</h2>
          <p className="text-[12px] text-muted m-0 leading-relaxed max-w-[62ch]">
            How many of the {allReviews.toLocaleString()} hotel reviews mention each topic, and whether
            they praised it or complained. Hover a bar for the exact numbers.
          </p>
        </div>
        <div className="flex items-center gap-4 text-[11px] font-bold text-ink/60 pt-1">
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-positive" />Positive</span>
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-negative" />Negative</span>
        </div>
      </div>

      {top && (
        <p className="mt-3 mb-0 text-[12px] leading-relaxed card-elevated px-3 py-2">
          <strong>{top.topic}</strong> comes up most: {top.reviews.toLocaleString()} reviews
          ({top.reviews_pct}%), {top.positive_reviews.toLocaleString()} positive and{" "}
          {top.negative_reviews.toLocaleString()} negative.
        </p>
      )}

      <div className="mt-3 grid grid-cols-[20px_minmax(96px,150px)_1fr_76px] sm:grid-cols-[20px_minmax(110px,160px)_1fr_84px_92px_78px]
                      gap-x-3 px-2 pb-1.5 border-b border-edge/70 text-[10px] font-bold uppercase tracking-wide text-muted">
        <span className="text-right">#</span>
        <span>Topic</span>
        <span>Share of reviews · positive / negative</span>
        <span className="text-right">Reviews</span>
        <span className="hidden sm:block text-right">Mostly</span>
        <span className="hidden sm:block text-right">Voted by</span>
      </div>

      <div className="mt-1">
        {sorted.map((t, i) => (
          <TopicRow key={t.topic} topic={t} rank={i + 1} maxReviews={maxReviews}
                    allReviews={allReviews} ready={ready} />
        ))}
      </div>

      <p className="mt-3 mb-0 text-[10.5px] text-muted leading-relaxed">
        Bar length: how many reviews mention the topic, relative to the most-mentioned one. A review
        can praise and complain about the same topic, so it can count on both sides.
      </p>
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

  return (
    <Frame>
      <PageHead
        eyebrow="Assumption-Based Argumentation"
        title="ABA Mining Studio"
        subtitle="Mining argumentation frameworks from hotel reviews — from raw review text through
          topic and literal extraction to a formally evaluated ABA/AA graph."
        actions={
          <>
            <Link className="btn" to={hotelLink("/review")}>Browse reviews</Link>
            <Link className="btn btn-active" to={hotelLink("/graph")}>Open Graph Studio</Link>
          </>
        }
      />

      {error && (
        <div className="mt-6 card-elevated p-4 text-negative text-sm">Could not load dataset: {error}</div>
      )}

      {summary && (
        <>
          <section className="mt-5 grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-3">
            <StatCard
              label="Hotel reviews"
              value={(summary.reviews || 0).toLocaleString()}
              hint={reviewHint(summary)}
              info={"How many guest reviews everything on this page is counted from. " +
                    "One review is one guest writing about their stay."}
            />
            <StatCard
              label="Raw sub opinions"
              value={summary.rows.toLocaleString()}
              hint={
                "one review's praise or complaint about one topic" +
                (summary.off_topic_rows ? ` · ${summary.off_topic_rows} off-topic rows left out` : "")
              }
              info={"One thing a guest said about one topic — a single piece of praise, or a " +
                    "single complaint. A guest who talks about three topics gives three of them, " +
                    "which is why there are more of these than reviews."}
            />
            <StatCard
              label="Topics"
              value={summary.topics}
              hint="room, staff, location, …"
              info={"The things guests talk about — the room, the staff, the location, and so " +
                    "on. Each one can be opened as its own graph."}
            />
            <StatCard
              label="Distinct opinion literals"
              value={summary.body_literals.toLocaleString()}
              hint="literals that processed from raw sub opinion"
              info={"The short phrases each opinion is boiled down to, like clean_room or " +
                    "friendly_staff. Each phrase is counted once, however many guests said it."}
            />
            <StatCard
              label="Positive / negative"
              value={`${summary.positive_pct}% / ${summary.negative_pct}%`}
              hint="of all raw sub opinions"
              info={"How much of what guests said was praise and how much was a complaint. One " +
                    "guest can praise one topic and complain about another, so they can count on " +
                    "both sides."}
            />
          </section>

          <section className="mt-4 grid grid-cols-1 lg:grid-cols-[minmax(0,340px)_1fr] gap-4 items-start">
            <article className="card p-5">
              <p className="eyebrow m-0">Overall mood</p>
              <h2 className="mt-1 mb-3 text-base font-extrabold">Praise vs. complaints</h2>
              <div className="h-px bg-black/10 mb-4" />
              <StatGauge positivePct={summary.positive_pct} total={summary.rows} label="Raw sub opinions" />
              <GaugeLegend positivePct={summary.positive_pct} negativePct={summary.negative_pct} />
              <p className="text-[12px] text-muted mt-4 mb-0 leading-relaxed">
                Across every topic, guests gave{" "}
                <strong className="text-positive">{summary.positive.toLocaleString()} positive</strong> and{" "}
                <strong className="text-negative">{summary.negative.toLocaleString()} negative</strong>{" "}
                raw sub opinions — about {Math.max(1, Math.round(summary.positive / Math.max(summary.negative, 1)))} praises
                for every complaint.
              </p>
            </article>

            <TopicStatistics topics={topics} allReviews={summary.reviews || 0} />
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
