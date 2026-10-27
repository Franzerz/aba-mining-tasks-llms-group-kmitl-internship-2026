// Plain-English explanation of whatever the Graph Studio canvas is currently showing, written for
// someone running a hotel rather than someone reading a framework.
//
// The panel always has something in it: the offline summary is computed locally from the same
// tallies the canvas drew, and a local Ollama model (if one is installed) can rewrite it into
// friendlier prose on request. A machine with no Ollama is a normal, supported state — the
// panel just says so and keeps showing the offline text.

import { useCallback, useEffect, useState } from "react";
import { fetchLlmModels, fetchLlmSummary } from "../api.js";
import { offlineSummary } from "../graphElements.js";

// A summary describes one specific picture, so switching view/aspect/filter makes it stale.
function factsKey(facts) {
  if (!facts) return "";
  return [facts.aspect, facts.contraries, facts.view, facts.semantics, facts.focusLabel, facts.headLabel].join("|");
}

/** Lives once in the page so the main panel and the fullscreen overlay share one model list and
 *  one generated summary, instead of each refetching and losing the text when fullscreen opens. */
export function useLlmSummary(facts) {
  const [probing, setProbing] = useState(true);
  const [available, setAvailable] = useState(false);
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("");
  const [summary, setSummary] = useState("");
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    setProbing(true);
    try {
      const result = await fetchLlmModels();
      const list = result.models || [];
      setAvailable(Boolean(result.available));
      setModels(list);
      // One installed model is the common case — pick it rather than making them choose from a
      // list of one. Otherwise keep their choice if it survived the refresh.
      setModel((current) => (current && list.includes(current) ? current : list[0] || ""));
    } catch {
      setAvailable(false);
      setModels([]);
    } finally {
      setProbing(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const key = factsKey(facts);
  useEffect(() => {
    setSummary("");
    setError("");
  }, [key]);

  const generate = useCallback(async () => {
    if (!facts || !model) return;
    setGenerating(true);
    setError("");
    try {
      const result = await fetchLlmSummary(model, facts);
      setSummary(result.summary || "");
    } catch (e) {
      setError(e.message);
    } finally {
      setGenerating(false);
    }
  }, [facts, model]);

  return { probing, available, models, model, setModel, refresh, summary, generating, error, generate };
}

// --- text → readable list ------------------------------------------------------------------

const BULLET = /^\s*(?:[-*•●▪]|\d+[.)])\s+/;

/** Splits summary text into list items. Models often wrap the bullets in a chatty intro or
 *  outro ("Here is the explanation:"); when the text has bullets, only the bullets are kept. */
export function toItems(text) {
  const lines = String(text || "")
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);
  const bullets = lines.filter((l) => BULLET.test(l));
  return (bullets.length ? bullets : lines)
    .map((l) => l.replace(BULLET, "").replace(/^#+\s*/, "").trim())
    .filter(Boolean);
}

/** Renders one item with **bold** honoured and every number emphasised, so the figures a
 *  manager skims for stand out. Built from text nodes only — model output is never parsed as HTML. */
function RichText({ text }) {
  const parts = [];
  String(text)
    .split(/(\*\*[^*]+\*\*)/g)
    .forEach((chunk, i) => {
      if (!chunk) return;
      if (chunk.startsWith("**") && chunk.endsWith("**")) {
        parts.push(<strong key={`b${i}`} className="font-semibold text-ink">{chunk.slice(2, -2)}</strong>);
        return;
      }
      chunk.split(/(\d[\d,]*(?:\.\d+)?%?)/g).forEach((piece, j) => {
        if (!piece) return;
        parts.push(
          /^\d/.test(piece)
            ? <strong key={`n${i}-${j}`} className="font-semibold text-ink">{piece}</strong>
            : <span key={`t${i}-${j}`}>{piece}</span>,
        );
      });
    });
  return <>{parts}</>;
}

function Stat({ label, value, tone }) {
  const color = tone === "positive" ? "text-positive" : tone === "negative" ? "text-negative" : "text-ink";
  return (
    <div className="card-elevated px-3 py-2.5 min-w-0">
      <div className="text-[11px] text-muted leading-tight">{label}</div>
      <div className={`text-[20px] font-extrabold leading-tight mt-0.5 ${color}`}>
        {(value ?? 0).toLocaleString()}
      </div>
    </div>
  );
}

/** The at-a-glance numbers, in the same words the explanation uses. */
function GlanceStats({ facts }) {
  if (!facts) return null;
  const { counts, edges } = facts;
  const stats =
    facts.view === "aa"
      ? [
          { label: "Arguments on screen", value: counts.points },
          { label: "Attacks between them", value: edges.conflict, tone: "negative" },
        ]
      : [
          { label: "Praise points", value: counts.praise, tone: "positive" },
          { label: "Complaint points", value: counts.complaints, tone: "negative" },
          { label: "Backing links", value: edges.support },
          { label: "Attacks", value: edges.conflict, tone: "negative" },
        ];
  return (
    <div className={`grid gap-2 ${stats.length === 2 ? "grid-cols-2" : "grid-cols-2 sm:grid-cols-4"}`}>
      {stats.map((s) => <Stat key={s.label} {...s} />)}
    </div>
  );
}

function ItemList({ items, compact }) {
  return (
    <ul className={`m-0 p-0 list-none ${compact ? "space-y-1.5" : "space-y-2.5"}`}>
      {items.map((item, i) => (
        <li key={i} className="flex gap-2.5 items-start">
          <span
            aria-hidden="true"
            className={`shrink-0 rounded-full bg-accent ${compact ? "mt-[6px] w-1.5 h-1.5" : "mt-[8px] w-2 h-2"}`}
          />
          <span className={`${compact ? "text-[12px]" : "text-[14px]"} leading-relaxed text-ink/90 break-words`}>
            <RichText text={item} />
          </span>
        </li>
      ))}
    </ul>
  );
}

function WritingPlaceholder({ model, compact }) {
  return (
    <div role="status" aria-live="polite">
      <div className="space-y-2 animate-pulse">
        {[92, 78, 86, 64].slice(0, compact ? 3 : 4).map((w) => (
          <div key={w} className="h-3 rounded bg-edge/70" style={{ width: `${w}%` }} />
        ))}
      </div>
      <p className="mt-3 mb-0 text-[12px] text-muted">
        {model} is writing the explanation. A local model can take a minute.
      </p>
    </div>
  );
}

function SourceBadge({ llm }) {
  const byModel = Boolean(llm.summary);
  return (
    <span
      className={`text-[11px] font-semibold rounded-full px-2.5 py-0.5 border ${
        byModel ? "border-positive/40 text-positive bg-positive/10" : "border-edge text-muted bg-elevated"
      }`}
      title={byModel ? "Rewritten by a local model from the numbers on screen" : "Counted directly from the current view"}
    >
      {byModel ? `Written by ${llm.model}` : "Quick summary"}
    </span>
  );
}

function ModelControls({ llm, facts, compact }) {
  if (!llm.available) {
    return (
      <p className="m-0 text-[12px] text-muted">
        {llm.probing
          ? "Checking for a local AI model…"
          : "No local AI model found (Ollama isn't running), so the quick summary above is shown."}
      </p>
    );
  }
  return (
    <div className="flex items-center gap-2 flex-wrap">
      {!compact && <span className="text-[12px] text-muted">Explain in plain words with</span>}
      <select
        className={`field min-w-0 ${compact ? "!py-1 !px-2 text-[11px] flex-1" : "!py-1.5"}`}
        value={llm.model}
        onChange={(e) => llm.setModel(e.target.value)}
        disabled={llm.generating || !llm.models.length}
        aria-label="Local model"
        title="Which installed Ollama model writes the explanation"
      >
        {llm.models.map((m) => <option key={m} value={m}>{m}</option>)}
      </select>
      <button
        className={`btn btn-active ${compact ? "!px-2 !py-1 text-[11px]" : "!py-1.5"}`}
        onClick={llm.generate}
        disabled={llm.generating || !llm.model || !facts}
      >
        {llm.generating ? "Writing…" : llm.summary ? "Write again" : "Explain with AI"}
      </button>
      <button
        className={`btn ${compact ? "!px-2 !py-1 text-[11px]" : "!py-1.5"}`}
        onClick={llm.refresh}
        disabled={llm.probing || llm.generating}
        title="Re-check which models Ollama has installed"
      >
        {llm.probing ? "…" : "Refresh models"}
      </button>
    </div>
  );
}

/** compact: the floating card in fullscreen. Otherwise: the full-width panel under the graph. */
export default function LlmSummaryBody({ llm, facts, compact = false }) {
  const items = toItems(llm.summary || offlineSummary(facts));

  return (
    <div>
      {!compact && (
        <div className="flex items-start justify-between gap-3 flex-wrap mb-3">
          <div>
            <h3 className="m-0 text-[16px] font-extrabold">What this graph says</h3>
            <p className="m-0 mt-0.5 text-[12px] text-muted">
              A plain-language reading of the picture above, for people who don't read argument graphs.
            </p>
          </div>
          <SourceBadge llm={llm} />
        </div>
      )}

      {!compact && facts && (
        <div className="mb-4">
          <GlanceStats facts={facts} />
        </div>
      )}

      <div className={compact ? "max-h-[260px] overflow-y-auto pr-1" : ""}>
        {llm.generating ? (
          <WritingPlaceholder model={llm.model} compact={compact} />
        ) : items.length ? (
          <ItemList items={items} compact={compact} />
        ) : (
          <p className="m-0 text-[13px] text-muted">No graph loaded yet.</p>
        )}
      </div>

      {llm.error && (
        <div className="mt-3 card-elevated px-3 py-2 text-[12px] text-negative break-words" role="alert">
          Couldn't write the explanation: {llm.error}
        </div>
      )}

      <div className={`${compact ? "mt-2.5 pt-2.5" : "mt-4 pt-3"} border-t border-edge`}>
        <ModelControls llm={llm} facts={facts} compact={compact} />
      </div>
    </div>
  );
}
