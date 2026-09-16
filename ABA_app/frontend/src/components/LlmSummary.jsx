// Plain-English summary of whatever the Graph Studio canvas is currently showing, written for
// someone running a hotel rather than someone reading a framework.
//
// The card always has something in it: the offline summary is computed locally from the same
// tallies the canvas drew, and a local Ollama model (if one is installed) can rewrite it into
// friendlier prose on request. A machine with no Ollama is a normal, supported state — the
// card just says so under the box and keeps showing the offline text.

import { useCallback, useEffect, useState } from "react";
import { fetchLlmModels, fetchLlmSummary } from "../api.js";
import { offlineSummary } from "../graphElements.js";

// A summary describes one specific picture, so switching view/aspect/filter makes it stale.
function factsKey(facts) {
  if (!facts) return "";
  return [facts.aspect, facts.view, facts.semantics, facts.focusLabel, facts.headLabel].join("|");
}

/** Lives once in the page so the sidebar card and the fullscreen panel share one model list and
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

export default function LlmSummaryBody({ llm, facts }) {
  const text = llm.summary || offlineSummary(facts);

  return (
    <>
      <div
        className="card-elevated p-2.5 text-[12px] leading-relaxed whitespace-pre-line break-words
                   min-h-[132px] max-h-[240px] overflow-y-auto"
      >
        {text || "No graph loaded yet."}
      </div>

      {llm.available ? (
        <div className="mt-2 flex items-center gap-1.5 flex-wrap">
          <select
            className="field !py-1 !px-2 text-[11px] min-w-0 flex-1"
            value={llm.model}
            onChange={(e) => llm.setModel(e.target.value)}
            disabled={llm.generating || !llm.models.length}
            title="Which installed Ollama model writes the summary"
          >
            {llm.models.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
          <button
            className="btn !px-2 !py-1 text-[11px]"
            onClick={llm.refresh}
            disabled={llm.probing || llm.generating}
            title="Re-check which models Ollama has installed"
          >
            {llm.probing ? "…" : "Refresh"}
          </button>
          <button
            className="btn !px-2 !py-1 text-[11px]"
            onClick={llm.generate}
            disabled={llm.generating || !llm.model || !facts}
            title="Rewrite the summary with the selected model"
          >
            {llm.generating ? "Writing…" : "Summarize"}
          </button>
        </div>
      ) : (
        <p className="mt-2 mb-0 text-[11px] text-muted">
          {llm.probing ? "Checking for Ollama…" : "ollama not found"}
        </p>
      )}

      {llm.error && <p className="mt-1.5 mb-0 text-[11px] text-negative break-words">{llm.error}</p>}

      <p className="mt-1.5 mb-0 text-[11px] text-muted">
        {llm.generating
          ? `${llm.model} is writing — a local model can take a few minutes.`
          : llm.summary
            ? `Written by ${llm.model}.`
            : "Offline summary, counted from the current view."}
      </p>
    </>
  );
}
