// Thin fetch wrapper. FastAPI returns {detail: "..."} on error; surface that text so the UI
// can show the real reason (e.g. a semantics timeout) instead of a generic failure.
async function get(path) {
  const res = await fetch(path);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || res.statusText);
  }
  return res.json();
}

async function post(path, payload) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || res.statusText);
  }
  return res.json();
}

export const fetchAspects = () => get("/api/aspects");
export const fetchDatasetSummary = () => get("/api/dataset-summary");
export const fetchTopicRatios = () => get("/api/topic-ratios");

export const fetchReviewData = (topic, sentiment) =>
  get(`/api/review-data?topic=${encodeURIComponent(topic)}&sentiment=${encodeURIComponent(sentiment)}`);

export const fetchGraph = (aspect, semantics) => {
  const qs = semantics ? `?semantics=${encodeURIComponent(semantics)}` : "";
  return get(`/api/graph/${encodeURIComponent(aspect)}${qs}`);
};

// Local Ollama, used only for the Graph Studio's plain-English summary. Absent Ollama is a
// normal state: fetchLlmModels resolves with available=false rather than rejecting.
export const fetchLlmModels = () => get("/api/llm/models");
export const fetchLlmSummary = (model, facts) => post("/api/llm/summary", { model, facts });
