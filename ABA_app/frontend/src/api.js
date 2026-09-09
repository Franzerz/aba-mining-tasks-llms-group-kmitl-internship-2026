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

export const fetchAspects = () => get("/api/aspects");
export const fetchDatasetSummary = () => get("/api/dataset-summary");
export const fetchTopicRatios = () => get("/api/topic-ratios");

export const fetchReviewData = (topic, sentiment) =>
  get(`/api/review-data?topic=${encodeURIComponent(topic)}&sentiment=${encodeURIComponent(sentiment)}`);

export const fetchGraph = (aspect, semantics) => {
  const qs = semantics ? `?semantics=${encodeURIComponent(semantics)}` : "";
  return get(`/api/graph/${encodeURIComponent(aspect)}${qs}`);
};
