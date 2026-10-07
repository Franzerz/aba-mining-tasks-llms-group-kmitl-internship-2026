// Thin fetch wrapper. FastAPI returns {detail: "..."} on error; surface that text so the UI
// can show the real reason (e.g. a semantics timeout) instead of a generic failure.

/** Which hotel the page is showing, read from the URL.
 *
 *  Keeping it in the URL rather than in React state is what makes a link shareable: paste a
 *  Graph Studio URL and the recipient sees the same hotel, topic and focus. Every request below
 *  picks it up from here, so no page has to pass it down. An empty value means the default,
 *  which the server decides. */
export function currentHotel() {
  try {
    return new URLSearchParams(window.location.search).get("hotel") || "";
  } catch {
    return "";   // no window (SSR, a test runner): the server default applies
  }
}

/** An in-app link that stays on the current hotel.
 *
 *  React Router navigation replaces the whole query string, so a bare "/graph" would drop the
 *  hotel and quietly move the reader back to the default one. Every internal link that changes
 *  page goes through here. */
export function hotelLink(path) {
  const hotel = currentHotel();
  if (!hotel) return path;
  return `${path}${path.includes("?") ? "&" : "?"}hotel=${encodeURIComponent(hotel)}`;
}

function withHotel(path) {
  const hotel = currentHotel();
  if (!hotel) return path;
  return `${path}${path.includes("?") ? "&" : "?"}hotel=${encodeURIComponent(hotel)}`;
}

async function get(path) {
  const res = await fetch(withHotel(path));
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

// The picker itself must not be filtered by the current hotel.
export const fetchHotels = () => get("/api/hotels");
export const fetchAspects = () => get("/api/aspects");
export const fetchDatasetSummary = () => get("/api/dataset-summary");
export const fetchTopicRatios = () => get("/api/topic-ratios");

export const fetchReviewData = (topic, sentiment) =>
  get(`/api/review-data?topic=${encodeURIComponent(topic)}&sentiment=${encodeURIComponent(sentiment)}`);

export const fetchGraph = (aspect, semantics, focus, contraries) => {
  const params = new URLSearchParams();
  if (semantics) params.set("semantics", semantics);
  if (focus) params.set("focus", focus);   // only needed for "partial" (very large) frameworks
  if (contraries) params.set("contraries", contraries);
  const qs = params.toString();
  return get(`/api/graph/${encodeURIComponent(aspect)}${qs ? `?${qs}` : ""}`);
};

// Local Ollama, used only for the Graph Studio's plain-English summary. Absent Ollama is a
// normal state: fetchLlmModels resolves with available=false rather than rejecting.
export const fetchLlmModels = () => get("/api/llm/models");
export const fetchLlmSummary = (model, facts) => post("/api/llm/summary", { model, facts });
