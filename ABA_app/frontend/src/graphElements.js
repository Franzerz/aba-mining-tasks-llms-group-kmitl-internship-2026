// Turns one /api/graph response into Cytoscape elements for whichever of the three views is
// selected. Ported from the previous vanilla graph-render.js — same structure, same guarantees:
// each view is a 1:1 rendering of the corresponding framework the backend actually built, so
// the picture can never disagree with what py_arg evaluated.
//
//   new_aba -> C*: A -> P(L), pre-translation. Multi-contrary fan-out drawn directly.
//   aba     -> C: A -> L, post-translation (what py_arg consumes), clones + clone-of edges.
//   aa      -> the Dung instantiation from generate_af(): arguments and defeats.
//
// Contrary edges are drawn as the ATTACK they stand for: a dashed arrow FROM the attacking
// literal TO the assumption it contradicts. That is the gold sheets' own reading — in a
// Contrary(N)Body(P) sheet, column D (the praise literal) attacks column B (the complaint
// assumption) — and it is the direction the reduced framework's defeats run in.
//
// An assumption nothing in the data attacks still needs a contrary, because the contrary function
// is total, and it gets a default one (have_evident_not_X / no_evident_X). Nothing derives that
// atom, so it never attacks anything and it is left out of the picture.

const isPlaceholder = (roles, atom) => roles[atom] === "placeholder";
const POS_ASSUMPTION = "no_evident_not_";
const NEG_ASSUMPTION = "have_evident_";

// Which side an attacked assumption is on decides where its contraries came from: the positive
// side is Contrary(P)Body(N) ("pn"), the negative side Contrary(N)Body(P) ("np"). Each side is
// either all human-voted or all LLM-predicted (see app/contrary_sources.py).
function predictedBy(data, roles, assumption) {
  const side = roles[assumption] === "assumption_neg" ? "np" : "pn";
  return data.contrary_source?.[side] === "llm";
}

// Clone atoms are named "<assumption>#<contrary>", which makes for very long labels that
// overlap badly on the canvas. Show only the assumption part; the full name stays on the node
// as `full` so clicking it can reveal which specific contrary this clone stands for.
export function shortLabel(name) {
  const hash = String(name).indexOf("#");
  return hash === -1 ? name : name.slice(0, hash);
}

// The previous team put an "evidence count" badge on each node — how many mined review spans
// produced that literal. `count` rides along in the node data; GraphCanvas renders it as a
// small overlapping circle badge rather than appending it to the text label.
function nodeCollector(counts) {
  const nodes = [];
  const seen = new Set();
  return {
    nodes,
    add(id, role) {
      if (seen.has(id)) return;
      seen.add(id);
      nodes.push({ data: { id, label: shortLabel(id), full: id, role,
                           count: (counts && counts[id]) || null } });
    },
  };
}

function buildNewAba(data, { placeholders = false } = {}) {
  const { add, nodes } = nodeCollector(data.counts);
  const edges = [];
  const roles = data.new_aba.atom_roles;
  for (const atom of data.new_aba.language) {
    if (!placeholders && isPlaceholder(roles, atom)) continue;
    add(atom, roles[atom] || "atom");
  }

  for (const rule of data.new_aba.rules) {
    for (const premise of rule.body) {
      edges.push({ data: { id: `${rule.id}_${premise}`, source: premise, target: rule.head, kind: "support", rule: rule.id } });
    }
  }
  // The whole point of the "New" formalization: one assumption, several distinct contraries.
  for (const [assumption, contraries] of Object.entries(data.new_aba.multi_contraries)) {
    for (const contrary of contraries) {
      const placeholder = isPlaceholder(roles, contrary);
      if (placeholder && !placeholders) continue;
      edges.push({
        data: { id: `contrary_${assumption}_${contrary}`, source: contrary, target: assumption,
                kind: "contrary", placeholder, llm: !placeholder && predictedBy(data, roles, assumption) },
      });
    }
  }
  return [...nodes, ...edges];
}

function buildReducedAba(data, { placeholders = false } = {}) {
  const { add, nodes } = nodeCollector(data.counts);
  const edges = [];
  const roles = data.aba.atom_roles;
  for (const atom of data.aba.language) {
    if (!placeholders && isPlaceholder(roles, atom)) continue;
    add(atom, roles[atom] || "atom");
  }

  for (const rule of data.aba.rules) {
    for (const premise of rule.body) {
      edges.push({ data: { id: `${rule.id}_${premise}`, source: premise, target: rule.head, kind: "support", rule: rule.id } });
    }
  }
  for (const [clone, contrary] of Object.entries(data.aba.contraries)) {
    const placeholder = isPlaceholder(roles, contrary);
    if (placeholder && !placeholders) continue;
    edges.push({ data: { id: `contrary_${clone}`, source: contrary, target: clone, kind: "contrary", placeholder,
                         llm: !placeholder && predictedBy(data, roles, clone) } });
  }
  // Faint dotted edge back to the original assumption, so the Translation Rule's split stays
  // visible instead of disappearing into flat clone IDs.
  // If an original were ALSO some clone's contrary (the translator bridges that case), it keeps
  // the role it plays as a contrary; only pure originals get the hollow hexagon.
  const contraryAtoms = new Set(Object.values(data.aba.contraries));
  for (const [original, clones] of Object.entries(data.aba.clones_by_original)) {
    add(original, "clone_origin");
    if (!contraryAtoms.has(original)) {
      const originalNode = nodes.find((n) => n.data.id === original);
      if (originalNode) originalNode.data.role = "clone_origin";
    }
    for (const clone of clones) {
      edges.push({ data: { id: `cloneof_${clone}`, source: clone, target: original, kind: "clone_of" } });
    }
  }
  return [...nodes, ...edges];
}

// SIMPLE ABA: one node per guest point. A point is a literal together with the assumption it
// rests on — they are the two premises of one rule and always stand or fall together — so drawing
// them as one node removes half the picture without losing anything. The node is named after the
// assumption, since that is the defeasible half: the thing an attack can knock out.
function buildSimpleAba(data, { head = "" } = {}) {
  const roles = data.new_aba.atom_roles || {};
  const counts = data.counts || {};
  const isLit = (a) => roles[a] === "proposition_pos" || roles[a] === "proposition_neg";
  const isAsm = (a) => String(roles[a] || "").startsWith("assumption");

  // Pair up the premises of each rule, and remember which claims each point backs.
  const literalOf = new Map();          // assumption -> its literal
  const assumptionsOf = new Map();      // literal -> [assumptions] (two for a dual-sentiment one)
  const headsOf = new Map();            // assumption -> Set(head)
  for (const rule of data.new_aba.rules) {
    const lits = rule.body.filter(isLit);
    const asms = rule.body.filter(isAsm);
    for (const a of asms) {
      if (!headsOf.has(a)) headsOf.set(a, new Set());
      headsOf.get(a).add(rule.head);
      for (const l of lits) {
        if (!literalOf.has(a)) literalOf.set(a, l);
        if (!assumptionsOf.has(l)) assumptionsOf.set(l, []);
        if (!assumptionsOf.get(l).includes(a)) assumptionsOf.get(l).push(a);
      }
    }
  }

  // An attacking literal is drawn as ITS point, on the side opposite the assumption it attacks.
  const pointFor = (literal, attackedRole) => {
    const options = assumptionsOf.get(literal) || [];
    if (options.length < 2) return options[0] || literal;
    const wanted = attackedRole === "assumption_pos" ? NEG_ASSUMPTION : POS_ASSUMPTION;
    return options.find((a) => a.startsWith(wanted)) || options[0];
  };

  const { add, nodes } = nodeCollector(null);
  const edges = [];
  for (const [assumption, heads] of headsOf) {
    const literal = literalOf.get(assumption);
    add(assumption, roles[assumption] || "atom");
    const node = nodes[nodes.length - 1];
    if (node && node.data.id === assumption) {
      node.data.count = (literal && counts[literal]) || null;
      node.data.literal = literal || null;
      // A merged point is shown under its LITERAL, never the no_evident_not_ / have_evident_
      // assumption that is folded into it. The prefix belongs to the ABA view, where the
      // assumption is its own node; here the two are one thing and the literal is what a reader
      // recognises. The id stays the assumption, so the attack wiring is unchanged.
      if (literal) {
        node.data.label = shortLabel(literal);
        node.data.full = `${literal}  (with ${assumption})`;
      } else {
        node.data.full = assumption;
      }
    }
    for (const head of heads) {
      add(head, "head");
      edges.push({ data: { id: `s_support_${assumption}_${head}`, source: assumption, target: head,
                           kind: "support", rule: `simple_${assumption}_${head}` } });
    }
  }

  for (const [assumption, contraries] of Object.entries(data.new_aba.multi_contraries)) {
    if (!headsOf.has(assumption)) continue;
    // Only what attacks the claim on screen. Merging a literal with its assumption puts both
    // roles on one circle, so with both contrary sets on, a praise point would attack a
    // complaint point AND be attacked by it -- a two-headed edge the ABA view never shows,
    // because there the two roles are separate nodes. Keeping just the attacks aimed at this
    // claim's own points restores the one-way picture; switch the head claim to see the
    // other direction.
    if (head && !headsOf.get(assumption).has(head)) continue;
    for (const contrary of contraries) {
      if (isPlaceholder(roles, contrary)) continue;      // nothing derives it, so it attacks nothing
      const from = pointFor(contrary, roles[assumption]);
      if (from === assumption) continue;
      edges.push({
        data: { id: `s_contrary_${from}_${assumption}`, source: from, target: assumption,
                kind: "contrary", llm: predictedBy(data, roles, assumption) },
      });
    }
  }
  const drawn = new Set(nodes.map((n) => n.data.id));
  return [...nodes, ...edges.filter((e) => drawn.has(e.data.source) && drawn.has(e.data.target))];
}

function buildAa(data) {
  const nodes = data.aa.arguments.map((arg) => ({
    data: {
      id: arg.id,
      label: shortLabel(arg.conclusion),
      full: arg.conclusion,
      premise: arg.premise.join(", "),
      role: "argument",
    },
  }));
  const edges = data.aa.defeats.map((d, i) => ({
    data: { id: `defeat_${i}`, source: d.source, target: d.target, kind: "defeat" },
  }));
  return [...nodes, ...edges];
}

export function buildElements(data, view, options = {}) {
  if (view === "simple") return buildSimpleAba(data, options);
  if (view === "new_aba") return buildNewAba(data, options);
  // A "partial" response carries these two views only for a chosen focus.
  if (view === "aba") return data.aba ? buildReducedAba(data, options) : [];
  return data.aa ? buildAa(data) : [];
}

// Which contrary sets a framework is built with (the server builds and evaluates exactly that).
export const CONTRARY_SETS = [
  { key: "pn", label: "Contrary(P)Body(N)", hint: "What attacks the praise: complaint literals against no_evident_not_X" },
  { key: "np", label: "Contrary(N)Body(P)", hint: "What attacks the complaints: praise literals against have_evident_X" },
  { key: "both", label: "Both sets", hint: "Both directions of attack at once, as the votes record them" },
];
// One direction by default. With both sets on, a Simple-view point is attacked by the other side
// AND attacks it, because the literal and the assumption are drawn as one node -- correct, but it
// reads as a node arguing with itself. The ABA view keeps them apart, so both sets are legible
// there. Either way "Both sets" is one toggle away.
export const DEFAULT_CONTRARY_SET = "pn";

// The reduced (post-translation) framework is still what py_arg evaluates, and its size is
// reported in the "Clones (C)" card -- but it is not offered as a picture any more. It only ever
// restated the ABA view with each assumption split once per contrary, which is machinery rather
// than something to read.
export const VIEWS = [
  { key: "simple", label: "Simple ABA", blurb: "One node per guest point: its literal and its assumption drawn together, so only the points, the claims they back and the attacks between them are left." },
  { key: "new_aba", label: "ABA", blurb: "Every atom separately: each point's literal, the assumption resting on it, and the literals that contradict it." },
];
export const DEFAULT_VIEW = "simple";

// [label, colour, shape] — shape names match the Cytoscape styles in GraphCanvas.jsx so the
// legend swatch looks like the thing it describes (see LegendIcon.jsx). Following the previous
// team's grammar: dashed = assumption (defeasible), solid = proposition (observed fact).
export const LEGENDS = {
  simple: [
    ["Head claim (good_X / bad_X)", "#a8947e", "roundrect"],
    ["A praise point (its literal and assumption together)", "#5e6b47", "circle-dashed"],
    ["A complaint point (its literal and assumption together)", "#8a3434", "circle-dashed"],
    ["Support rule", "#5e6b47", "edge-solid"],
    ["Attack: a point → the point it contradicts (human-voted)", "#6e6257", "edge-dashed", "human"],
    ["Attack predicted by the LLM, not human-voted", "#4f6d9a", "edge-dashed", "llm"],
  ],
  new_aba: [
    ["Head claim (good_X / bad_X)", "#a8947e", "roundrect"],
    ["Assumption — no_evident_not_X (positive)", "#5e6b47", "circle-dashed"],
    ["Assumption — have_evident_X (negative)", "#8a3434", "circle-dashed"],
    ["Proposition / fact (positive literal)", "#5e6b47", "circle"],
    ["Proposition / fact (negative literal)", "#8a3434", "circle"],
    ["Support rule", "#5e6b47", "edge-solid"],
    ["Attack: a contrary literal → the assumption it contradicts (human-voted)", "#6e6257", "edge-dashed", "human"],
    ["Attack predicted by the LLM, not human-voted", "#4f6d9a", "edge-dashed", "llm"],
  ],
  aba: [
    ["Head claim (good_X / bad_X)", "#a8947e", "roundrect"],
    ["Assumption clone (positive)", "#5e6b47", "circle-dashed"],
    ["Assumption clone (negative)", "#8a3434", "circle-dashed"],
    ["Proposition / fact (positive literal)", "#5e6b47", "circle"],
    ["Proposition / fact (negative literal)", "#8a3434", "circle"],
    ["Original assumption (pre-split)", "#a8947e", "hexagon"],
    ["Support rule", "#5e6b47", "edge-solid"],
    ["Attack: a contrary literal → the clone it contradicts (human-voted)", "#6e6257", "edge-dashed", "human"],
    ["Attack predicted by the LLM, not human-voted", "#4f6d9a", "edge-dashed", "llm"],
    ["Clone-of (Translation Rule split)", "#a8947e", "edge-dotted"],
  ],
  aa: [
    ["Constructed argument (premise ⊢ conclusion)", "#7c8aa5", "circle"],
    ["Defeat (derived attack)", "#8a3434", "edge-solid"],
  ],
};


// --- plain-English summary -------------------------------------------------------------------
// The Graph Studio's LLM Summary card is for someone running a hotel, not reading a framework:
// everything below converts the picture into guest-review terms before it reaches either the
// offline text or the model prompt, so no framework vocabulary can leak into what they read.

/** "no_evident_not_clear_instruction#x" -> "clear instruction", "good_check-in" -> "good check-in". */
export function humanAtom(name) {
  return shortLabel(String(name))
    .replace(/^no_evident_not_/, "")
    .replace(/^have_evident_/, "")
    .replace(/_/g, " ")
    .trim();
}

function uniqueHuman(names, limit) {
  const seen = new Set();
  const out = [];
  for (const name of names) {
    const human = humanAtom(name);
    if (!human || seen.has(human)) continue;
    seen.add(human);
    out.push(human);
    if (limit && out.length >= limit) break;
  }
  return out;
}

/** Tallies the graph AS DRAWN — same build + head/focus filters the canvas uses — so a summary
 *  can never describe nodes that were filtered off the screen. */
export function graphFacts(data, view, focus, head, options = {}) {
  if (!data) return null;
  const elements = filterToFocus(
    filterToHead(buildElements(data, view, { ...options, head }), head), focus);
  const nodes = elements.filter((e) => !e.data.source);
  const edges = elements.filter((e) => e.data.source);

  const byRole = (role) => nodes.filter((n) => n.data.role === role);
  // Placeholder links are bookkeeping, not something a guest said, so they are never counted.
  const byKind = (kind) => edges.filter((e) => e.data.kind === kind && !e.data.placeholder).length;

  // In the Simple view a point IS its assumption node; everywhere else it is the mined literal.
  const simple = view === "simple";
  const praise = byRole(simple ? "assumption_pos" : "proposition_pos").length;
  const complaints = byRole(simple ? "assumption_neg" : "proposition_neg").length;
  const argumentNodes = byRole("argument").length;

  const counts = {
    elements: elements.length,
    nodes: nodes.length,
    verdicts: byRole("head").length,
    praise,
    complaints,
    // The AA view draws constructed arguments instead of the mined literals; either way this
    // is "how many separate points are on screen" to the reader.
    points: praise + complaints || argumentNodes,
  };

  const edgeCounts = {
    support: byKind("support"),
    predicted: edges.filter((e) => e.data.llm).length,
    // A contrary link and a defeat are both "these two cannot both stand" to a manager.
    conflict: byKind("contrary") + byKind("defeat"),
  };

  // Evidence counts ride on the node data (how many mined review spans produced that literal),
  // so the most-mentioned themes come straight off what is drawn.
  const topThemes = uniqueHuman(
    nodes
      .filter((n) => n.data.count)
      .sort((a, b) => b.data.count - a.data.count)
      .map((n) => n.data.full || n.data.id),
    5,
  );

  const unfiltered = !focus && !head;
  const extensionSizes = (data.extensions || []).map((ext) => ext.length);

  return {
    aspect: data.aspect,
    contraries: data.contraries,
    view,
    semantics: data.semantics,
    counts,
    edges: edgeCounts,
    headLabels: uniqueHuman(byRole("head").map((n) => n.data.full || n.data.id)),
    topThemes,
    focusLabel: focus ? humanAtom(focus) : "",
    headLabel: head ? humanAtom(head) : "",
    predictedBy: Object.values(data.contrary_source || {}).includes("llm")
      ? (data.pair_stats || []).find((s) => s.source === "llm")?.label || "an AI model"
      : "",
    // Extensions are computed on the whole framework, so quoting them next to a filtered
    // picture would describe something the reader cannot see.
    extensions: unfiltered
      ? { count: extensionSizes.length, largest: extensionSizes.length ? Math.max(...extensionSizes) : 0 }
      : { count: 0, largest: 0 },
  };
}

function countPhrase(n, singular, plural) {
  return `${n.toLocaleString()} ${n === 1 ? singular : plural}`;
}

/** Shown before (or instead of) any model output — so the card is never empty, including on a
 *  machine with no Ollama at all. */
export function offlineSummary(facts) {
  if (!facts) return "";
  const { counts, edges, headLabels, topThemes, extensions } = facts;
  const lines = [];

  const scope = facts.focusLabel
    ? `what guests said about ${facts.focusLabel}`
    : `what guests said about ${facts.aspect}`;
  lines.push(`- This picture shows ${scope}, and where those comments back each other up or clash.`);

  if (counts.points) {
    // Filtering to one verdict often leaves zero of one kind — say nothing about it rather
    // than "0 are complaints".
    const parts = [];
    if (counts.praise) parts.push(countPhrase(counts.praise, "is praise", "are praise"));
    if (counts.complaints) parts.push(countPhrase(counts.complaints, "is a complaint", "are complaints"));
    const split = parts.length ? ` — ${parts.join(" and ")}` : "";
    lines.push(`- ${countPhrase(counts.points, "guest point is", "guest points are")} on screen${split}.`);
  }

  if (headLabels.length) {
    lines.push(`- They add up to the overall verdicts: ${headLabels.join(" and ")}.`);
  }

  lines.push(
    `- ${countPhrase(edges.support, "link shows", "links show")} a comment backing a verdict, and ` +
      `${countPhrase(edges.conflict, "link shows", "links show")} two comments that contradict each other.`,
  );

  if (edges.predicted) {
    lines.push(
      `- ${countPhrase(edges.predicted, "of those clashes was", "of those clashes were")} suggested by an AI model ` +
        `and not checked by a person, so treat them as leads rather than facts.`,
    );
  }

  if (topThemes.length) {
    lines.push(`- The themes guests raised most often: ${topThemes.join(", ")}.`);
  }

  if (extensions.count) {
    lines.push(
      `- Even with those disagreements, ${countPhrase(
        extensions.count,
        "set of comments holds",
        "different sets of comments hold",
      )} together at once.`,
    );
  }

  return lines.join("\n");
}

// --- merging near-identical points -----------------------------------------------------------
// Topics repeat themselves: check-in alone has easy_check-in, easy_check-in_at_3am,
// easy_self_check-in, easy_late_night_check-in… Each is its own node, and the picture is mostly
// the same point said several ways. Merging folds names that are within an edit distance of each
// other into one node, so the graph shows themes rather than phrasings.
//
// This is STRING distance, not meaning: it groups easy_check-in with easy_self_check-in, and it
// will never group it with quick_check-in. Grouping by meaning needs embeddings, not this.

const MERGEABLE = new Set(["proposition_pos", "proposition_neg", "assumption_pos", "assumption_neg"]);

// Thresholds are a share of the longer name: 0.8 only catches typo-level differences, 0.65 groups
// easy_check-in with easy_self_check-in / quick_check-in, and 0.55 starts pulling in points that
// merely share vocabulary. Named after what they do rather than the number.
export const MERGE_LEVELS = [
  { value: 0, label: "Off — every point" },
  { value: 0.8, label: "Near-duplicates" },
  { value: 0.65, label: "Similar wording" },
  { value: 0.55, label: "Loose" },
];

/** Levenshtein distance, abandoned as soon as it cannot come in at or under `max`. */
function editDistance(a, b, max) {
  if (a === b) return 0;
  if (Math.abs(a.length - b.length) > max) return max + 1;
  let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const row = [i];
    let best = i;
    for (let j = 1; j <= b.length; j++) {
      const cost = a.charCodeAt(i - 1) === b.charCodeAt(j - 1) ? 0 : 1;
      row[j] = Math.min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + cost);
      if (row[j] < best) best = row[j];
    }
    if (best > max) return max + 1;      // no cell in this row can still lead to a short path
    prev = row;
  }
  return prev[b.length];
}

/** 1 = identical, 0 = nothing in common. Length-normalised so long names are not penalised. */
export function nameSimilarity(a, b) {
  const [sa, sb] = [saidPart(a), saidPart(b)];
  const longest = Math.max(sa.length, sb.length);
  if (!longest) return 1;
  return 1 - editDistance(sa, sb, longest) / longest;
}

// Compare what the guest said, not the prefix bolted on to it: otherwise every assumption looks
// 70% similar to every other one purely because they all begin "no_evident_not_", and a threshold
// would mean something different in the Simple view than in the others.
const saidPart = (name) => {
  const id = String(name);
  for (const prefix of [POS_ASSUMPTION, NEG_ASSUMPTION]) {
    if (id.startsWith(prefix)) return id.slice(prefix.length);
  }
  return id;
};

const tokensOf = (name) => saidPart(name).toLowerCase().split(/[_\-\s]+/).filter(Boolean);

/** {id -> the id of the group it belongs to}, for ids that are within `threshold` of each other.
 *  Single-linkage: a chain of near-identical names ends up as one group. Only names that share a
 *  word are compared at all, which keeps this usable on the 600-literal topics. */
export function similarityGroups(names, threshold) {
  const parent = new Map(names.map((n) => [n, n]));
  const find = (x) => {
    let r = x;
    while (parent.get(r) !== r) r = parent.get(r);
    while (parent.get(x) !== r) { const next = parent.get(x); parent.set(x, r); x = next; }
    return r;
  };
  const union = (a, b) => {
    const [ra, rb] = [find(a), find(b)];
    if (ra === rb) return;
    // The shorter name wins, so a group is named after its most general member.
    const keep = ra.length !== rb.length ? (ra.length < rb.length ? ra : rb) : (ra < rb ? ra : rb);
    parent.set(keep === ra ? rb : ra, keep);
  };

  const byToken = new Map();
  for (const name of names) {
    for (const t of new Set(tokensOf(name))) {
      if (!byToken.has(t)) byToken.set(t, []);
      byToken.get(t).push(name);
    }
  }
  const compared = new Set();
  for (const bucket of byToken.values()) {
    if (bucket.length > 400) continue;            // a token everything shares says nothing
    for (let i = 0; i < bucket.length; i++) {
      for (let j = i + 1; j < bucket.length; j++) {
        const [a, b] = bucket[i] < bucket[j] ? [bucket[i], bucket[j]] : [bucket[j], bucket[i]];
        const pair = `${a}\u0000${b}`;
        if (compared.has(pair)) continue;
        compared.add(pair);
        const [sa, sb] = [saidPart(a), saidPart(b)];
        const longest = Math.max(sa.length, sb.length);
        const budget = Math.floor((1 - threshold) * longest);
        if (Math.abs(sa.length - sb.length) > budget) continue;
        if (editDistance(sa, sb, budget) <= budget) union(a, b);
      }
    }
  }
  const groups = new Map();
  for (const name of names) groups.set(name, find(name));
  return groups;
}

/** Collapses each group of near-identical nodes into one. Runs LAST, after the head and focus
 *  filters, so those still work on the names the rest of the app uses. */
export function mergeSimilar(elements, threshold) {
  if (!threshold) return elements;
  const nodes = elements.filter((e) => !e.data.source);
  const edges = elements.filter((e) => e.data.source);

  // Only same-role names are ever compared, so a literal is never folded into an assumption.
  const groupOf = new Map();
  const members = new Map();
  for (const role of MERGEABLE) {
    const names = nodes.filter((n) => n.data.role === role).map((n) => String(n.data.id));
    if (names.length < 2) continue;
    for (const [name, group] of similarityGroups(names, threshold)) {
      groupOf.set(name, group);
      if (!members.has(group)) members.set(group, []);
      members.get(group).push(name);
    }
  }
  const to = (id) => groupOf.get(id) || id;

  const merged = new Map();
  for (const n of nodes) {
    const id = to(n.data.id);
    const group = members.get(id) || [n.data.id];
    const existing = merged.get(id);
    if (!existing) {
      merged.set(id, {
        data: {
          ...n.data,
          id,
          label: group.length > 1 ? `${shortLabel(id)} +${group.length - 1}` : n.data.label,
          full: group.length > 1 ? group.slice().sort().join(", ") : n.data.full,
          count: n.data.count || null,
          merged: group.length > 1 ? group.length : undefined,
        },
      });
    } else if (n.data.count) {
      existing.data.count = (existing.data.count || 0) + n.data.count;
    }
  }

  const seen = new Set();
  const keptEdges = [];
  for (const e of edges) {
    const source = to(e.data.source);
    const target = to(e.data.target);
    if (source === target) continue;               // the group swallowed this link
    const key = `${e.data.kind}|${source}|${target}`;
    if (seen.has(key)) continue;
    seen.add(key);
    keptEdges.push({ data: { ...e.data, id: `m_${key}`, source, target } });
  }
  return [...merged.values(), ...keptEdges];
}

// --- what holds up -----------------------------------------------------------------------------

/** The accepted assumptions of the WHOLE framework, the way py_arg's own ABA visualiser reports
 *  them: one flat list, not split by claim and not filtered by the head currently on screen.
 *
 *  The framework py_arg evaluates is the reduced one, where the Translation Rule has split each
 *  assumption into one clone per contrary. An original assumption is accepted exactly when every
 *  one of its clones is in the extension -- a single surviving line of defence is not enough,
 *  because the clones together stand for the one assumption. Reading it off the extension keeps
 *  this identical to what py_arg computed rather than re-deriving acceptance here.
 */
export function acceptedAssumptions(data) {
  if (!data || !data.extensions || !data.extensions.length) return null;
  const multi = (data.new_aba && data.new_aba.multi_contraries) || {};
  const assumptions = (data.new_aba && data.new_aba.assumptions) || [];
  if (!assumptions.length) return null;

  const clonesIn = new Map();
  for (const member of data.extensions[0]) {
    const original = String(member).split("#")[0];
    clonesIn.set(original, (clonesIn.get(original) || 0) + 1);
  }

  const accepted = [];
  const rejected = [];
  for (const a of assumptions) {
    const expected = (multi[a] || []).length || 1;
    ((clonesIn.get(a) || 0) >= expected ? accepted : rejected).push(a);
  }
  const byName = (x, y) => x.localeCompare(y);
  return {
    accepted: accepted.sort(byName),
    rejected: rejected.sort(byName),
    total: assumptions.length,
    semantics: data.semantics || "",
  };
}

// --- focus ---------------------------------------------------------------------------------
// mu_team's studio draws one graph per selected supporting literal rather than the whole
// framework at once. Same idea here, but applied to whichever view is showing: pick any
// assumption or contrary atom and the canvas is cut down to its neighbourhood.

/** Atoms the user can focus on, grouped so the picker reads sensibly. */
/** True when `atom` is a premise of a rule for `head` — i.e. that literal backs that claim. */
export function backsHead(data, atom, head) {
  if (!data || !atom || !head) return true;
  return (data.new_aba.rules || []).some((r) => r.head === head && r.body.includes(atom));
}

/** The focus picker's contents. With a `head` claim chosen, only the literals that back THAT
 *  claim are listed: the other side is not on screen, so offering it would focus on nothing. */
export function focusOptions(data, head = "") {
  if (!data) return [];
  // The picker lists PROPOSITIONS — the plain mined literals, exactly what the Review page's
  // "Supporting" column shows. The assumptions (no_evident_not_* / have_evident_*) are derived
  // from them and would only clutter the list; focusing a proposition pulls its assumption in
  // anyway, because they are co-premises of the same rule.
  const roles = data.new_aba.atom_roles || {};
  const backing = head
    ? new Set((data.new_aba.rules || []).filter((r) => r.head === head).flatMap((r) => r.body))
    : null;
  const byRole = (role) =>
    Object.entries(roles)
      .filter(([atom, r]) => r === role && !atom.includes("#") && (!backing || backing.has(atom)))
      .map(([atom]) => atom)
      .sort();

  return [
    { group: "Positive review", atoms: byRole("proposition_pos") },
    { group: "Negative review", atoms: byRole("proposition_neg") },
  ].filter((g) => g.atoms.length);
}

/** Keeps only what sits under one head claim: the head, everything that supports it, and the
 *  attackers / clone origins of those supporters. Lets you look at just the good_X side or
 *  just the bad_X side instead of both at once. */
export function filterToHead(elements, head) {
  if (!head) return elements;

  const nodes = elements.filter((e) => !e.data.source);
  const edges = elements.filter((e) => e.data.source);
  if (!nodes.some((n) => n.data.id === head)) return elements;

  const keep = new Set([head]);
  for (const e of edges) {
    if (e.data.kind === "support" && e.data.target === head) keep.add(e.data.source);
  }
  // then what attacks those supporters, and their pre-split originals
  const supporters = new Set(keep);
  for (const e of edges) {
    if (e.data.kind === "contrary" && supporters.has(e.data.target)) keep.add(e.data.source);
    if (e.data.kind === "clone_of" && supporters.has(e.data.source)) keep.add(e.data.target);
  }
  return [
    ...nodes.filter((n) => keep.has(n.data.id)),
    ...edges.filter((e) => keep.has(e.data.source) && keep.has(e.data.target)),
  ];
}

/** Cuts the graph down to the focused atom and everything directly attached to it.
 *
 *  Expansion is a SINGLE hop out of the seed, never cumulative. Both head atoms (good_X /
 *  bad_X, which every assumption supports) and shared contrary atoms (one attacker can attack
 *  many assumptions) are hubs — a second hop through either drags the whole framework back in.
 *  The seed already contains the focused atom's clones, so one hop is enough to reach their
 *  contraries and head in the reduced view too. */
export function filterToFocus(elements, focus) {
  if (!focus) return elements;

  const nodes = elements.filter((e) => !e.data.source);
  const edges = elements.filter((e) => e.data.source);

  // An atom can appear under several ids (its clones, or AA arguments built on it), so seed
  // with every node that refers to the focused atom. Matching is on WHOLE atoms, never a
  // substring: "easy_check-in" must not drag in "easy_check-in_at_3am".
  const refersToFocus = (n) => {
    const id = String(n.data.id);
    const full = String(n.data.full || id);
    if (id === focus || full === focus) return true;
    if (full.startsWith(`${focus}#`)) return true;              // a clone of the focused atom
    // A Simple-view node is keyed by its assumption but stands for its literal, which is what
    // the focus picker names. Without this the focus silently did nothing in that view.
    if (String(n.data.literal || "") === focus) return true;
    // AA arguments carry their premise as a comma-separated list of atoms.
    const premise = String(n.data.premise || "");
    if (!premise) return false;
    return premise
      .split(",")
      .map((a) => a.trim())
      .some((a) => a === focus || a.startsWith(`${focus}#`));
  };

  // A literal's own assumption (and its clones) belong to it too — in the AA view they are the
  // only arguments that mention it, since arguments are built from assumptions.
  const ownAssumptions = [`no_evident_not_${focus}`, `have_evident_${focus}`];
  const refersToOwnAssumption = (n) => {
    const full = String(n.data.full || n.data.id);
    const premise = String(n.data.premise || "").split(",").map((a) => a.trim());
    return ownAssumptions.some((a) =>
      full === a || full.startsWith(`${a}#`) || premise.some((p) => p === a || p.startsWith(`${a}#`)));
  };
  const seed = new Set(nodes.filter((n) => refersToFocus(n) || refersToOwnAssumption(n)).map((n) => n.data.id));
  if (!seed.size) return elements;

  const keep = new Set(seed);
  for (const e of edges) {
    if (seed.has(e.data.source)) keep.add(e.data.target);
    if (seed.has(e.data.target)) keep.add(e.data.source);
  }

  // A claim is supported by a COMPOUND rule (the observed fact together with the assumption),
  // so those two premises belong in the same picture. They are only linked through the head,
  // which is never expanded through — pull in co-premises of the same rule explicitly.
  const seededRules = new Set(
    edges.filter((e) => e.data.rule && seed.has(e.data.source)).map((e) => e.data.rule)
  );
  const coPremises = new Set();
  for (const e of edges) {
    if (e.data.rule && seededRules.has(e.data.rule)) {
      keep.add(e.data.source);
      keep.add(e.data.target);
      if (!seed.has(e.data.source)) coPremises.add(e.data.source);
    }
  }

  // Focusing a proposition should show what can knock its assumption out — otherwise the
  // interesting half of the picture (the attackers) is one hop too far away.
  for (const e of edges) {
    if (e.data.kind === "contrary" && coPremises.has(e.data.target)) keep.add(e.data.source);
  }

  // In the reduced view the co-premise is a CLONE. Pull in its pre-split original so the
  // Translation Rule's split is visible here too — without this the reduced view of a
  // single-contrary assumption looks identical to the pre-translation one.
  for (const e of edges) {
    if (e.data.kind === "clone_of" && keep.has(e.data.source)) keep.add(e.data.target);
  }

  return [
    ...nodes.filter((n) => keep.has(n.data.id)),
    ...edges.filter((e) => keep.has(e.data.source) && keep.has(e.data.target)),
  ];
}
