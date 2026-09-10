// Turns one /api/graph response into Cytoscape elements for whichever of the three views is
// selected. Ported from the previous vanilla graph-render.js — same structure, same guarantees:
// each view is a 1:1 rendering of the corresponding framework the backend actually built, so
// the picture can never disagree with what py_arg evaluated.
//
//   new_aba -> C*: A -> P(L), pre-translation. Multi-contrary fan-out drawn directly.
//   aba     -> C: A -> L, post-translation (what py_arg consumes), clones + clone-of edges.
//   aa      -> the Dung instantiation from generate_af(): arguments and defeats.

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

function buildNewAba(data) {
  const { add, nodes } = nodeCollector(data.counts);
  const edges = [];
  for (const atom of data.new_aba.language) add(atom, data.new_aba.atom_roles[atom] || "atom");

  for (const rule of data.new_aba.rules) {
    for (const premise of rule.body) {
      edges.push({ data: { id: `${rule.id}_${premise}`, source: premise, target: rule.head, kind: "support", rule: rule.id } });
    }
  }
  // The whole point of the "New" formalization: one assumption, several distinct contraries.
  for (const [assumption, contraries] of Object.entries(data.new_aba.multi_contraries)) {
    for (const contrary of contraries) {
      edges.push({
        data: { id: `contrary_${assumption}_${contrary}`, source: assumption, target: contrary, kind: "contrary" },
      });
    }
  }
  return [...nodes, ...edges];
}

function buildReducedAba(data) {
  const { add, nodes } = nodeCollector(data.counts);
  const edges = [];
  for (const atom of data.aba.language) add(atom, data.aba.atom_roles[atom] || "atom");

  for (const rule of data.aba.rules) {
    for (const premise of rule.body) {
      edges.push({ data: { id: `${rule.id}_${premise}`, source: premise, target: rule.head, kind: "support", rule: rule.id } });
    }
  }
  for (const [clone, contrary] of Object.entries(data.aba.contraries)) {
    edges.push({ data: { id: `contrary_${clone}`, source: clone, target: contrary, kind: "contrary" } });
  }
  // Faint dotted edge back to the original assumption, so the Translation Rule's split stays
  // visible instead of disappearing into flat clone IDs.
  // An atom can be BOTH a pre-split original and some other clone's contrary (a Task-3 "Yes"
  // attacker is itself an assumption). When it is acting as a contrary here, keep the diamond
  // — that is the role it plays in this picture. Only atoms that are purely originals get the
  // hollow hexagon, so the legend's two entries stay distinguishable.
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

export function buildElements(data, view) {
  if (view === "new_aba") return buildNewAba(data);
  if (view === "aba") return buildReducedAba(data);
  return buildAa(data);
}

export const VIEWS = [
  { key: "new_aba", label: "New ABA (C*)", blurb: "Pre-translation. One assumption may carry several distinct contraries." },
  { key: "aba", label: "Reduced ABA (C)", blurb: "Post-translation. Each clone has exactly one contrary — this is what py_arg consumes." },
  { key: "aa", label: "AA", blurb: "Dung instantiation from generate_af(): constructed arguments and derived defeats." },
];

// [label, colour, shape] — shape names match the Cytoscape styles in GraphCanvas.jsx so the
// legend swatch looks like the thing it describes (see LegendIcon.jsx).
// [label, colour, shape] — shape names match the Cytoscape styles in GraphCanvas.jsx so the
// legend swatch looks like the thing it describes (see LegendIcon.jsx). Following the previous
// team's grammar: dashed = assumption (defeasible), solid = proposition (observed fact).
export const LEGENDS = {
  new_aba: [
    ["Head claim (good_X / bad_X)", "#a8947e", "roundrect"],
    ["Assumption — no_evident_not_X (positive)", "#5e6b47", "circle-dashed"],
    ["Assumption — have_evident_X (negative)", "#8a3434", "circle-dashed"],
    ["Proposition / fact (positive literal)", "#5e6b47", "circle"],
    ["Proposition / fact (negative literal)", "#8a3434", "circle"],
    ["Contrary atom (default, nothing derives it)", "#6e6257", "diamond"],
    ["Support rule", "#5e6b47", "edge-solid"],
    ["C*(A): assumption → one of its contraries", "#6e6257", "edge-dashed"],
  ],
  aba: [
    ["Head claim (good_X / bad_X)", "#a8947e", "roundrect"],
    ["Assumption clone (positive)", "#5e6b47", "circle-dashed"],
    ["Assumption clone (negative)", "#8a3434", "circle-dashed"],
    ["Proposition / fact (positive literal)", "#5e6b47", "circle"],
    ["Proposition / fact (negative literal)", "#8a3434", "circle"],
    ["Contrary atom (exactly one per clone)", "#6e6257", "diamond"],
    ["Original assumption (pre-split)", "#a8947e", "hexagon"],
    ["Support rule", "#5e6b47", "edge-solid"],
    ["Clone-of (Translation Rule split)", "#a8947e", "edge-dotted"],
  ],
  aa: [
    ["Constructed argument (premise ⊢ conclusion)", "#7c8aa5", "circle"],
    ["Defeat (derived attack)", "#8a3434", "edge-solid"],
  ],
};


// --- focus ---------------------------------------------------------------------------------
// mu_team's studio draws one graph per selected supporting literal rather than the whole
// framework at once. Same idea here, but applied to whichever view is showing: pick any
// assumption or contrary atom and the canvas is cut down to its neighbourhood.

/** Atoms the user can focus on, grouped so the picker reads sensibly. */
export function focusOptions(data) {
  if (!data) return [];
  // The picker lists PROPOSITIONS — the plain mined literals, exactly what the Review page's
  // "Supporting" column shows. The assumptions (no_evident_not_* / have_evident_*) are derived
  // from them and would only clutter the list; focusing a proposition pulls its assumption in
  // anyway, because they are co-premises of the same rule.
  const roles = data.new_aba.atom_roles || {};
  const byRole = (role) =>
    Object.entries(roles)
      .filter(([atom, r]) => r === role && !atom.includes("#"))
      .map(([atom]) => atom)
      .sort();

  return [
    { group: "Positive review", atoms: byRole("proposition_pos") },
    { group: "Negative review", atoms: byRole("proposition_neg") },
  ].filter((g) => g.atoms.length);
}

/** Keeps only what sits under one head claim: the head, everything that supports it, and the
 *  contraries / clone origins of those supporters. Lets you look at just the good_X side or
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
  // then whatever those supporters point at: their contraries and pre-split originals
  for (const e of edges) {
    if ((e.data.kind === "contrary" || e.data.kind === "clone_of") && keep.has(e.data.source)) {
      keep.add(e.data.target);
    }
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
    // AA arguments carry their premise as a comma-separated list of atoms.
    const premise = String(n.data.premise || "");
    if (!premise) return false;
    return premise
      .split(",")
      .map((a) => a.trim())
      .some((a) => a === focus || a.startsWith(`${focus}#`));
  };

  const seed = new Set(nodes.filter(refersToFocus).map((n) => n.data.id));
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
    if (e.data.kind === "contrary" && coPremises.has(e.data.source)) keep.add(e.data.target);
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
