import { useEffect, useImperativeHandle, useRef, forwardRef } from "react";
import cytoscape from "cytoscape";
import fcose from "cytoscape-fcose";
import dagre from "cytoscape-dagre";
import { buildElements, filterToFocus, filterToHead, mergeSimilar } from "../graphElements.js";

cytoscape.use(fcose);
cytoscape.use(dagre);

// Palette aligned with the site's warm tokens (tailwind.config.js).
const ROLE_COLORS = {
  head: "#a8947e",
  assumption_pos: "#5e6b47",
  assumption_neg: "#8a3434",
  placeholder: "#6e6257",
  clone_origin: "#a8947e",
};

const STYLESHEET = [
  {
    selector: "node",
    style: {
      "background-color": "#7c8aa5",
      label: "data(label)",
      color: "#2e241d",
      "font-size": 10,
      "font-family": "Inter, sans-serif",
      "text-wrap": "wrap",
      "text-max-width": "130px",
      "text-background-color": "#faf7f2",
      "text-background-opacity": 0.75,
      "text-background-padding": 2,
      "text-background-shape": "roundrectangle",
      width: 26,
      height: 26,
      "text-valign": "bottom",
      "text-margin-y": 5,
      // Below this rendered font size labels are dropped entirely, so a zoomed-out view shows
      // clean structure and the names fade in as you zoom into a region.
      "min-zoomed-font-size": 6,
    },
  },
  {
    selector: "node[role='head']",
    style: { "background-color": ROLE_COLORS.head, shape: "round-rectangle", width: 40, height: 40, "font-size": 12 },
  },
  // Following the previous team's visual grammar: dashed border = assumption (defeasible),
  // solid border = proposition/claim (an observed fact). Green = positive side, red = negative.
  {
    selector: "node[role='assumption_pos']",
    style: { "background-color": ROLE_COLORS.assumption_pos, "border-width": 2,
             "border-color": "#2f5238", "border-style": "dashed" },
  },
  {
    selector: "node[role='assumption_neg']",
    style: { "background-color": ROLE_COLORS.assumption_neg, "border-width": 2,
             "border-color": "#5f1f1f", "border-style": "dashed" },
  },
  {
    selector: "node[role='proposition_pos']",
    style: { "background-color": ROLE_COLORS.assumption_pos, "border-width": 2,
             "border-color": "#2f5238", "border-style": "solid" },
  },
  {
    selector: "node[role='proposition_neg']",
    style: { "background-color": ROLE_COLORS.assumption_neg, "border-width": 2,
             "border-color": "#5f1f1f", "border-style": "solid" },
  },
  {
    selector: "node[role='placeholder']",
    style: {
      "background-color": ROLE_COLORS.placeholder,
      shape: "diamond",
      "border-width": 2,
      "border-color": "#d6cabc",
    },
  },
  {
    selector: "node[role='clone_origin']",
    style: {
      "background-color": "#faf7f2",
      shape: "hexagon",
      "border-width": 2,
      "border-color": "#a8947e",
      width: 22,
      height: 22,
      "font-style": "italic",
    },
  },
  { selector: "node[role='argument']", style: { "background-color": "#7c8aa5", shape: "ellipse", width: 30, height: 30 } },
  {
    selector: "node:selected",
    style: { "border-width": 4, "border-color": "#2e241d", "text-background-opacity": 1, "z-index": 99 },
  },
  {
    selector: "edge",
    style: {
      width: 1.5,
      "line-color": "#b6a897",
      "target-arrow-color": "#b6a897",
      "target-arrow-shape": "triangle",
      "arrow-scale": 0.9,
      "curve-style": "bezier",
      opacity: 0.85,
    },
  },
  { selector: "edge[kind='support']", style: { "line-color": "#5e6b47", "target-arrow-color": "#5e6b47" } },
  {
    selector: "edge[kind='contrary']",
    style: {
      "line-color": "#6e6257",
      "target-arrow-color": "#6e6257",
      "line-style": "dashed",
      // A triangle, not the old diamond: at this scale a diamond reads as a small square and
      // gives no sense of which way the attack runs. Scaled up a little for the same reason.
      "target-arrow-shape": "triangle",
      "arrow-scale": 1.3,
    },
  },
  // Predicted, not voted: a different colour so the two can never be mistaken for each other.
  { selector: "edge[?llm]", style: { "line-color": "#4f6d9a", "target-arrow-color": "#4f6d9a" } },
  {
    selector: "edge[kind='clone_of']",
    style: { "line-color": "#a8947e", "target-arrow-shape": "none", "line-style": "dotted", width: 1, opacity: 0.45 },
  },
  { selector: "edge[kind='defeat']", style: { "line-color": "#8a3434", "target-arrow-color": "#8a3434" } },

  // "Evidence count" badge — a small circle sitting above and slightly overlapping its node,
  // synthesized in syncBadges() below (Cytoscape nodes only support one label each, so this is
  // a real second node rather than a style trick).
  {
    selector: ".count-badge",
    style: {
      shape: "ellipse",
      width: 15,
      height: 15,
      "background-color": "#faf7f2",
      "border-width": 1.5,
      "border-color": "#a8947e",
      color: "#2e241d",
      label: "data(label)",
      "font-size": 8,
      "font-family": "Inter, sans-serif",
      "text-valign": "center",
      "text-halign": "center",
      "text-margin-y": 0,
      "text-background-opacity": 0,
      "z-index": 500,
      events: "no",
    },
  },

  // Thousands of edges drawn at full strength merge into a solid wall; on big graphs they are
  // faint until a node is hovered (the hover rules below still bring its own edges forward).
  { selector: "edge.dense", style: { width: 0.6, opacity: 0.22 } },

  // --- hover highlighting (declared last so it overrides the role/kind rules above) --------
  { selector: ".faded", style: { opacity: 0.12, "text-opacity": 0.06 } },
  {
    selector: ".hl-edge",
    style: { width: 3, opacity: 1, "text-opacity": 1, "z-index": 900 },
  },
  // Name the relation the way CLAR 2025 does, but only on the edges being hovered: hundreds of
  // permanent edge labels would bury the graph.
  {
    selector: "edge[kind='contrary'].hl-edge",
    style: {
      label: (e) => (e.data("llm") ? "attacks (LLM)" : "attacks"),
      "font-size": 9,
      "font-family": "Inter, sans-serif",
      color: "#6e6257",
      "text-background-color": "#faf7f2",
      "text-background-opacity": 0.9,
      "text-background-padding": 2,
      "text-rotation": "autorotate",
    },
  },
  {
    selector: ".hl-neighbour",
    style: { opacity: 1, "text-opacity": 1, "z-index": 950 },
  },
  {
    selector: ".hl-focus",
    style: {
      opacity: 1,
      "text-opacity": 1,
      "border-width": 4,
      "border-color": "#2e241d",
      "text-background-opacity": 1,
      "z-index": 1000,
    },
  },
];

// Both ABA views are a STAR: every assumption clone supports one of just two head atoms
// (good_X / bad_X). Force-directed and concentric layouts both collapse that into an
// unreadable hub-and-spoke ring. The structure is really a small number of LAYERS —
//     attacking literal  ->  assumption (clone)  ->  head claim
// so a layered (dagre) left-to-right layout draws it as readable columns instead: you pan
// down the list of assumptions rather than squinting at a starburst.
function abaLayout(nodeCount) {
  const big = nodeCount > 400;
  return {
    name: "dagre",
    animate: false,
    fit: true,
    padding: 40,
    rankDir: "LR",
    nodeDimensionsIncludeLabels: true,
    // rank = column gap, node = gap between siblings stacked in a column.
    rankSep: big ? 260 : 320,
    nodeSep: big ? 14 : 26,
    edgeSep: 12,
    ranker: "tight-tree",
  };
}

// The AA view is a genuine attack network, not a star, so force-directed suits it there.
function aaLayout(nodeCount) {
  const big = nodeCount > 400;
  return {
    name: "fcose",
    quality: big ? "default" : "proof",
    animate: false,
    randomize: true,
    nodeDimensionsIncludeLabels: true,
    padding: 60,
    nodeRepulsion: big ? 60000 : 30000,
    idealEdgeLength: big ? 260 : 190,
    edgeElasticity: 0.3,
    gravity: big ? 0.06 : 0.2,
    gravityRange: 3.8,
    numIter: big ? 3500 : 2500,
    tile: true,
    tilingPaddingVertical: 40,
    tilingPaddingHorizontal: 40,
  };
}

// Fit the whole graph, but never zoom out past the point where labels disappear — a big
// framework opens centred and readable, and the user pans (or hits Fit) for the overview.
const MIN_READABLE_ZOOM = 0.55;

function fitReadable(cy) {
  cy.fit(undefined, 50);
  if (cy.zoom() < MIN_READABLE_ZOOM) {
    cy.zoom(MIN_READABLE_ZOOM);
    cy.center();
  }
}

// dagre's crossing minimisation is what makes it readable, and what makes it slow: measured
// ~1.2 s at 3k elements and ~10 s at 11k, with the tab frozen meanwhile. Up to this size the
// wait is worth the picture; above it the graph gets a linear-time layout with the same columns.
const DAGRE_MAX_ELEMENTS = 4000;
const COLUMN_GAP = 380;
const ROW_GAP = 46;   // node (26px) + its label underneath, without touching the next row

const isAssumption = (n) => ["assumption_pos", "assumption_neg"].includes(n.data("role"));
const isLiteral = (n) => ["proposition_pos", "proposition_neg"].includes(n.data("role"));
const baseName = (n) => String(n.data("full") || n.id()).split("#")[0];

// The column layout keeps dagre's picture, and reads the way the arrows run: the literals that
// attack something on the left, then each assumption with its own literal right above it (the two
// premises of one rule), then the claims, and (reduced view) the pre-split originals last.
function columnLayout(cy, view) {
  const nodes = cy.nodes().not(".count-badge");
  if (view === "aa") {
    twoColumnAa(cy, nodes);
    return;
  }

  const attackers = new Set(cy.edges("[kind = 'contrary']").map((e) => e.source().id()));
  // A literal's partner: the assumption (or first clone) it shares a support rule with.
  const partner = new Map();
  const byRule = new Map();
  cy.edges("[kind = 'support']").forEach((e) => {
    const rule = e.data("rule");
    if (!rule) return;
    if (!byRule.has(rule)) byRule.set(rule, []);
    byRule.get(rule).push(e.source());
  });
  for (const members of byRule.values()) {
    const lit = members.find(isLiteral);
    const asm = members.find(isAssumption);
    if (lit && asm && !partner.has(lit.id())) partner.set(lit.id(), asm);
  }

  const column = (n) => {
    const role = n.data("role");
    if (role === "head") return 2;
    if (role === "clone_origin") return 3;
    if (attackers.has(n.id())) return 0;        // the literals the dashed arrows start from
    return 1;   // assumptions, their own literals, and any default contrary they point at
  };

  // Premise column (the one everything else is lined up against): praise side first, then
  // complaints; each assumption's clones together, and a paired literal above its assumption.
  const sideOf = (n) => (n.data("role") === "assumption_neg" || n.data("role") === "proposition_neg" ? 1 : 0);
  const middleKey = (n) => {
    const anchor = isLiteral(n) ? (partner.get(n.id()) || n) : n;
    return [sideOf(anchor), baseName(anchor), isLiteral(n) ? 0 : 1, String(n.data("full") || n.id())];
  };
  const compare = (a, b) => {
    for (let i = 0; i < a.length; i++) {
      if (a[i] < b[i]) return -1;
      if (a[i] > b[i]) return 1;
    }
    return 0;
  };

  const columns = new Map();
  nodes.forEach((n) => {
    const c = column(n);
    if (!columns.has(c)) columns.set(c, []);
    columns.get(c).push(n);
  });

  const row = new Map();
  const place = (list) => list.forEach((n, i) => row.set(n.id(), i - (list.length - 1) / 2));
  const premises = (columns.get(1) || [])
    .map((n) => ({ n, key: middleKey(n) }))
    .sort((a, b) => compare(a.key, b.key))
    .map((x) => x.n);
  place(premises);

  // Every other column follows the premise column (barycentre of its neighbours there).
  const barycentre = (n) => {
    const rows = n.neighborhood("node").filter((m) => row.has(m.id()) && column(m) === 1)
      .map((m) => row.get(m.id()));
    return rows.length ? rows.reduce((a, b) => a + b, 0) / rows.length : Infinity;
  };
  for (const c of [0, 2, 3]) {
    if (!columns.has(c)) continue;
    const list = columns.get(c)
      .map((n) => ({ n, key: barycentre(n) }))
      .sort((a, b) => a.key - b.key || String(a.n.id()).localeCompare(String(b.n.id())))
      .map((x) => x.n);
    place(list);
    if (c === 2 || c === 3) {
      // Short columns (the two claims) sit level with what they connect to.
      list.forEach((n) => {
        const key = barycentre(n);
        if (Number.isFinite(key)) row.set(n.id(), key);
      });
    }
  }

  cy.batch(() => {
    nodes.forEach((n) => n.position({ x: column(n) * COLUMN_GAP, y: (row.get(n.id()) || 0) * ROW_GAP }));
  });
}

// AA view: the attacking arguments (built on facts, no premise) left, the rest right, each
// attacker level with what it attacks.
function twoColumnAa(cy, nodes) {
  const right = nodes.filter((n) => n.data("premise")).sort((a, b) => String(a.id()).localeCompare(String(b.id())));
  const row = new Map();
  right.forEach((n, i) => row.set(n.id(), i - (right.length - 1) / 2));
  const left = nodes.filter((n) => !n.data("premise"))
    .map((n) => {
      const rows = n.outgoers("node").map((m) => row.get(m.id())).filter((v) => v !== undefined);
      return { n, key: rows.length ? rows.reduce((a, b) => a + b, 0) / rows.length : Infinity };
    })
    .sort((a, b) => a.key - b.key)
    .map((x) => x.n);
  left.forEach((n, i) => row.set(n.id(), i - (left.length - 1) / 2));
  cy.batch(() => {
    nodes.forEach((n) => n.position({ x: (n.data("premise") ? 1 : 0) * COLUMN_GAP, y: (row.get(n.id()) || 0) * ROW_GAP }));
  });
}

function runLayout(cy, view) {
  const nodeCount = cy.nodes().not(".count-badge").length;
  if (cy.elements().length > DAGRE_MAX_ELEMENTS) {
    columnLayout(cy, view);
    return;
  }
  cy.layout(view === "aa" ? aaLayout(nodeCount) : abaLayout(nodeCount)).run();
}

// Canvas shortcuts that only pay off on big graphs: hide edges and reuse a bitmap while
// panning / zooming. Small graphs keep full-quality interaction.
function tuneRenderer(cy) {
  const big = cy.elements().length > DAGRE_MAX_ELEMENTS;
  cy.edges().toggleClass("dense", big);
  const renderer = cy.renderer();
  if (renderer) {
    renderer.hideEdgesOnViewport = big;
    renderer.textureOnViewport = big;
  }
}

// A badge sits at the node's top-right corner, pulled in by this fraction of the node's
// half-width/half-height so it overlaps rather than just touching the edge. The vertical pull
// is smaller (closer to 1 = closer to the true corner) so the badge rides a bit higher than a
// symmetric corner placement would put it.
const BADGE_PULL_X = 0.75;
const BADGE_PULL_Y = 0.95;

function badgePosition(n) {
  const pos = n.position();
  return {
    x: pos.x + (n.width() / 2) * BADGE_PULL_X,
    y: pos.y - (n.height() / 2) * BADGE_PULL_Y,
  };
}

// Rebuilds the count-badge nodes from scratch, one per real node carrying a `count`, perched on
// that node's top-right corner so it reads as a small circle overlapping the big one.
function syncBadges(cy) {
  cy.nodes(".count-badge").remove();
  cy.nodes()
    .not(".count-badge")
    .forEach((n) => {
      const count = n.data("count");
      if (!count) return;
      cy.add({
        group: "nodes",
        data: { id: `${n.id()}__badge`, label: String(count) },
        position: badgePosition(n),
        classes: "count-badge",
        selectable: false,
        grabbable: false,
      });
    });
}

const GraphCanvas = forwardRef(function GraphCanvas(
  { data, view, focus, head, merge = 0, placeholders = false, suppressed = false, onNodeSelect },
  ref,
) {
  const containerRef = useRef(null);
  const cyRef = useRef(null);
  const selectRef = useRef(onNodeSelect);
  selectRef.current = onNodeSelect;
  const viewRef = useRef(view);
  viewRef.current = view;
  const cleanupRef = useRef(null);

  useImperativeHandle(ref, () => ({
    zoomIn: () => cyRef.current?.zoom(cyRef.current.zoom() * 1.25),
    zoomOut: () => cyRef.current?.zoom(cyRef.current.zoom() / 1.25),
    fit: () => cyRef.current?.fit(undefined, 50),
    // Cytoscape caches its container's pixel size, so a resize the browser didn't cause itself
    // (e.g. the containing panel toggling fullscreen) needs an explicit nudge.
    resize: () => {
      const cy = cyRef.current;
      if (!cy) return;
      cy.resize();
      fitReadable(cy);
    },
    relayout: () => {
      const cy = cyRef.current;
      if (!cy) return;
      runLayout(cy, viewRef.current);
      fitReadable(cy);
      syncBadges(cy);
    },
  }));

  useEffect(() => {
    const cy = cytoscape({
      container: containerRef.current,
      style: STYLESHEET,
      wheelSensitivity: 0.2,
    });
    cy.on("tap", "node", (evt) => {
      const d = evt.target.data();
      selectRef.current?.({ full: d.full || d.id, label: d.label, role: d.role, premise: d.premise });
    });
    cy.on("tap", (evt) => {
      if (evt.target === cy) selectRef.current?.(null);
    });

    // Hover: lift the node and everything it touches, dim the rest. With hundreds of
    // overlapping edges this is the only practical way to read one atom's connections.
    const clearHighlight = () => cy.elements().removeClass("faded hl-edge hl-neighbour hl-focus");
    cy.on("mouseover", "node", (evt) => {
      const node = evt.target;
      const edges = node.connectedEdges();
      const neighbourhood = edges.connectedNodes().union(node).union(edges);
      cy.elements().difference(neighbourhood).addClass("faded");
      edges.addClass("hl-edge");
      edges.connectedNodes().not(node).addClass("hl-neighbour");
      node.addClass("hl-focus");
    });
    cy.on("mouseout", "node", clearHighlight);
    // Keep each count-badge glued to its node's top edge when the node is dragged.
    cy.on("position", "node", (evt) => {
      const n = evt.target;
      if (n.hasClass("count-badge")) return;
      const badge = cy.getElementById(`${n.id()}__badge`);
      if (badge.nonempty()) {
        badge.position(badgePosition(n));
      }
    });
    // A pointer can leave the canvas without firing mouseout on the node it was over.
    containerRef.current?.addEventListener("pointerleave", clearHighlight);
    cleanupRef.current = () => containerRef.current?.removeEventListener("pointerleave", clearHighlight);
    cyRef.current = cy;
    return () => {
      cleanupRef.current?.();
      cleanupRef.current = null;
      cy.destroy();
      cyRef.current = null;
    };
  }, []);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !data) return;
    selectRef.current?.(null);
    cy.elements().removeClass("faded hl-edge hl-neighbour hl-focus");
    cy.elements().remove();
    if (suppressed) return;   // too big to draw readably; the page says so instead
    // Merging runs LAST: the head and focus filters work on the names the rest of the app uses.
    cy.add(mergeSimilar(
      filterToFocus(filterToHead(buildElements(data, view, { placeholders, head }), head), focus),
      merge,
    ));
    cy.resize();
    tuneRenderer(cy);
    runLayout(cy, view);
    fitReadable(cy);
    syncBadges(cy);
  }, [data, view, focus, head, merge, placeholders, suppressed]);

  return <div ref={containerRef} className="w-full h-full rounded-panel bg-elevated" />;
});

export default GraphCanvas;
