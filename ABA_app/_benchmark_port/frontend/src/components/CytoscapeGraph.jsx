import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import cytoscape from "cytoscape";
import { buildDisplayGraphModel, buildLayoutRows } from "../graph/displayModel.js";

// Cytoscape rendering of the ABA graph, replacing mu_team's hand-rolled SVG renderer.
//
// What is preserved from theirs: which nodes and edges are shown (displayModel.js is a port of
// their buildDisplayGraphModel), the layered row ordering, and the visual vocabulary — grey
// claim, green supporting reason, red opposite view, dashed border = assumption, solid = claim
// or reason, evidence count badge, red curved attack edges.
//
// What changes: the drawing itself is Cytoscape's, so node shapes and edge routing will not be
// pixel-identical to theirs. The graph DATA behind it is verified identical — see
// benchmark/compare_http.py.

const COLORS = {
  claimBg: "#c8ccd8", claimBorder: "#8b93a7",
  supportBg: "#9fe3b8", supportBorder: "#3f9c62",
  opposeBg: "#f6c3c3", opposeBorder: "#b45252",
  text: "#2e241d",
  supportEdge: "#6e6257",
  attackEdge: "#b03636",
};

const STYLESHEET = [
  {
    selector: "node",
    style: {
      label: "data(label)",
      "font-size": 11,
      "font-family": "Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, Arial",
      color: COLORS.text,
      "text-valign": "center",
      "text-halign": "center",
      "text-wrap": "wrap",
      "text-max-width": "data(w)",
      shape: "round-rectangle",
      // Explicit sizes computed in JS rather than width/height: "label". The auto-sizing form
      // left some nodes reporting visible() === false and never painting, even with display,
      // visibility and opacity all normal.
      width: "data(w)",
      height: "data(h)",
      "border-width": 2,
      "background-color": COLORS.supportBg,
      "border-color": COLORS.supportBorder,
    },
  },
  // Their vocabulary: claim = "Main Opinion", proposition = "Supporting Reason",
  // proposition on the opposing side = "Opposite View".
  { selector: "node[type='claim']",
    style: { "background-color": COLORS.claimBg, "border-color": COLORS.claimBorder, "font-weight": "bold" } },
  { selector: "node[side='bad']",
    style: { "background-color": COLORS.opposeBg, "border-color": COLORS.opposeBorder } },
  // Border dashed = assumption, solid = claim/reason.
  { selector: "node[type='assumption']", style: { "border-style": "dashed" } },
  { selector: "node[?isFocus]", style: { "border-width": 4 } },
  {
    selector: "edge",
    style: {
      "curve-style": "bezier",
      width: 1.6,
      "line-color": COLORS.supportEdge,
      "target-arrow-color": COLORS.supportEdge,
      "target-arrow-shape": "triangle",
      "arrow-scale": 0.9,
      opacity: 0.9,
    },
  },
  {
    selector: "edge[type='attack']",
    style: {
      "line-color": COLORS.attackEdge,
      "target-arrow-color": COLORS.attackEdge,
      "curve-style": "unbundled-bezier",
      "control-point-distances": [-26],
      "control-point-weights": [0.5],
      width: 2,
    },
  },
  { selector: "node.count-badge", style: {
      shape: "ellipse", width: 18, height: 18, "background-color": "#2e241d", color: "#fff",
      "font-size": 9, "border-width": 0, padding: "0px", "text-max-width": "18px" } },
];

// Their layout is row-based (one row per framework level), centred, top to bottom. Reproduced
// here as explicit preset positions rather than a force-directed layout, so the graph still
// reads as layers.
const ROW_GAP = 130;
const MIN_NODE_GAP = 40;
const CHAR_WIDTH = 6.4;

function estimateWidth(label) {
  return Math.min(190, Math.max(70, String(label || "").length * CHAR_WIDTH + 28));
}

// Labels wrap inside the box, so tall enough for however many lines the width implies.
function nodeHeight(label) {
  const width = estimateWidth(label);
  const perLine = Math.max(1, Math.floor((width - 20) / CHAR_WIDTH));
  const lines = Math.max(1, Math.ceil(String(label || "").length / perLine));
  return 22 + (lines - 1) * 13;
}

// Their splitRowIntoChunks: a framework level can hold hundreds of atoms, and one row that
// wide is unreadable, so each level is wrapped into lines of at most 8.
const MAX_PER_ROW = 8;

function chunkRows(rows) {
  const out = [];
  for (const row of rows) {
    for (let i = 0; i < row.length; i += MAX_PER_ROW) out.push(row.slice(i, i + MAX_PER_ROW));
  }
  return out;
}

function presetPositions(rows) {
  const positions = new Map();
  const rowWidths = rows.map((row) =>
    row.reduce((sum, n) => sum + estimateWidth(n.label), 0) + MIN_NODE_GAP * Math.max(0, row.length - 1)
  );
  const widest = Math.max(600, ...rowWidths);

  rows.forEach((row, rowIndex) => {
    let x = (widest - rowWidths[rowIndex]) / 2;
    const y = 80 + rowIndex * ROW_GAP;
    for (const node of row) {
      const w = estimateWidth(node.label);
      positions.set(node.id, { x: x + w / 2, y });
      x += w + MIN_NODE_GAP;
    }
  });
  return positions;
}

const CytoscapeGraph = forwardRef(function CytoscapeGraph({ graph, supporting, layerMode }, ref) {
  const containerRef = useRef(null);
  const cyRef = useRef(null);

  useImperativeHandle(ref, () => ({
    zoomIn: () => cyRef.current?.zoom({ level: cyRef.current.zoom() * 1.15,
                                        renderedPosition: centre(cyRef.current) }),
    zoomOut: () => cyRef.current?.zoom({ level: cyRef.current.zoom() / 1.15,
                                         renderedPosition: centre(cyRef.current) }),
    fit: () => cyRef.current?.fit(undefined, 40),
  }));

  useEffect(() => {
    const cy = cytoscape({
      container: containerRef.current,
      style: STYLESHEET,
      wheelSensitivity: 0.2,
      minZoom: 0.15,
      maxZoom: 3,
    });
    cyRef.current = cy;

    // The container is often still 0x0 when this effect runs (its height comes from CSS that
    // has not been applied yet). Cytoscape caches the viewport size at construction, so without
    // this it lays out and fits against a zero-sized viewport and only part of the graph ever
    // gets painted. Re-measure whenever the panel resizes, including the first time it gains
    // a real size.
    const observer = new ResizeObserver(() => {
      cy.resize();
      if (cy.elements().length) cy.fit(undefined, 40);
    });
    if (containerRef.current) observer.observe(containerRef.current);

    return () => { observer.disconnect(); cy.destroy(); cyRef.current = null; };
  }, []);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.elements().remove();
    if (!graph) return;

    const model = buildDisplayGraphModel(graph, { supporting, layerMode });
    const rows = chunkRows(buildLayoutRows(model));
    const positions = presetPositions(rows);
    const visible = new Set(model.nodes.map((n) => n.id));

    cy.add(model.nodes.map((n) => ({
      data: {
        id: n.id,
        label: n.count != null && Number(n.count) > 0 ? `${n.label}  (${Number(n.count)})` : n.label,
        type: n.type,
        side: n.clusterSentiment,
        isFocus: !!n.isFocus,
        w: estimateWidth(n.label),
        h: nodeHeight(n.label),
      },
      position: positions.get(n.id) || { x: 0, y: 0 },
    })));
    cy.add(model.edges
      .filter((e) => visible.has(e.source) && visible.has(e.target))
      .map((e) => ({ data: { id: e.id, source: e.source, target: e.target, type: e.type } })));

    cy.resize();
    cy.layout({ name: "preset", fit: true, padding: 40 }).run();
    cy.fit(undefined, 40);
  }, [graph, supporting, layerMode]);

  return <div ref={containerRef} className="cy-canvas" />;
});

function centre(cy) {
  const { w, h } = cy.extent ? { w: cy.width(), h: cy.height() } : { w: 0, h: 0 };
  return { x: w / 2, y: h / 2 };
}

export default CytoscapeGraph;
