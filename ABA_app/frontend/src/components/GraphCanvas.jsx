import { useEffect, useImperativeHandle, useRef, forwardRef } from "react";
import cytoscape from "cytoscape";
import fcose from "cytoscape-fcose";
import dagre from "cytoscape-dagre";
import { buildElements, filterToFocus, filterToHead } from "../graphElements.js";

cytoscape.use(fcose);
cytoscape.use(dagre);

// Palette aligned with the site's warm tokens (tailwind.config.js).
const ROLE_COLORS = {
  head: "#a8947e",
  assumption_pos: "#5e6b47",
  assumption_neg: "#8a3434",
  contrary: "#6e6257",
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
    selector: "node[role='contrary']",
    style: {
      "background-color": ROLE_COLORS.contrary,
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
      "target-arrow-shape": "diamond",
    },
  },
  {
    selector: "edge[kind='clone_of']",
    style: { "line-color": "#a8947e", "target-arrow-shape": "none", "line-style": "dotted", width: 1, opacity: 0.45 },
  },
  { selector: "edge[kind='defeat']", style: { "line-color": "#8a3434", "target-arrow-color": "#8a3434" } },

  // --- hover highlighting (declared last so it overrides the role/kind rules above) --------
  { selector: ".faded", style: { opacity: 0.12, "text-opacity": 0.06 } },
  {
    selector: ".hl-edge",
    style: { width: 3, opacity: 1, "text-opacity": 1, "z-index": 900 },
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
//     contrary atom  <-  assumption clone  ->  head claim
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

function layoutFor(view, nodeCount) {
  return view === "aa" ? aaLayout(nodeCount) : abaLayout(nodeCount);
}

const GraphCanvas = forwardRef(function GraphCanvas({ data, view, focus, head, onNodeSelect }, ref) {
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
    relayout: () => {
      const cy = cyRef.current;
      if (!cy) return;
      cy.layout(layoutFor(viewRef.current, cy.nodes().length)).run();
      fitReadable(cy);
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
    cy.add(filterToFocus(filterToHead(buildElements(data, view), head), focus));
    cy.resize();
    cy.layout(layoutFor(view, cy.nodes().length)).run();
    fitReadable(cy);
  }, [data, view, focus, head]);

  return <div ref={containerRef} className="w-full h-full rounded-panel bg-elevated" />;
});

export default GraphCanvas;
