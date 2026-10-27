// Legend swatches that mirror the ACTUAL node/edge shapes drawn on the canvas, rather than a
// generic coloured square — so "diamond" in the legend means the diamond you see in the graph.
// The shape names here match the Cytoscape `shape` / edge styles in GraphCanvas.jsx.

export default function LegendIcon({ shape, color }) {
  const common = { width: 18, height: 18, viewBox: "0 0 20 20", className: "shrink-0" };

  switch (shape) {
    case "roundrect": // node[role='head'] -> round-rectangle
      return (
        <svg {...common}>
          <rect x="2" y="4" width="16" height="12" rx="3" fill={color} />
        </svg>
      );

    case "diamond": // node[role='contrary'] -> diamond, light border
      return (
        <svg {...common}>
          <polygon points="10,2 18,10 10,18 2,10" fill={color} stroke="#d6cabc" strokeWidth="1.5" />
        </svg>
      );

    case "hexagon": // node[role='clone_origin'] -> hexagon, hollow with accent border
      return (
        <svg {...common}>
          <polygon points="10,2 17,6 17,14 10,18 3,14 3,6" fill="#faf7f2" stroke={color} strokeWidth="2" />
        </svg>
      );

    case "circle-dashed": // assumptions -> dashed border, per the previous team's grammar
      return (
        <svg {...common}>
          <circle cx="10" cy="10" r="7" fill={color} stroke="#2e241d" strokeWidth="1.5"
                  strokeDasharray="3 2" opacity="0.95" />
        </svg>
      );

    case "circle": // propositions / arguments -> solid ellipse
    default:
      return (
        <svg {...common}>
          <circle cx="10" cy="10" r="7" fill={color} />
        </svg>
      );

    case "edge-solid": // support / defeat edges, with the arrowhead the canvas draws
      return (
        <svg {...common}>
          <line x1="1" y1="10" x2="13" y2="10" stroke={color} strokeWidth="2" />
          <polygon points="13,6.5 19,10 13,13.5" fill={color} />
        </svg>
      );

    case "edge-dashed": // C*(A) contrary edges -> dashed with a diamond head
      return (
        <svg {...common}>
          <line x1="1" y1="10" x2="12" y2="10" stroke={color} strokeWidth="2" strokeDasharray="3 2" />
          <polygon points="12,7 16,10 12,13 8,10" fill={color} />
        </svg>
      );

    case "edge-dotted": // clone-of edges -> dotted, no arrowhead
      return (
        <svg {...common}>
          <line x1="1" y1="10" x2="19" y2="10" stroke={color} strokeWidth="2"
                strokeDasharray="1 3" strokeLinecap="round" opacity="0.7" />
        </svg>
      );
  }
}
