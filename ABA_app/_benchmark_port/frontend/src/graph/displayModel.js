// Port of mu_team's buildDisplayGraphModel (frontend/assets/js/pyarg-page.js).
//
// The rendering library is now Cytoscape rather than their hand-rolled SVG, but WHICH nodes
// and edges are shown is kept identical to theirs — otherwise the two graphs would differ for
// two reasons at once and neither could be attributed. So this keeps their rules:
//
//   * defense-layer nodes are relabelled to the selected claim, and their claim/assumption
//     nodes are hidden (all of them in layer1);
//   * defense-layer propositions that duplicate a selected-claim proposition are merged into
//     the selected-claim node, with edges redirected;
//   * in layer1 the opposing claim/assumption nodes are hidden; in layer2 the opposing claim
//     is hidden if nothing in its cluster is involved in an attack;
//   * support edges through rule nodes are flattened to premise -> claim;
//   * attacks from the focused supporting literal to opposing-cluster assumptions are dropped.

export function buildDisplayGraphModel(rawGraph, { supporting, layerMode }) {
  const focusSupportingLabel = String(supporting || "").trim().toLowerCase();
  const meta = rawGraph?.meta || {};
  const selectedClaim = String(meta.selectedClaim || "").trim();
  const opposingClaim = String(meta.opposingClaim || "").trim();
  const defenseLayerLabel = meta.defenseLayerSynthetic ? String(meta.defenseLayerLabel || "").trim() : "";
  const defenseLayerDisplay = meta.defenseLayerSynthetic ? String(meta.selectedClaim || "").trim() : "";

  const allNodes = (rawGraph.nodes || []).map((n) => {
    const data = n.data;
    if (data && data.type === "claim" && defenseLayerLabel && defenseLayerDisplay &&
        String(data.label || "").trim() === defenseLayerLabel) {
      return { ...data, label: defenseLayerDisplay };
    }
    return data;
  });

  const selectedClaimClusterId = selectedClaim ? `framework::${selectedClaim}` : "";
  const opposingClaimClusterId = opposingClaim ? `framework::${opposingClaim}` : "";
  const defenseLayerClusterId = defenseLayerLabel ? `framework::${defenseLayerLabel}` : "";

  const duplicateNodeRedirects = new Map();
  if (selectedClaimClusterId && defenseLayerClusterId) {
    const selectedByLabel = new Map();
    for (const node of allNodes) {
      if (node?.type !== "proposition" || String(node?.clusterId || "") !== selectedClaimClusterId) continue;
      const key = String(node?.label || "").trim().toLowerCase();
      if (key && !selectedByLabel.has(key)) selectedByLabel.set(key, node.id);
    }
    for (const node of allNodes) {
      if (node?.type !== "proposition" || String(node?.clusterId || "") !== defenseLayerClusterId) continue;
      const keeperId = selectedByLabel.get(String(node?.label || "").trim().toLowerCase());
      if (keeperId && keeperId !== node.id) duplicateNodeRedirects.set(node.id, keeperId);
    }
  }

  const allEdges = (rawGraph.edges || []).map((e) => {
    const data = e.data || {};
    return {
      ...data,
      source: duplicateNodeRedirects.get(data.source) || data.source,
      target: duplicateNodeRedirects.get(data.target) || data.target,
    };
  });

  const claimsByCluster = new Map();
  const nodeById = new Map();
  for (const n of allNodes) {
    nodeById.set(n.id, n);
    if (n.type === "claim") claimsByCluster.set(n.clusterId, n.id);
  }

  const nodesToHide = new Set();
  if (defenseLayerClusterId) {
    for (const node of allNodes) {
      if (String(node?.clusterId || "") !== defenseLayerClusterId) continue;
      if (layerMode === "layer1") { nodesToHide.add(node.id); continue; }
      if (node?.type === "claim" || node?.type === "assumption") nodesToHide.add(node.id);
    }
  }

  if (layerMode === "layer1" && opposingClaimClusterId) {
    for (const node of allNodes) {
      if (String(node?.clusterId || "") !== opposingClaimClusterId) continue;
      if (node?.type === "claim" || node?.type === "assumption") nodesToHide.add(node.id);
    }
  } else if (opposingClaimClusterId) {
    let opposingClusterHasAttack = false;
    for (const edge of allEdges) {
      if (edge?.type !== "attack") continue;
      const src = nodeById.get(edge.source);
      const tgt = nodeById.get(edge.target);
      if (!src || !tgt) continue;
      if (String(src.clusterId || "") === opposingClaimClusterId ||
          String(tgt.clusterId || "") === opposingClaimClusterId) {
        opposingClusterHasAttack = true;
        break;
      }
    }
    if (!opposingClusterHasAttack) {
      const id = claimsByCluster.get(opposingClaimClusterId);
      if (id) nodesToHide.add(id);
    }
  }

  const visibleNodes = allNodes.filter(
    (n) => n.type !== "rule" && !duplicateNodeRedirects.has(n.id) && !nodesToHide.has(n.id)
  );
  const visibleById = new Set(visibleNodes.map((n) => n.id));

  const uiEdgeMap = new Map();
  function addUiEdge(source, target, type) {
    if (!source || !target || source === target) return;
    if (!visibleById.has(source) || !visibleById.has(target)) return;
    const k = `${type}::${source}::${target}`;
    if (!uiEdgeMap.has(k)) {
      uiEdgeMap.set(k, { id: `ui_${uiEdgeMap.size + 1}`, source, target, type, weight: 1 });
    } else if (type === "attack") {
      uiEdgeMap.get(k).weight += 1;
    }
  }

  for (const e of allEdges) {
    if (e.type !== "support") continue;
    const src = nodeById.get(e.source);
    const tgt = nodeById.get(e.target);
    if (!src || !tgt) continue;
    if (src.type === "rule" || tgt.type === "rule") {
      const clusterId = src.type === "rule" ? src.clusterId : tgt.clusterId;
      const claimId = claimsByCluster.get(clusterId);
      const premiseId = src.type === "rule" ? e.target : e.source;
      const premiseNode = nodeById.get(premiseId);
      if (premiseNode && premiseNode.type !== "claim" && premiseNode.type !== "rule") {
        addUiEdge(premiseId, claimId, "support");
      }
    } else {
      addUiEdge(e.source, e.target, "support");
    }
  }

  for (const e of allEdges) {
    if (e.type !== "attack") continue;
    const src = nodeById.get(e.source);
    const tgt = nodeById.get(e.target);
    if (!src || !tgt) continue;
    const isFromFocusedSupporting =
      focusSupportingLabel && String(src.label || "").trim().toLowerCase() === focusSupportingLabel;
    const isTargetOpposingAssumption =
      tgt.type === "assumption" && String(tgt.clusterId || "") === String(opposingClaimClusterId || "");
    if (isFromFocusedSupporting && isTargetOpposingAssumption) continue;
    addUiEdge(e.source, e.target, "attack");
  }

  return {
    nodes: visibleNodes,
    edges: Array.from(uiEdgeMap.values()),
    displayRows: Array.isArray(rawGraph?.displayRows) ? rawGraph.displayRows : [],
  };
}

// Their row ordering, reused so the layered reading of the graph is preserved.
export function buildLayoutRows(model) {
  const nodeById = new Map((model?.nodes || []).map((n) => [n.id, n]));
  const typeRank = { claim: 0, proposition: 1, assumption: 2 };

  const sortRow = (nodes) =>
    [...nodes].sort((a, b) => {
      const rank = (typeRank[a?.type] ?? 9) - (typeRank[b?.type] ?? 9);
      if (rank !== 0) return rank;
      if (!!a?.isFocus !== !!b?.isFocus) return a?.isFocus ? -1 : 1;
      return String(a?.label || "").localeCompare(String(b?.label || ""));
    });

  const unique = (nodes) => {
    const seen = new Set();
    return nodes.filter((n) => n?.id && !seen.has(n.id) && seen.add(n.id));
  };

  const explicit = (model.displayRows || [])
    .map((row) => (Array.isArray(row) ? row.map((id) => nodeById.get(id)).filter(Boolean) : []))
    .map((row) => sortRow(unique(row)))
    .filter((row) => row.length);
  if (explicit.length) return explicit;

  const byLevel = new Map();
  for (const node of model.nodes || []) {
    const level = Number.isFinite(Number(node?.level)) ? Number(node.level) : 99;
    if (!byLevel.has(level)) byLevel.set(level, []);
    byLevel.get(level).push(node);
  }
  return [...byLevel.entries()].sort((a, b) => a[0] - b[0])
    .map(([, nodes]) => sortRow(unique(nodes)))
    .filter((row) => row.length);
}
