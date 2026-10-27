import { useMemo, useState } from "react";

// The added contribution, shown alongside mu_team's graph rather than replacing it:
//   - New ABA (C*): the generalized contrary function C*: A -> P(L)
//   - Translation:  the clone-splitting reduction back to standard ABA
//
// Both are derived from the same framework selection their payload is built from, so the
// comparison is like-for-like. Styled with their existing .sidebar-card / .token-wrap classes
// so it belongs to the same page rather than looking bolted on.

function Stat({ label, value, hint }) {
  return (
    <div className="new-aba-stat">
      <span className="new-aba-stat__label">{label}</span>
      <strong className="new-aba-stat__value">{value}</strong>
      {hint && <span className="new-aba-stat__hint">{hint}</span>}
    </div>
  );
}

export default function NewAbaPanel({ graph }) {
  const [view, setView] = useState("new");
  const [showAll, setShowAll] = useState(false);

  const newAba = graph?.newAba;
  const translation = graph?.translation;

  // Assumptions that genuinely have several distinct attackers — the case standard ABA
  // cannot express, and the reason C* exists.
  const multi = useMemo(() => {
    if (!newAba) return [];
    return Object.entries(newAba.multiContraries || {})
      .filter(([, cs]) => cs.length > 1)
      .sort((a, b) => b[1].length - a[1].length);
  }, [newAba]);

  if (!newAba || !translation) return null;

  const s = newAba.stats;
  const t = translation.stats;
  const shownMulti = showAll ? multi : multi.slice(0, 3);

  return (
    <section className="sidebar-card new-aba-card">
      <div className="new-aba-head">
        <h3>Generalized Contrary (C*) &amp; Translation</h3>
        <div className="new-aba-tabs">
          <button type="button" className={view === "new" ? "active" : ""}
                  onClick={() => setView("new")}>New ABA (C*)</button>
          <button type="button" className={view === "translated" ? "active" : ""}
                  onClick={() => setView("translated")}>Translated (C)</button>
        </div>
      </div>

      <p className="sidebar-hint">
        Standard ABA allows one contrary per assumption, so the graph above funnels every
        attacker of an assumption through a single <code>__ctr__</code> atom. C* keeps each
        attacker distinct; the Translation Rule then splits each assumption into one clone per
        contrary so py_arg can still evaluate it.
      </p>

      {view === "new" ? (
        <>
          <div className="new-aba-stats">
            <Stat label="Assumptions" value={s.assumptions} />
            <Stat label="With >1 contrary" value={s.multiContraryAssumptions}
                  hint="collapsed into one atom by standard ABA" />
            <Stat label="Largest C*(A)" value={s.maxContraries} />
            <Stat label="Total contraries" value={s.totalContraries} />
          </div>

          {multi.length === 0 ? (
            <p className="sidebar-hint">
              No assumption in this framework has more than one attacker, so C* and the standard
              contrary function agree here. Try a supporting literal with more opposing evidence
              (or Layer&nbsp;2).
            </p>
          ) : (
            <>
              <p className="sidebar-hint" style={{ marginTop: "0.6rem" }}>
                Multi-valued contraries in this framework:
              </p>
              {shownMulti.map(([assumption, contraries]) => (
                <div className="new-aba-entry" key={assumption}>
                  <div className="new-aba-entry__head">
                    C*(<span className="new-aba-atom">{assumption}</span>) &mdash;{" "}
                    {contraries.length} contraries
                  </div>
                  <div className="token-wrap">
                    {contraries.map((c) => (
                      <span className="token" key={c}>{c}</span>
                    ))}
                  </div>
                </div>
              ))}
              {multi.length > shownMulti.length && (
                <button type="button" className="tag-more" onClick={() => setShowAll(true)}>
                  show {multi.length - shownMulti.length} more
                </button>
              )}
            </>
          )}
        </>
      ) : (
        <>
          <div className="new-aba-stats">
            <Stat label="Original assumptions" value={t.originalAssumptions} />
            <Stat label="After splitting" value={t.cloneAssumptions} hint="one clone per contrary" />
            <Stat label="Rules" value={t.rules} />
            <Stat label="Flat" value={t.isFlat ? "yes" : "no"} hint="py_arg accepts it" />
          </div>

          <p className="sidebar-hint" style={{ marginTop: "0.6rem" }}>
            Each clone is named <code>assumption#contrary</code> and carries exactly one
            contrary, plus a bridging rule so the original atom stays derivable.
          </p>

          <div className="new-aba-clones">
            {Object.entries(translation.clonesByOriginal || {})
              .filter(([, clones]) => clones.length > 1)
              .slice(0, showAll ? undefined : 3)
              .map(([original, clones]) => (
                <div className="new-aba-entry" key={original}>
                  <div className="new-aba-entry__head">
                    <span className="new-aba-atom">{original}</span> &rarr; {clones.length} clones
                  </div>
                  <div className="token-wrap">
                    {clones.map((c) => <span className="token" key={c}>{c}</span>)}
                  </div>
                </div>
              ))}
          </div>
        </>
      )}
    </section>
  );
}
