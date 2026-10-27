import Frame, { PageHead } from "../components/Frame.jsx";

/** What this site is, where its data came from, and what the numbers on it do and do not mean.
 *
 *  Everything cited here is checkable: each source is a published paper, a public dataset or a
 *  file in this repository. Where a number on the site rests on a judgement call rather than on
 *  the data, that is said here rather than left for the reader to discover. */

const Card = ({ title, children }) => (
  <section className="card p-5 min-w-0">
    <h3 className="m-0 mb-2 text-[15px] font-extrabold">{title}</h3>
    <div className="text-[13px] leading-relaxed text-ink/90 space-y-2">{children}</div>
  </section>
);

const Cite = ({ href, children }) =>
  href ? (
    <a href={href} target="_blank" rel="noreferrer"
       className="text-accent underline underline-offset-2 hover:opacity-80">{children}</a>
  ) : (
    <span>{children}</span>
  );

const Row = ({ label, children }) => (
  <div className="grid grid-cols-[8.5rem_minmax(0,1fr)] gap-3 py-1.5 border-b border-edge/60 last:border-0">
    <span className="text-[12px] font-bold text-muted">{label}</span>
    <span className="text-[12px] leading-relaxed">{children}</span>
  </div>
);

export default function About() {
  return (
    <Frame>
      <PageHead
        eyebrow="About"
        title="What this site is"
        subtitle="A reader for assumption-based argumentation frameworks mined from hotel
          reviews: the mined points, the claims they back, what contradicts them, and which
          claims survive under formal argumentation semantics."
      />

      <div className="grid gap-4 lg:grid-cols-2 mt-6">
        <Card title="The idea">
          <p className="m-0">
            A hotel has thousands of short, conflicting reviews. One guest says the room was
            clean, another says it was not. Reading them one by one does not scale, and asking a
            language model to summarise them hides how it weighed the disagreement.
          </p>
          <p className="m-0">
            Here a language model only <strong>reads</strong>. It extracts the points guests
            made and which points contradict each other; what follows from those points is then
            decided by <strong>assumption-based argumentation</strong> (ABA), a formal system
            with published semantics. So a verdict such as "the staff are good" can always be
            traced to the reviews behind it and to the reviews standing in its way.
          </p>
        </Card>

        <Card title="How a framework is built">
          <p className="m-0">Three mining tasks run over the raw reviews:</p>
          <ol className="m-0 pl-5 space-y-1">
            <li><strong>Task 1</strong> split each review into topic-relevant spans and label
              each with a sentiment.</li>
            <li><strong>Task 2</strong> rewrite each span into short propositions
              (<code>clean_room</code>) and pair each with an evidential assumption
              (<code>no_evident_not_clean_room</code>, read as "nothing in the evidence speaks
              against it").</li>
            <li><strong>Task 3</strong> decide, for each pair of propositions, whether one
              contradicts the other. A "Yes" becomes a contrary, which is where one argument can
              undercut another.</li>
          </ol>
          <p className="m-0">
            The result is evaluated by <Cite href="https://pyarg.npai.science.uu.nl/">PyArg</Cite>,
            so the picture on screen and the accepted set always come from the same object.
          </p>
        </Card>

        <Card title="Where the reviews come from">
          <Row label="La Veranda">
            Reviews of the La Veranda Hotel, Larnaca, Cyprus, collected from Booking.com and
            published as{" "}
            <Cite href="https://www.kaggle.com/datasets/michelhatab/hotel-reviews-bookingcom">
              Hotel Reviews Booking.com (Kaggle, Michel Hatab, 2023, version 3)
            </Cite>. 1,500 reviews, of which 785 carry free text split into a positive and a
            negative comment.
          </Row>
          <Row label="Green Harbor">
            A second hotel crawled separately for this project, to test whether the pipeline
            generalises beyond the hotel it was developed on.
          </Row>
          <Row label="Annotation">
            Topics, spans, sentiments and literals were annotated by hand. The contrary labels
            for five topics were voted by annotators; the rest are model predictions and are
            labelled as such everywhere they appear.
          </Row>
        </Card>

        <Card title="Reading the numbers honestly">
          <p className="m-0">
            Two things on this site are easy to over-read, so they are marked wherever they
            appear:
          </p>
          <ul className="m-0 pl-5 space-y-1">
            <li><strong>Voted vs predicted.</strong> A contrary set is either human-voted gold or
              a model's guess that nobody has checked. The Graph Studio sidebar says which, per
              topic and per direction.</li>
            <li><strong>Missing counts.</strong> Green Harbor's literals come from its vote files,
              which name a literal but not the reviews it was mined from. Its Review page shows a
              dash rather than a zero, because "no count available" is not "appeared zero times".</li>
          </ul>
          <p className="m-0">
            Topics also differ enormously in size. Check-out has 8 candidate pairs where Facility
            has 11,686, so a score on the former carries far more uncertainty than the same score
            on the latter.
          </p>
        </Card>

        <Card title="Method and prior work">
          <Row label="ABA">
            Bondarenko, Dung, Kowalski and Toni, <em>An Abstract, Argumentation-Theoretic
            Approach to Default Reasoning</em>, Artificial Intelligence 93 (1997), 63–101.
          </Row>
          <Row label="ABA mining">
            Racharak, Jearanaiwongkul, Pooksook and Takashima, <em>Towards Assumption-based
            Argumentation Mining in Hotel Reviews</em>, CLAR 2025, Springer, 272–290. Defines
            the three-task pipeline.
          </Row>
        </Card>

        <Card title="This application">
          <p className="m-0">
            FastAPI serves both the API and the built React page, so there is one process and one
            port. The graph is drawn with Cytoscape.js. There is no database: every number is
            read from the mined files at startup and cached.
          </p>
          <p className="m-0">
            The framework shown is built once per topic and contrary set, and the three views,
            Simple ABA, ABA, and the reduced framework the reasoner consumes, are renderings of
            that same object, so the picture can never disagree with what was evaluated.
          </p>
        </Card>

        <section className="card p-5 min-w-0 lg:col-span-2">
          <h3 className="m-0 mb-1 text-[15px] font-extrabold">The team</h3>
          <p className="m-0 mb-4 text-[12px] text-muted">
            An undergraduate project at King Mongkut&apos;s Institute of Technology Ladkrabang
            (KMITL), 2026.
          </p>

          <div className="grid gap-5 sm:grid-cols-2">
            <div>
              <h4 className="m-0 mb-2 text-[11px] font-bold uppercase tracking-wide text-muted">
                Students
              </h4>
              <ul className="m-0 p-0 list-none space-y-2">
                {[
                  ["Patcharakorn Ngoendee", "66011132"],
                  ["Ratchanon Thonkam", "66011195"],
                ].map(([name, sid]) => (
                  <li key={sid} className="flex items-baseline justify-between gap-3">
                    <span className="text-[13px] font-semibold">{name}</span>
                    <span className="text-[12px] font-mono text-muted tabular-nums">{sid}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div>
              <h4 className="m-0 mb-2 text-[11px] font-bold uppercase tracking-wide text-muted">
                Acknowledgements
              </h4>
              <ul className="m-0 p-0 list-none space-y-2">
                <li>
                  <span className="block text-[11px] text-muted">Advisor</span>
                  <span className="text-[13px] font-semibold">
                    Assoc. Prof. Teeradaj Racharak
                  </span>
                </li>
                <li>
                  <span className="block text-[11px] text-muted">Co-advisor</span>
                  <span className="text-[13px] font-semibold">
                    Assoc. Prof. Rathachai Chawuthai
                  </span>
                </li>
              </ul>
            </div>
          </div>

          <p className="m-0 mt-4 pt-3 border-t border-edge/60 text-[12px] text-muted leading-relaxed">
            The ABA mining task definitions this work builds on are the advisor&apos;s own
            published research, cited under <em>Method and prior work</em> above.
          </p>
        </section>
      </div>
    </Frame>
  );
}
