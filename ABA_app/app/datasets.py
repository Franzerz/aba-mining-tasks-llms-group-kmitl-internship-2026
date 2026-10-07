#### PATCHARAKORN ####

"""
Which hotel's data the app is showing.

The site started as one hotel's dataset read through module-level constants. A second hotel
arrived with its data laid out differently, so "where the data lives" became a value rather
than a constant: one HotelDataset per hotel, and every reader takes the key of the one it is
reading. Adding a third hotel means adding an entry here, not touching the readers.

The two hotels genuinely differ in what was annotated, and the app never blurs that:

  la-veranda     Task 1/2 literals are HUMAN annotated, in one wide CSV (Body N / Cont. Body N
                 column pairs). Task 3 votes cover 5 topics, Contrary(P)Body(N) only; the other
                 topics' contraries are LLM predictions, and the Contrary(N)Body(P) side is the
                 PN votes read symmetrically (the gold workbooks do this with a spreadsheet
                 formula). See contrary_sources for why that mirroring is faithful.

  green-harbor   Task 1 gives spans and sentiment but no literals. The literals live in the
                 Task 3 vote files themselves, which cover 9 topics on BOTH categories, each
                 voted independently -- its NP votes disagree with its PN votes on about 29%
                 of pairs, so they are read as two separate human sources and never mirrored.
                 Nothing here is LLM predicted.

A consequence of that second layout: a vote file names literals but not the spans they were
mined from, so a Green Harbor literal has no review-occurrence count. The Review page shows
no count for it rather than inventing one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import (
    ABA_MINING_ROOT,
    ASPECT_GOLD_FILES,
    LLM_PAIRS_DIR,
    LLM_PREDICTIONS_DIR,
    TASK1_GOLD_CSV,
    TASK3_GOLD_DIR,
)

# How a hotel's mined spans and literals are stored.
GOLD_WIDE = "gold_wide"              # one row per span, literals in Body N / Cont. Body N pairs
TASK1_SEQUENCE = "task1_sequence"    # one row per span, no literals (a Task 1 run's output)

# Where a hotel's contrary votes come from.
GOLD_WORKBOOKS = "gold_workbooks"    # the numbered Silver/Gold verification workbooks
VOTE_DIR = "vote_dir"                # one "<Topic> - Contrary(X)Body(Y).csv" per topic+category


@dataclass(frozen=True)
class HotelDataset:
    key: str                 # URL and API value ("green-harbor")
    label: str               # how the UI names it ("Green Harbor Hotel")
    blurb: str               # one line for the picker

    spans_csv: Path          # Task 1/2 rows: the mined spans
    spans_format: str        # GOLD_WIDE | TASK1_SEQUENCE

    task3_format: str        # GOLD_WORKBOOKS | VOTE_DIR
    task3_dir: Path
    aspect_gold_files: dict[str, tuple[str, str | None]] = field(default_factory=dict)
    vote_dir: Path | None = None
    llm_pairs_dir: Path | None = None
    llm_predictions_dir: Path | None = None

    @property
    def has_literal_counts(self) -> bool:
        """True when a literal can be traced to the spans that produced it, so the Review page
        can count its occurrences. Only the wide gold layout pairs literals with spans."""
        return self.spans_format == GOLD_WIDE

    @property
    def mirrors_pn_votes(self) -> bool:
        """True when the Contrary(N)Body(P) side is the PN votes read symmetrically rather than
        its own answers. A hotel whose NP sheet was voted separately must not be mirrored."""
        return self.task3_format == GOLD_WORKBOOKS


LA_VERANDA = HotelDataset(
    key="la-veranda",
    label="La Veranda Hotel",
    blurb="Human-annotated literals; 5 topics with voted contraries, the rest LLM predicted.",
    spans_csv=TASK1_GOLD_CSV,
    spans_format=GOLD_WIDE,
    task3_format=GOLD_WORKBOOKS,
    task3_dir=TASK3_GOLD_DIR,
    aspect_gold_files=ASPECT_GOLD_FILES,
    llm_pairs_dir=LLM_PAIRS_DIR,
    llm_predictions_dir=LLM_PREDICTIONS_DIR,
)

GREEN_HARBOR = HotelDataset(
    key="green-harbor",
    label="Green Harbor Hotel",
    blurb="9 topics with contraries voted on both categories; literals come from those votes.",
    spans_csv=ABA_MINING_ROOT / "Dataset" / "green_new_hotel.csv",
    spans_format=TASK1_SEQUENCE,
    task3_format=VOTE_DIR,
    task3_dir=TASK3_GOLD_DIR / "green_new_hotel",
    vote_dir=TASK3_GOLD_DIR / "green_new_hotel" / "Task3_Vote",
)

DATASETS: dict[str, HotelDataset] = {d.key: d for d in (LA_VERANDA, GREEN_HARBOR)}
DEFAULT_DATASET = LA_VERANDA.key


def get(key: str | None) -> HotelDataset:
    """The dataset for a key, falling back to the default for None or an unknown key, so a stale
    bookmark shows the default hotel instead of a 500."""
    if not key:
        return DATASETS[DEFAULT_DATASET]
    return DATASETS.get(str(key).strip().lower(), DATASETS[DEFAULT_DATASET])


def listing() -> list[dict]:
    """What the hotel picker shows."""
    return [
        {"key": d.key, "label": d.label, "blurb": d.blurb, "default": d.key == DEFAULT_DATASET}
        for d in DATASETS.values()
    ]
