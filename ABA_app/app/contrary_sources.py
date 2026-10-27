#### PATCHARAKORN ####

"""
Where a topic's contrary relation comes from.

Two kinds of source, and the app never blurs them:
  human  the Task 3 gold sets: every pair voted by annotators (5 topics, Contrary(P)Body(N) only)
  llm    predictions from one local model over the full, unvoted pair sets written by
         ABA_mining/build_task3_llm_pairs.py — every other topic's Contrary(P)Body(N), and
         every topic's Contrary(N)Body(P)

Both are normalised into the same four columns so the builder treats them alike:
    assumption   the attacked assumption (no_evident_not_A / have_evident_B)
    literal      that assumption's own literal (A / B)
    attacker     the candidate contrary (a literal of the opposite sentiment)
    vote         "Yes" / "No" / None (no usable answer)

A topic with human votes uses ONLY those votes. The annotators answered one question per literal
PAIR, on the Contrary(P)Body(N) sheet: "does B contradict A?". Contrary(N)Body(P) asks the same
pair the other way round — its assumption is have_evident_B and its candidate contrary is A — so
its answer is that same vote, read symmetrically. The gold workbooks' own Contrary(N)Body(P) sheet
is filled in by a spreadsheet formula that does exactly this (its Vote column is an ARRAYFORMULA,
not hand-entered, and in the one workbook kept here as .xlsx that column is empty), and mirroring
the votes here reproduces it row for row.

So a voted topic is human on BOTH sides, and the LLM's guesses for it are never read. The LLM
covers only the topics nobody voted on.

An LLM source is only used once it covers its whole pair set: a half-finished run would leave
some assumptions looking unattacked when they simply haven't been asked about yet.

A topic can also have NO pairs at all: every pair needs one praise literal and one complaint
literal, so a topic mined with only one sentiment (booking-issue: six complaints, no praise)
has nothing to ask about. It still gets a framework — its literals and assumptions, with every
contrary a placeholder — built from the Task 1/2 gold spans.
"""

from __future__ import annotations

import functools
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import datasets
from .config import CACHE_DIR, LLM_CONTRARY_MODEL, LLM_CONTRARY_PROMPT
from .datasets import DEFAULT_DATASET, VOTE_DIR

HUMAN, LLM = "human", "llm"
PN, NP = "Contrary(P)Body(N)", "Contrary(N)Body(P)"
POS_PREFIX, NEG_PREFIX = "no_evident_not_", "have_evident_"
COLUMNS = ["assumption", "literal", "attacker", "vote"]


@dataclass(frozen=True)
class ContrarySource:
    category: str
    kind: str            # HUMAN | LLM
    label: str           # how the UI names it
    rows: pd.DataFrame   # COLUMNS
    total: int           # pairs in the set
    done: int            # pairs that have a vote / a prediction

    @property
    def complete(self) -> bool:
        return self.total > 0 and self.done >= self.total

    @functools.cached_property
    def _stats(self) -> dict:
        votes = self.rows["vote"]
        return {
            "category": self.category, "source": self.kind, "label": self.label,
            "total": self.total, "done": self.done, "complete": self.complete,
            "yes": int((votes == "Yes").sum()), "no": int((votes == "No").sum()),
        }

    def stats(self) -> dict:
        return dict(self._stats)


def topic_key(topic: str) -> str:
    """Review-page topic label -> aspect key ("Taxi-issue" -> "taxi-issue"); same rule as
    ABA_mining/build_task3_llm_pairs.py uses for the pair-set file names."""
    return re.sub(r"[^\w\-]+", "_", topic.strip()).lower()


def _vote(value: object) -> str | None:
    text = str(value).strip().lower() if isinstance(value, str) else ""
    return {"yes": "Yes", "no": "No"}.get(text)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _detect(df: pd.DataFrame, candidates: list[str]) -> str | None:
    mapping = {str(c).strip().lower(): c for c in df.columns}
    return next((mapping[c.lower()] for c in candidates if c.lower() in mapping), None)


# --- human votes -----------------------------------------------------------------------------

def _read_gold_sheet(dataset: str, aspect: str, path: Path, sheet: str) -> pd.DataFrame:
    """One sheet of a gold workbook, via a CSV copy: openpyxl spends ~14 s opening facility's
    seven-sheet workbook just to hand back one of them. The copy is named after the workbook's
    size and mtime, so an edited workbook is read afresh."""
    stat = path.stat()
    token = _category_token(sheet)
    stem = f"{dataset}__{aspect}__{token}"
    copy = CACHE_DIR / "gold" / f"{stem}__{stat.st_size}_{int(stat.st_mtime)}.csv"
    if copy.exists():
        return pd.read_csv(copy, dtype=str, keep_default_na=False, na_values=[""])
    df = pd.read_excel(path, sheet_name=sheet, dtype=str)
    try:
        copy.parent.mkdir(parents=True, exist_ok=True)
        for old in copy.parent.glob(f"{stem}__*.csv"):
            old.unlink(missing_ok=True)
        tmp = copy.with_suffix(".tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(copy)
    except OSError:
        pass   # an optimisation only
    return df


def _normalise_votes(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """A voted sheet, in whatever column naming it uses, normalised to COLUMNS.

    The gold workbooks are not internally consistent about this (Original A / Original
    Assumption, A / Assumption, B / Proposition) and the per-topic vote files use a third mix,
    so the columns are detected rather than assumed.
    """
    cols = [_detect(df, ["Original Assumption", "Original A"]), _detect(df, ["Assumption", "A"]),
            _detect(df, ["Proposition", "B"]), _detect(df, ["Vote"])]
    if not all(cols):
        raise ValueError(f"Missing expected columns in {path.name}. Found: {list(df.columns)}")
    df = df[cols].copy()
    df.columns = COLUMNS
    df = df.dropna(subset=["assumption", "literal", "attacker"])
    for c in ["assumption", "literal", "attacker"]:
        df[c] = df[c].astype(str).str.strip()
    df = df[(df["literal"] != "") & (df["attacker"] != "")]
    df["vote"] = df["vote"].map(_vote)
    return df.reset_index(drop=True)


@functools.lru_cache(maxsize=32)
def _human_rows(dataset: str, aspect: str, _stamp: float) -> pd.DataFrame:
    ds = datasets.get(dataset)
    filename, sheet = ds.aspect_gold_files[aspect]
    path = ds.task3_dir / filename
    df = (pd.read_csv(path, dtype=str) if sheet is None
          else _read_gold_sheet(dataset, aspect, path, sheet))
    return _normalise_votes(df, path)


@functools.lru_cache(maxsize=32)
def _human_source(dataset: str, aspect: str, stamp: float) -> ContrarySource:
    rows = _human_rows(dataset, aspect, stamp)
    # A voted set is complete by definition; "Undetermined" votes simply carry no contrary.
    return ContrarySource(PN, HUMAN, "human votes", rows, total=len(rows), done=len(rows))


def human_source(dataset: str, aspect: str) -> ContrarySource | None:
    """The Contrary(P)Body(N) votes for a topic, in whichever layout this hotel stores them."""
    ds = datasets.get(dataset)
    if ds.task3_format == VOTE_DIR:
        return vote_source(dataset, aspect, PN)
    if aspect not in ds.aspect_gold_files:
        return None
    filename, _ = ds.aspect_gold_files[aspect]
    return _human_source(dataset, aspect, _mtime(ds.task3_dir / filename))


# --- votes stored as one file per topic and category -----------------------------------------

def _vote_csv(dataset: str, aspect: str, category: str) -> Path | None:
    """The vote file for one topic and category, or None.

    These are named with the Review-page topic label ("Check-in - Contrary(P)Body(N).csv"), so
    the aspect key is matched against each name rather than rebuilt from it.
    """
    ds = datasets.get(dataset)
    if ds.vote_dir is None or not ds.vote_dir.is_dir():
        return None
    suffix = f" - {category}.csv"
    for path in ds.vote_dir.glob(f"*{suffix}"):
        if topic_key(path.name[: -len(suffix)]) == aspect:
            return path
    return None


@functools.lru_cache(maxsize=64)
def _vote_source(dataset: str, aspect: str, category: str, _stamp: float) -> ContrarySource:
    path = _vote_csv(dataset, aspect, category)
    rows = _normalise_votes(pd.read_csv(path, dtype=str), path)
    return ContrarySource(category, HUMAN, "human votes", rows, total=len(rows), done=len(rows))


def vote_source(dataset: str, aspect: str, category: str) -> ContrarySource | None:
    """One category's votes for one topic.

    Unlike the gold workbooks, both categories here were put to the annotators separately, so
    each is read from its own file and the PN answers are never mirrored onto NP.
    """
    path = _vote_csv(dataset, aspect, category)
    return None if path is None else _vote_source(dataset, aspect, category, _mtime(path))


def vote_topics(dataset: str) -> list[str]:
    """The aspects this hotel has vote files for."""
    ds = datasets.get(dataset)
    if ds.vote_dir is None or not ds.vote_dir.is_dir():
        return []
    out = set()
    for path in ds.vote_dir.glob("*.csv"):
        label = path.stem.split(" - ")[0].strip()
        if label:
            out.add(topic_key(label))
    return sorted(out)


def vote_literals(dataset: str, aspect: str) -> dict[str, set[str]]:
    """A topic's praise and complaint literals, read off its vote files.

    A PN row pairs a praise literal (its A) with a complaint (its B); an NP row is the same pair
    the other way round. For a hotel whose literals exist nowhere else, this is where they come
    from.
    """
    out: dict[str, set[str]] = {"Positive": set(), "Negative": set()}
    pn = vote_source(dataset, aspect, PN)
    if pn is not None:
        out["Positive"] |= set(pn.rows["literal"])
        out["Negative"] |= set(pn.rows["attacker"])
    np_side = vote_source(dataset, aspect, NP)
    if np_side is not None:
        out["Negative"] |= set(np_side.rows["literal"])
        out["Positive"] |= set(np_side.rows["attacker"])
    return out


# --- LLM predictions -------------------------------------------------------------------------

def _category_token(category: str) -> str:
    return re.sub(r"[^\w\-]+", "_", category).strip("_")


def _pairs_csv(dataset: str, topic: str, category: str) -> Path | None:
    # Plain-CSV copy of the pair-set workbook sheet; the workbook itself takes seconds to parse.
    ds = datasets.get(dataset)
    if ds.llm_pairs_dir is None:
        return None          # this hotel has no LLM predictions at all
    return ds.llm_pairs_dir / "csv" / f"{topic}__{_category_token(category)}.csv"


def _prediction_csv(dataset: str, topic: str, category: str) -> Path | None:
    ds = datasets.get(dataset)
    if ds.llm_predictions_dir is None:
        return None
    token = _category_token(category)
    return (ds.llm_predictions_dir / LLM_CONTRARY_MODEL / LLM_CONTRARY_PROMPT / topic / token
            / "csv" / f"task3_{LLM_CONTRARY_PROMPT}_{topic}_{token}_run1.csv")


def _read_pairs(dataset: str, topic: str, category: str) -> pd.DataFrame:
    pairs = pd.read_csv(_pairs_csv(dataset, topic, category), dtype=str).fillna("")
    asm_col = _detect(pairs, ["Original Assumption", "Original A"])
    pairs = pairs[["ID", asm_col, "A", "B"]].copy()
    pairs.columns = ["ID", "assumption", "literal", "attacker"]
    return pairs


def _read_predictions(dataset: str, topic: str, category: str) -> pd.DataFrame | None:
    path = _prediction_csv(dataset, topic, category)
    if path is None or not path.exists():
        return None

    # Each prediction row also stores its whole rendered prompt, so a finished run's file is
    # ~250 MB and takes seconds to parse just for four short columns. Those columns are kept in a
    # slim copy named after the source file's size and mtime, so any change to the source
    # (a resumed run, a re-run) makes a new copy instead of serving a stale one.
    stat = path.stat()
    slim_dir = CACHE_DIR / "predictions"
    slim_stem = f"{dataset}__{topic}__{_category_token(category)}"
    slim = slim_dir / f"{slim_stem}__{stat.st_size}_{int(stat.st_mtime)}.csv"
    if slim.exists():
        return pd.read_csv(slim, dtype=str, keep_default_na=False)

    def read() -> pd.DataFrame:
        # The Prompt column (the whole rendered prompt per row) is never needed here.
        return pd.read_csv(path, dtype=str, usecols=lambda c: c != "Prompt").fillna("")

    try:
        pred = read()
    except pd.errors.ParserError:
        # A run appends to this file while we read it; a row caught mid-write parses on retry.
        time.sleep(0.3)
        pred = read()
    pred = pred.drop_duplicates(subset=["ID", "PhraseA", "PhraseB"], keep="last")
    pred = pred.rename(columns={"PhraseA": "assumption", "PhraseB": "attacker", pred.columns[-1]: "raw"})
    pred = pred[["ID", "assumption", "attacker", "raw"]]

    # Only a file that has stopped changing is worth a copy; a live run rewrites it every second.
    if time.time() - stat.st_mtime > 60:
        try:
            slim_dir.mkdir(parents=True, exist_ok=True)
            for old in slim_dir.glob(f"{slim_stem}__*.csv"):
                old.unlink(missing_ok=True)
            tmp = slim.with_suffix(".tmp")
            pred.to_csv(tmp, index=False)
            tmp.replace(slim)
        except OSError:
            pass   # the copy is an optimisation; reading the source again is always correct
    return pred


# One entry per (topic, category), replaced when either file changes. A size-capped LRU cache
# here thrashed: every page asks about all 18 sets, more than the cap, so the biggest ones
# (room: ~98k rows each) were re-read and re-merged on every request (2.5 s per call).
_LLM_CACHE: dict[tuple[str, str, str], tuple[tuple[float, float], ContrarySource]] = {}
_LLM_LOCK = threading.Lock()
# One loader per set at a time: a request arriving while the startup warm-up is loading the same
# set waits for that load instead of repeating it in parallel.
_LOAD_LOCKS: defaultdict[tuple[str, str, str], threading.Lock] = defaultdict(threading.Lock)


def llm_source(dataset: str, topic: str, category: str) -> ContrarySource | None:
    pairs_path = _pairs_csv(dataset, topic, category)
    if pairs_path is None or not pairs_path.exists():
        return None
    key = (dataset, topic, category)
    stamps = (_mtime(pairs_path), _mtime(_prediction_csv(dataset, topic, category)))
    with _LLM_LOCK:
        hit = _LLM_CACHE.get(key)
        load_lock = _LOAD_LOCKS[key]
    if hit and hit[0] == stamps:
        return hit[1]
    with load_lock:
        with _LLM_LOCK:
            hit = _LLM_CACHE.get(key)
        if hit and hit[0] == stamps:
            return hit[1]
        return _load_llm_source(dataset, topic, category, stamps)


def _load_llm_source(dataset: str, topic: str, category: str,
                     stamps: tuple[float, float]) -> ContrarySource:
    pairs = _read_pairs(dataset, topic, category)
    pred = _read_predictions(dataset, topic, category)
    if pred is None:
        rows, done = pairs.assign(vote=None)[COLUMNS], 0
    else:
        merged = pairs.merge(pred, on=["ID", "assumption", "attacker"], how="left")
        done = int(merged["raw"].notna().sum())
        merged["vote"] = merged["raw"].map(_vote)
        rows = merged[COLUMNS]
    label = f"{LLM_CONTRARY_MODEL} · {LLM_CONTRARY_PROMPT}"
    source = ContrarySource(category, LLM, label, rows, total=len(pairs), done=done)
    with _LLM_LOCK:
        _LLM_CACHE[(dataset, topic, category)] = (stamps, source)
    return source


# --- what exists -----------------------------------------------------------------------------

def mirrored_human_source(dataset: str, aspect: str) -> ContrarySource | None:
    """Contrary(N)Body(P) for a voted topic, read off the Contrary(P)Body(N) votes.

    A PN row asks "is B a contrary of A?" about the assumption no_evident_not_A. The NP row for
    the same literal pair asks about have_evident_B with A as the candidate, and the annotators'
    answer for the pair is the answer for both. The NP pair set supplies the row identities
    (ids and exact atom names); only the votes come from the gold sheet.
    """
    ds = datasets.get(dataset)
    if not ds.mirrors_pn_votes:
        return None          # this hotel voted its NP sheet separately; read it, do not derive it
    pairs = _pairs_csv(dataset, aspect, NP)
    if human_source(dataset, aspect) is None or pairs is None or not pairs.exists():
        return None
    return _mirrored_human_source(dataset, aspect, _mtime(pairs), _human_stamp(dataset, aspect))


@functools.lru_cache(maxsize=32)
def _mirrored_human_source(dataset: str, aspect: str, _pairs_stamp: float,
                           _gold_stamp: float) -> ContrarySource:
    votes = human_source(dataset, aspect).rows
    by_pair = dict(zip(zip(votes["literal"], votes["attacker"]), votes["vote"]))
    rows = _read_pairs(dataset, aspect, NP).copy()
    # (literal, attacker) here is (B, A); the vote was recorded against the pair (A, B).
    rows["vote"] = [by_pair.get((atk, lit)) for lit, atk in zip(rows["literal"], rows["attacker"])]
    # Complete for the same reason the PN set is: every pair was put to the annotators, and an
    # "Undetermined" answer is an answer that yields no contrary, not a missing one.
    return ContrarySource(NP, HUMAN, "human votes (same pair, read the other way)",
                          rows[COLUMNS], total=len(rows), done=len(rows))


def _human_stamp(dataset: str, aspect: str) -> float:
    ds = datasets.get(dataset)
    filename, _ = ds.aspect_gold_files[aspect]
    return _mtime(ds.task3_dir / filename)


def llm_topics(dataset: str) -> list[str]:
    ds = datasets.get(dataset)
    if ds.llm_pairs_dir is None:
        return []
    folder = ds.llm_pairs_dir / "csv"
    return sorted({p.name.split("__")[0] for p in folder.glob("*__*.csv")}) if folder.is_dir() else []


def span_literals(dataset: str, aspect: str) -> dict[str, set[str]]:
    """{"Positive": {...}, "Negative": {...}}: the body literals of one topic.

    For a hotel whose spans carry their own literals these come from the spans; for one whose
    literals live only in its vote files they come from there instead.
    """
    from .review_data import topic_label, topic_literals  # local import: review_data imports us

    ds = datasets.get(dataset)
    if ds.task3_format == VOTE_DIR:
        return vote_literals(dataset, aspect)
    label = topic_label(dataset, aspect)
    return topic_literals(dataset, label) if label else {"Positive": set(), "Negative": set()}


def no_pair_topics(dataset: str) -> list[str]:
    """Topics with literals but no possible pair: mined with only one sentiment."""
    from .review_data import topic_keys  # local import: review_data imports us

    ds = datasets.get(dataset)
    known = set(ds.aspect_gold_files) | set(llm_topics(dataset)) | set(vote_topics(dataset))
    out = []
    for aspect in topic_keys(dataset):
        if aspect in known:
            continue
        lits = span_literals(dataset, aspect)
        if (lits["Positive"] or lits["Negative"]) and not (lits["Positive"] and lits["Negative"]):
            out.append(aspect)
    return sorted(out)


def sources_for(dataset: str, aspect: str) -> dict[str, ContrarySource | None]:
    """The positive side's contraries (PN) and the negative side's (NP) for one topic.

    Votes win wherever they exist. A voted topic is human on both sides: PN straight from the gold
    sheet, NP the same votes read the other way round (see mirrored_human_source). The LLM covers
    only the topics nobody voted on.
    """
    ds = datasets.get(dataset)
    if ds.task3_format == VOTE_DIR:
        # Both categories were voted separately here, so each side is read from its own file.
        return {"pn": vote_source(dataset, aspect, PN), "np": vote_source(dataset, aspect, NP)}
    human = human_source(dataset, aspect)
    if human is not None:
        return {"pn": human,
                "np": mirrored_human_source(dataset, aspect) or llm_source(dataset, aspect, NP)}
    return {"pn": llm_source(dataset, aspect, PN), "np": llm_source(dataset, aspect, NP)}


def source_signature(dataset: str, aspect: str) -> tuple:
    """Changes whenever the data a topic's framework is built from changes."""
    out = []
    for side, src in sources_for(dataset, aspect).items():
        if src is None:
            out.append((side, None))
        else:
            stats = src.stats()
            out.append((side, src.kind, src.total, src.done, stats["yes"]))
    ds = datasets.get(dataset)
    if aspect in ds.aspect_gold_files:
        out.append(_mtime(ds.task3_dir / ds.aspect_gold_files[aspect][0]))
    for category in (PN, NP):
        path = _vote_csv(dataset, aspect, category)
        if path is not None:
            out.append(_mtime(path))
    return tuple(out)


def available_aspects(dataset: str) -> list[str]:
    """Topics that can be built right now: voted ones, LLM topics whose Contrary(P)Body(N)
    predictions are complete, and topics with no possible pair."""
    ds = datasets.get(dataset)
    out = set(ds.aspect_gold_files) | set(vote_topics(dataset)) | set(no_pair_topics(dataset))
    for topic in llm_topics(dataset):
        pn = llm_source(dataset, topic, PN)
        if pn is not None and pn.complete:
            out.add(topic)
    return sorted(out)


def aspect_status(dataset: str) -> dict[str, dict]:
    """Per topic: where each side's contraries come from and how far an LLM run has got."""
    ds = datasets.get(dataset)
    status = {}
    voted = set(ds.aspect_gold_files) | set(vote_topics(dataset))
    for aspect in sorted(voted | set(llm_topics(dataset))):
        sources = sources_for(dataset, aspect)
        status[aspect] = {side: (src.stats() if src else None) for side, src in sources.items()}
    for aspect in no_pair_topics(dataset):
        status[aspect] = {"pn": None, "np": None, "no_pairs": True}
    return status
