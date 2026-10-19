#### PATCHARAKORN ####

from __future__ import annotations

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = APP_ROOT.parent.parent  # d:\user\ReaLearn

# Task 1/3 gold data lives in the ABA_mining pipeline project, not copied here — read directly
# from there so this app always reflects the same data the pipeline produces.
#
# ABA_MINING_ROOT overrides the location. A container has no such repo layout, and most of that
# tree is deliberately not in git (outputs/task3_llm_pairs alone is several GB), so the data is
# mounted at run time rather than baked into the image. See docker-compose.yml.
ABA_MINING_ROOT = Path(
    os.environ.get("ABA_MINING_ROOT") or WORKSPACE_ROOT / "internship" / "ABA_mining"
)
TASK3_GOLD_DIR = ABA_MINING_ROOT / "Dataset" / "Task3"

# Task 1/2 gold: one row per (review, topic, sentiment) with the review ID, the mined span, head
# claim, and up to 15 Body / Cont. Body literal pairs. Drives the Home dashboard, the Review page
# and the literal sets of topics nobody voted on. Same file as `gold_csv` in
# ABA_mining/configs/paths.yaml, so the app and the pipeline read one dataset.
TASK1_GOLD_CSV = (
    ABA_MINING_ROOT
    / "Dataset"
    / "Original ABA Dataset for Version 2 [Oct 23, 2025], Senior Project, MUICT - Sheet2_.csv"
)

# Derived files the app can rebuild at any time (safe to delete).
CACHE_DIR = APP_ROOT / ".cache"

FRONTEND_DIR = APP_ROOT / "frontend"
DIST_DIR = FRONTEND_DIR / "dist"  # Vite build output, served by main.py

# Aspect -> (gold filename, sheet name or None). Four aspects were exported as single-sheet
# CSVs; facility's gold is still a multi-sheet .xlsx, so it names the sheet to read. All five
# hold the same "Contrary(P)Body(N)" category with real human votes (see
# ABA_mining/src/task3.py for the full multi-category loader this is a simplified read of).
ASPECT_GOLD_FILES: dict[str, tuple[str, str | None]] = {
    "check-in": ("2. Verify - Task 3 - Check-in (Silver).xlsx - Contrary(P)Body(N).csv", None),
    "check-out": ("1. Verify - Task 3 - Check-out (Silver).xlsx - Contrary(P)Body(N).csv", None),
    "price": ("3. Verify - Task 3 - Price (Silver).xlsx - Contrary(P)Body(N).csv", None),
    "staff": ("4. Verify - Task 3 - Staff (Silver).xlsx - Contrary(P)Body(N).csv", None),
    "facility": ("5. Verify - Task 3 - Facility (Gold).xlsx", "Contrary(P)Body(N)"),
}

# Contraries nobody has voted on come from one local model run over the full pair sets
# (ABA_mining/build_task3_llm_pairs.py + run_task3.py --pairs-dir ... --output-root ...).
# The model / prompt pair is the best llama3.2 setting from the Task 3 evaluation — see
# ABA_mining/outputs/eval/task3_server/TASK3_MODEL_COMPARISON.md.
LLM_PAIRS_DIR = TASK3_GOLD_DIR / "llm_pairs"
LLM_PREDICTIONS_DIR = ABA_MINING_ROOT / "outputs" / "task3_llm_pairs"
LLM_CONTRARY_MODEL = "llama3.2"
LLM_CONTRARY_PROMPT = "contrary_v1"
