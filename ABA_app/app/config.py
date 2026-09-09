#### PATCHARAKORN ####

from __future__ import annotations

from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = APP_ROOT.parent.parent  # d:\user\ReaLearn

# Task 1/3 gold data lives in the ABA_mining pipeline project, not copied here — read directly
# from there so this app always reflects the same data the pipeline produces.
ABA_MINING_ROOT = WORKSPACE_ROOT / "internship" / "ABA_mining"
TASK3_GOLD_DIR = ABA_MINING_ROOT / "Dataset" / "Task3"

# Task 1 gold: one row per (review, topic, sentiment) with the mined span, head claim, and up
# to 15 Body / Cont. Body literal pairs. Drives the Home dashboard and the Review page.
TASK1_GOLD_CSV = (
    ABA_MINING_ROOT
    / "Dataset"
    / "[Create Gold] Remove duplicate + add more version - Create Gold from Nonny Version.csv"
)

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

# Review-page topic label (as it appears in the Task 1 gold "Topic" column) -> graph aspect key.
# Only these five have Task 3 gold data and can open a Graph Studio; the other topics in the
# dataset are shown on the Review page but marked as having no framework yet.
TOPIC_TO_ASPECT = {
    "Check-in": "check-in",
    "Check-out": "check-out",
    "Price": "price",
    "Staff": "staff",
    "Facility": "facility",
}
