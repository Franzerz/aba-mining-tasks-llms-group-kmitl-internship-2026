#!/usr/bin/env bash
#### PATCHARAKORN ####
#
# Runs one local model over the full, unvoted Task 3 pair sets (build_task3_llm_pairs.py), so
# topics nobody has voted on still get contraries. The output is LLM prediction, not gold.
#
#   1. Contrary(P)Body(N) for the topics without human votes
#   2. Contrary(N)Body(P) for every topic
#
# Model / prompt: the best llama3.2 setting from the Task 3 evaluation (contrary_v1, estimated
# F1 0.50 over all five voted topics). Decoding is the evaluation's deterministic setting
# (temperature 0, top_p 0.05 from configs/model.yaml), so one run is enough: repeated runs
# differed by at most a few rows.
#
# Checkpointed like every Task 3 run: re-running this script skips finished pairs.
#
# Reliability, from running this on a laptop GPU: the local Ollama runner wedges now and then
# (every ~10 minutes with 8 concurrent requests). So requests time out quickly, a failed
# attempt restarts the runner, and each topic gets many attempts. Ollama here serves one
# request at a time, so 2 workers cost almost nothing in speed.
#
# Usage:  bash run_task3_llm_pairs.sh

set -u
cd "$(dirname "$0")"

MODEL=llama3.2
PROMPT=prompts/task3/contrary_v1.txt
PAIRS=Dataset/Task3/llm_pairs
OUT=outputs/task3_llm_pairs
WORKERS=2
ATTEMPTS=40
# A one-word answer from a 3B model takes well under a second; a request open this long means
# the runner has wedged, so fail and retry.
export OLLAMA_REQUEST_TIMEOUT=60

run_topic() {  # category topic
  local category=$1 topic=$2
  for attempt in $(seq 1 $ATTEMPTS); do
    echo "=== $category / $topic (attempt $attempt) ==="
    if python -u run_task3.py "$MODEL" --prompt "$PROMPT" --n-runs 1 --num-workers "$WORKERS" \
        --pairs-dir "$PAIRS" --output-root "$OUT" --max-pairs-per-category 0 \
        --aspects "$topic" --categories "$category"; then
      return 0
    fi
    # On Windows, Ollama serves models from a llama-server.exe child; restarting it clears a
    # wedged runner (Ollama starts a fresh one on the next request).
    taskkill //F //IM llama-server.exe > /dev/null 2>&1 || true
    sleep 15
  done
  echo "!!! $category / $topic still failing after $ATTEMPTS attempts" >&2
  return 1
}

# Smallest first, so topics become usable in the web app as soon as each one finishes.
for topic in taxi-issue food location room; do
  run_topic "Contrary(P)Body(N)" "$topic"
done
for topic in check-out taxi-issue check-in price staff food facility location room; do
  run_topic "Contrary(N)Body(P)" "$topic"
done
echo "=== all done ==="
