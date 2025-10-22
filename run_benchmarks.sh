#!/bin/bash

# Run MLX Omni Server benchmarks with different parameters
# Modify the --rounds and --concurrency values as needed

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$SCRIPT_DIR"

# Prefer repo venv python if available
PY_BIN="$REPO_DIR/venv/bin/python"
if [ ! -x "$PY_BIN" ]; then
  PY_BIN="python"
fi

BENCH_PY="$REPO_DIR/examples/benchmark_chat.py"
OUT="$REPO_DIR/examples/results10.txt"
: > "$OUT"

for c in $(seq 1 10); do
  echo "==== Concurrency $c ====\n" >> "$OUT"
  STREAMING=true "$PY_BIN" "$BENCH_PY" --rounds 3 --vary-prompt --concurrency "$c" >> "$OUT" 2>&1
  echo "" >> "$OUT"
done

echo "All runs complete. Results written to $OUT"
