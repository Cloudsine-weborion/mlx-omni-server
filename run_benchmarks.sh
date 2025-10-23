#!/bin/bash

# Run MLX Omni Server vision benchmark (rounds=5) across specified concurrencies

ROUNDS=5
CONCURRENCIES=(2 3 4 5 6 7 8 9 10 15)
MODEL="mlx-community/gemma-3-12b-it-4bit"
BASE_URL="http://localhost:10240/v1"
IMAGE="/Users/applesmacbookpro/Documents/mlx-omni-server/examples/stickman.png"

for c in "${CONCURRENCIES[@]}"; do
echo "Running vision benchmark with rounds=${ROUNDS}, concurrency=${c}"
python examples/benchmark_vision.py \
  --rounds "${ROUNDS}" \
  --concurrency "${c}" \
  --model "${MODEL}" \
  --base-url "${BASE_URL}" \
  --image "${IMAGE}"
echo ""
done