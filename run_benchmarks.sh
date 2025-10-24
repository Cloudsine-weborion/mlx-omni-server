#!/bin/bash

# Run MLX Omni Server vision benchmark (rounds=5) across specified concurrencies

ROUNDS=3
CONCURRENCIES=(10 15)
MODEL="mlx-community/gemma-3-12b-it-4bit"
BASE_URL="http://localhost:10240/v1"
IMAGE="/Users/cloudsineai/Documents/mlx-omni-server/examples/stickman.png"

# Enable streaming so TTFB is measured and printed by the benchmark
# export STREAMING=true

# for c in "${CONCURRENCIES[@]}"; do
# echo "Running vision benchmark with rounds=${ROUNDS}, concurrency=${c}"
# python examples/benchmark_vision.py \
#   --rounds "${ROUNDS}" \
#   --concurrency "${c}" \
#   --model "${MODEL}" \
#   --base-url "${BASE_URL}" \
#   --image "${IMAGE}"
# echo ""
# done



# # Enable streaming so TTFB is measured and printed by the benchmark
export STREAMING=true

for c in "${CONCURRENCIES[@]}"; do
echo "Running chat benchmark with rounds=${ROUNDS}, concurrency=${c}"
python examples/benchmark_chat.py \
  --rounds "${ROUNDS}" \
  --concurrency "${c}" \
  --model "${MODEL}" \
  --base-url "${BASE_URL}" \
  --vary-prompt
echo ""
done