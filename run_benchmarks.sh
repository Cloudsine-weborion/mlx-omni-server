#!/bin/bash

# Run MLX Omni Server benchmarks with different parameters
# Modify the --rounds and --concurrency values as needed

echo "Running benchmark with rounds=5, concurrency=4"
python examples/benchmark_chat.py --rounds 5 --concurrency 4 --model mlx-community/gemma-3-12b-it-4bit

echo ""
echo "Running benchmark with rounds=3, concurrency=2"
python examples/benchmark_chat.py --rounds 3 --concurrency 2 --model mlx-community/gemma-3-12b-it-4bit

echo ""
echo "Running benchmark with rounds=10, concurrency=1"
python examples/benchmark_chat.py --rounds 10 --concurrency 1 --model mlx-community/gemma-3-12b-it-4bit

echo ""
echo "Running benchmark with rounds=1, concurrency=8"
python examples/benchmark_chat.py --rounds 1 --concurrency 8 --model mlx-community/gemma-3-12b-it-4bit

# Add more benchmark runs below as needed
# python examples/benchmark_chat.py --rounds X --concurrency Y --model mlx-community/gemma-3-12b-it-4bit
