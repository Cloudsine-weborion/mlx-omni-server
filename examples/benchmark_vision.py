#!/usr/bin/env python3
"""
Benchmark concurrent vision chat completion requests against the MLX Omni Server.

Simple usage:
  python examples/benchmark_vision.py --rounds 5 --concurrency 4 \
    --model mlx-community/gemma-3-9b-pt --base-url http://localhost:10240/v1 \
    --image examples/cow.jpg

Options:
  --rounds N          Number of rounds to run (default: 5)
  --concurrency N     Number of parallel requests per batch (default: 2)
  --requests N        Requests per round (default: equals --concurrency)
  --model NAME        Model name served by the server (vision-capable)
  --base-url URL      Base URL of server (default: http://localhost:10240/v1)
  --image PATH        Path to the image file (default: examples/cow.jpg)
  --pid PID           Server PID to sample memory (optional; auto-detect by port if missing)
  --vary-prompt       Vary prompt slightly per request to avoid cache effects

The script prints per-round latency stats and a memory RSS snapshot of the server
process(es) listening on the server port. Memory readings use lsof/ps (no extra deps).
"""

from __future__ import annotations

import argparse
import base64
import os
import statistics
import mimetypes
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from openai import OpenAI
import httpx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prompts pool used by choose_base_text (vision-specific phrasings)
PROMPTS: List[str] = [
    "Describe this image briefly.",
    "Provide a short caption for this image.",
    "Summarize what is shown in the image in one sentence.",
    "Give a concise description of this picture.",
    "What is depicted in this image? Keep it brief.",
    "Write a short description of the image.",
    "In one line, describe the content of this image.",
    "Offer a brief explanation of what the image shows.",
    "Create a short caption describing the image.",
    "What is the main subject of this image?",
    "Describe the scene in this photo succinctly.",
    "Provide a compact summary of the image contents.",
    "Generate a concise caption for the picture.",
    "Briefly explain what can be seen in the image.",
    "In a single sentence, what does this image show?",
    "Give a one-sentence description of this image.",
    "Provide a minimal caption for this picture.",
    "Describe the key elements in this image briefly.",
    "What is happening in this image? Short answer.",
    "Write a brief alt-text for this image.",
    "Summarize the visual content in a short phrase.",
    "Give a short, clear description of this image.",
    "Offer a succinct description of the picture.",
    "Provide a brief summary of what the image depicts.",
    "Describe this image in under 15 words.",
    "Provide a compact alt description of the image.",
    "Give a terse caption for what this image shows.",
    "Briefly identify the subject and context of the image.",
    "State what the image contains in one short line.",
    "Provide a minimalistic description of this image.",
]

# ---> perform_chat > [choose_base_text] > select unique base_text per request
def choose_base_text(global_index: int) -> str:
    return PROMPTS[global_index % len(PROMPTS)]

# ---> CLI entry > [build_arg_parser] > argparse parses CLI flags for benchmark
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Concurrent vision benchmark for MLX Omni Server")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds to run")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Number of parallel requests per batch",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=None,
        help="Requests per round (defaults to --concurrency if not provided)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MLX_BENCH_MODEL", "mlx-community/gemma-3-9b-pt"),
        help="Model name to query (vision-capable)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("MLX_BENCH_BASE_URL", "http://localhost:10240/v1"),
        help="Base URL of server (e.g., http://localhost:10240/v1)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="examples/stickman.png",
        help="Path to the image file for vision input",
    )
    parser.add_argument("--pid", type=int, default=None, help="Server PID for memory sampling (optional)")
    parser.add_argument(
        "--vary-prompt",
        action="store_true",
        help="Vary prompt per request to reduce caching effects",
    )
    return parser


@dataclass
class RequestResult:
    index: int
    latency_s: float
    ttfb_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error: Optional[str] = None
    # Absolute timestamps for generation window calculations
    start_ts: float = 0.0
    first_byte_ts: float = 0.0
    end_ts: float = 0.0


# ---> main() > [parse_port_from_base_url] > derive port for lsof/ps memory sampling
def parse_port_from_base_url(base_url: str) -> Optional[int]:
    parsed = urlparse(base_url)
    if parsed.port is not None:
        return parsed.port
    # Fallbacks if no explicit port provided
    if parsed.scheme == "http":
        return 80
    if parsed.scheme == "https":
        return 443
    return None


# ---> main() > [snapshot_server_memory] > calls lsof/ps to read server RSS
def _find_listening_pids_on_port(port: int) -> List[int]:
    # Prefer compact PID-only output
    commands: List[List[str]] = [
        ["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN", "-n", "-P"],
        ["lsof", "-t", "-i", f":{port}", "-sTCP:LISTEN", "-n", "-P"],
    ]
    for cmd in commands:
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
            pids = sorted({int(line.strip()) for line in out.splitlines() if line.strip().isdigit()})
            if pids:
                return pids
        except Exception:
            continue
    return []


def _rss_kb_for_pid(pid: int) -> Optional[int]:
    try:
        # macOS: rss in KB; the '=' removes header
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True, stderr=subprocess.DEVNULL)
        value = out.strip()
        if value:
            return int(value.split()[0])
    except Exception:
        return None
    return None


def snapshot_server_memory(port: Optional[int], explicit_pid: Optional[int]) -> Tuple[float, List[int]]:
    """Return (rss_mb_total, pids) for server processes listening on port or explicit PID.

    If explicit_pid is provided, only that PID is sampled. Otherwise, all listening PIDs
    for the port are sampled and summed.
    """
    pids: List[int] = []
    if explicit_pid is not None:
        pids = [explicit_pid]
    elif port is not None:
        pids = _find_listening_pids_on_port(port)

    total_kb = 0
    for pid in pids:
        kb = _rss_kb_for_pid(pid)
        if kb is not None:
            total_kb += kb

    rss_mb = round(total_kb / 1024.0, 2) if total_kb > 0 else 0.0
    return rss_mb, pids


# ---> main() > [build_client] > OpenAI client targets MLX Omni Server
def build_client(base_url: str) -> OpenAI:
    return OpenAI(
        base_url=base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
        timeout=httpx.Timeout(60.0),
    )


def load_image_base64(image_path: str) -> str:
    """Load and base64 encode the image file."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def perform_chat(client: OpenAI, model: str, index: int, global_index: int, image_path: Optional[str], vary_prompt: bool, streaming: bool) -> RequestResult:
    # Build user message content, optionally with image attachment
    has_image = image_path is not None
    base_text = choose_base_text(global_index) if has_image else "Write a brief description of a cow."
    text = base_text  # We follow chat benchmark: choose from PROMPTS for natural variation

    messages: List[dict]
    if has_image:
        try:
            base64_image = load_image_base64(image_path)  # type: ignore[arg-type]
        except Exception as e:
            return RequestResult(index=index, latency_s=0.0, ttfb_s=0.0, error=f"Failed to load image: {str(e)}")

        mime, _ = mimetypes.guess_type(image_path)  # type: ignore[arg-type]
        if mime is None:
            mime = "image/jpeg"
        data_url = f"data:{mime};base64,{base64_image}"
        content = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": data_url},  # string value as required by server schema
        ]
        messages = [{"role": "user", "content": content}]
    else:
        messages = [{"role": "user", "content": text}]

    logger.info(f"Starting vision request {index}")
    start = time.perf_counter()

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    ttfb = 0.0
    first_byte_ts = 0.0

    try:
        if streaming:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )

            first_content = True
            full_content = ""
            for chunk in completion:
                if first_content and chunk.choices[0].delta.content is not None:
                    ttfb = time.perf_counter() - start
                    first_byte_ts = start + ttfb
                    first_content = False

                if chunk.choices[0].delta.content is not None:
                    full_content += chunk.choices[0].delta.content

                # Extract usage from chunks if available
                if hasattr(chunk, 'usage') and chunk.usage:
                    if chunk.usage.prompt_tokens:
                        prompt_tokens = chunk.usage.prompt_tokens
                    if chunk.usage.completion_tokens:
                        completion_tokens += chunk.usage.completion_tokens  # Delta
                    if hasattr(chunk.usage, 'total_tokens'):
                        total_tokens = chunk.usage.total_tokens

            # Fallback if no usage in stream
            if total_tokens == 0:
                prompt_tokens = len(text.split())  # Approximate
                completion_tokens = len(full_content.split())
                total_tokens = prompt_tokens + completion_tokens

            end = time.perf_counter()
            latency = end - start
            error = "response took longer than 10s" if latency > 10 else None
            logger.info(f"Vision request {index} completed (streaming)" if error is None else f"Vision request {index} slow (streaming)")
            if not full_content:
                raise ValueError("No content received in stream")
        else:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            # Accessing the message confirms a well-formed response
            _ = completion.choices[0].message
            # Capture token usage if provided
            usage = getattr(completion, "usage", None)
            if usage is not None:
                try:
                    prompt_tokens = int(getattr(usage, "prompt_tokens", 0))
                    completion_tokens = int(getattr(usage, "completion_tokens", 0))
                    total_tokens = int(getattr(usage, "total_tokens", 0))
                except Exception:
                    # Fallback if usage is a dict-like
                    if isinstance(usage, dict):
                        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                        total_tokens = int(usage.get("total_tokens", 0) or 0)

            end = time.perf_counter()
            latency = end - start
            ttfb = 0.0  # No TTFB in non-streaming
            error = "response took longer than 10s" if latency > 10 else None
            logger.info(f"Vision request {index} completed" if error is None else f"Vision request {index} slow")

        return RequestResult(
            index=index,
            latency_s=latency,
            ttfb_s=ttfb,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            error=error,
            start_ts=start,
            first_byte_ts=first_byte_ts if first_byte_ts > 0 else start,
            end_ts=end,
        )
    except Exception as e:  # non-trivial: capture server/client errors
        end = time.perf_counter()
        logger.error(f"Vision request {index} failed after {end - start:.2f}s: {str(e)}")
        return RequestResult(index=index, latency_s=end - start, ttfb_s=0.0, error=str(e), start_ts=start, first_byte_ts=start, end_ts=end)


# ---> main() > [run_round] > ThreadPoolExecutor coordinates parallel requests
def run_round(
    client: OpenAI, model: str, num_requests: int, concurrency: int, image_path: Optional[str], vary_prompt: bool, streaming: bool
    , start_index: int
) -> Tuple[List[RequestResult], float]:
    results: List[RequestResult] = []
    started_at = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                perform_chat,
                client=client,
                model=model,
                index=i,
                global_index=start_index + i,
                image_path=image_path,
                vary_prompt=vary_prompt,
                streaming=streaming,
            )
            for i in range(num_requests)
        ]
        for fut in as_completed(futures):
            results.append(fut.result())
    finished_at = time.perf_counter()
    return results, finished_at - started_at


# ---> main() > [summarize_latencies] > compute avg/p50/p95/p99/min/max and errors
def summarize_latencies(results: List[RequestResult]) -> dict:
    latencies = [r.latency_s for r in results if r.error is None]
    ttfb_list = [r.ttfb_s for r in results if r.error is None and r.ttfb_s > 0]
    errors = [r for r in results if r.error]
    count = len(results)

    def pct(values: List[float], p: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        # Nearest-rank percentile (1-based rank)
        rank = max(1, min(len(values), int(round(p * len(values)))))
        return values[rank - 1]

    summary = {
        "count": count,
        "avg_s": round(statistics.fmean(latencies), 4) if latencies else 0.0,
        "p50_s": round(pct(latencies, 0.50), 4),
        "p95_s": round(pct(latencies, 0.95), 4),
        "p99_s": round(pct(latencies, 0.99), 4),
        "min_s": round(min(latencies), 4) if latencies else 0.0,
        "max_s": round(max(latencies), 4) if latencies else 0.0,
        "errors": len(errors),
        "error_rate": (len(errors) / count) if count > 0 else 0.0,
    }

    if ttfb_list:
        summary.update({
            "avg_ttfb_s": round(statistics.fmean(ttfb_list), 4),
            "p50_ttfb_s": round(pct(ttfb_list, 0.50), 4),
            "p95_ttfb_s": round(pct(ttfb_list, 0.95), 4),
            "p99_ttfb_s": round(pct(ttfb_list, 0.99), 4),
            "min_ttfb_s": round(min(ttfb_list), 4),
            "max_ttfb_s": round(max(ttfb_list), 4),
        })

    return summary


# ---> main() > [compute_generation_window_s] > duration from first byte to last end
def compute_generation_window_s(results: List[RequestResult]) -> float:
    """Compute round duration as time from earliest first byte to latest end.

    Falls back to wall duration from min(start)->max(end) if no first bytes present.
    """
    successes = [r for r in results if r.error is None]
    if not successes:
        return 0.0
    gen_start = min((r.first_byte_ts if r.first_byte_ts > 0 else r.start_ts) for r in successes)
    gen_end = max((r.end_ts if r.end_ts > 0 else (r.start_ts + r.latency_s)) for r in successes)
    duration = max(0.0, gen_end - gen_start)
    return duration if duration > 1e-9 else 0.0

# ---> CLI entry > [main] > orchestrates rounds, metrics, and memory snapshots
def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    num_rounds = max(1, args.rounds)
    concurrency = max(1, args.concurrency)
    requests_per_round = args.requests if args.requests is not None else concurrency

    base_url = args.base_url
    model = args.model
    image_path = args.image
    vary_prompt = bool(args.vary_prompt)
    streaming = os.environ.get('STREAMING', 'false').lower() == 'true'

    port = parse_port_from_base_url(base_url)
    client = build_client(base_url)

    print(f"Benchmarking vision model='{model}' at base_url='{base_url}' with image='{image_path}'")
    print(f"Rounds={num_rounds}, Requests/Round={requests_per_round}, Concurrency={concurrency}")
    if streaming:
        print("Mode: Streaming (TTFB measured)")
    if port is not None:
        print(f"Memory sampling targets processes listening on port {port} (or PID={args.pid} if provided)")
    else:
        print("Memory sampling disabled (no port inferred)")
    print("")

    overall_latencies: List[float] = []
    overall_ttfbs: List[float] = []
    overall_errors: int = 0
    overall_requests: int = 0
    total_wall_time: float = 0.0
    total_generation_window_s: float = 0.0
    overall_completion_tokens: int = 0

    for r in range(1, num_rounds + 1):
        print(f"Round {r}/{num_rounds}:")
        results, round_wall_s = run_round(
            client=client,
            model=model,
            num_requests=requests_per_round,
            concurrency=concurrency,
            image_path=image_path,
            vary_prompt=vary_prompt,
            streaming=streaming,
            start_index=overall_requests,
        )
        total_wall_time += round_wall_s
        round_gen_s = compute_generation_window_s(results)
        total_generation_window_s += round_gen_s
        summary = summarize_latencies(results)
        overall_requests += summary["count"]
        overall_errors += summary["errors"]
        overall_latencies.extend([res.latency_s for res in results if res.error is None])
        if streaming:
            overall_ttfbs.extend([res.ttfb_s for res in results if res.error is None and res.ttfb_s > 0])

        # Throughput based on generation window (first byte -> end of stream)
        tput = round(summary["count"] / round_gen_s, 2) if round_gen_s > 0 else 0.0
        print(
            f"  Latency avg={summary['avg_s']}s p50={summary['p50_s']}s p95={summary['p95_s']}s p99={summary['p99_s']}s min={summary['min_s']}s max={summary['max_s']}s"
        )
        if "avg_ttfb_s" in summary:
            print(
                f"  TTFB avg={summary['avg_ttfb_s']}s p50={summary['p50_ttfb_s']}s p95={summary['p95_ttfb_s']}s p99={summary['p99_ttfb_s']}s min={summary['min_ttfb_s']}s max={summary['max_ttfb_s']}s"
            )
        print(f"  Errors={summary['errors']}  Round wall={round(round_wall_s, 3)}s  Gen window={round(round_gen_s, 3)}s  Throughput={tput} req/s")

        # Tokens/sec (generation throughput): sum of completion tokens divided by round wall time
        round_completion_tokens = sum(r.completion_tokens for r in results)
        overall_completion_tokens += round_completion_tokens
        tok_tput = round(round_completion_tokens / round_wall_s, 2) if round_wall_s > 0 else 0.0
        print(f"  Tokens: completion={round_completion_tokens}  Gen throughput={tok_tput} tok/s")

        rss_mb, pids = snapshot_server_memory(port=port, explicit_pid=args.pid)
        pids_str = ",".join(str(pid) for pid in pids) if pids else "-"
        print(f"  Server RSS after round: {rss_mb} MB  (PIDs: {pids_str})")
        print("")

    # Overall summary
    if overall_requests > 0:
        print("Overall:")
        if overall_latencies:
            overall_latencies.sort()
            def pct(values: List[float], p: float) -> float:
                if not values:
                    return 0.0
                if len(values) == 1:
                    return values[0]
                rank = max(1, min(len(values), int(round(p * len(values)))))
                return values[rank - 1]

            overall_avg = round(statistics.fmean(overall_latencies), 4)
            overall_p50 = round(pct(overall_latencies, 0.50), 4)
            overall_p95 = round(pct(overall_latencies, 0.95), 4)
            overall_p99 = round(pct(overall_latencies, 0.99), 4)
            overall_min = round(overall_latencies[0], 4)
            overall_max = round(overall_latencies[-1], 4)
            print(
                f"  Latency avg={overall_avg}s p50={overall_p50}s p95={overall_p95}s p99={overall_p99}s min={overall_min}s max={overall_max}s"
            )
        else:
            print("  No successful requests to report latency stats.")

        if overall_ttfbs:
            overall_ttfbs.sort()
            def pct_ttfb(values: List[float], p: float) -> float:
                if not values:
                    return 0.0
                if len(values) == 1:
                    return values[0]
                rank = max(1, min(len(values), int(round(p * len(values)))))
                return values[rank - 1]
            overall_avg_ttfb = round(statistics.fmean(overall_ttfbs), 4)
            overall_p50_ttfb = round(pct_ttfb(overall_ttfbs, 0.50), 4)
            overall_p95_ttfb = round(pct_ttfb(overall_ttfbs, 0.95), 4)
            overall_p99_ttfb = round(pct_ttfb(overall_ttfbs, 0.99), 4)
            overall_min_ttfb = round(overall_ttfbs[0], 4)
            overall_max_ttfb = round(overall_ttfbs[-1], 4)
            print(
                f"  TTFB avg={overall_avg_ttfb}s p50={overall_p50_ttfb}s p95={overall_p95_ttfb}s p99={overall_p99_ttfb}s min={overall_min_ttfb}s max={overall_max_ttfb}s"
            )

        overall_tput = round(overall_requests / total_generation_window_s, 2) if total_generation_window_s > 0 else 0.0
        print(
            f"  Errors={overall_errors}  Total wall={round(total_wall_time, 3)}s  Gen window total={round(total_generation_window_s, 3)}s  Throughput={overall_tput} req/s"
        )
        overall_tok_tput = round(overall_completion_tokens / total_generation_window_s, 2) if total_generation_window_s > 0 else 0.0
        print(f"  Tokens: completion={overall_completion_tokens}  Gen throughput={overall_tok_tput} tok/s")


if __name__ == "__main__":
    # ---> CLI > [main] > run benchmark with provided arguments
    main()

