import argparse
import asyncio
import atexit
import os
import signal
import sys
from itertools import cycle
from typing import Iterable, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from mlx_omni_server.utils import logger
from .middleware.logging import RequestResponseLoggingMiddleware
from contextlib import asynccontextmanager


# ---> CLI entry `mlx-omni-cluster` > [build_parser] > parsed by start()
def build_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for the cluster launcher."""
    parser = argparse.ArgumentParser(description="MLX Omni Cluster Launcher")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the proxy server to, defaults to 0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10240,
        help="Port for the proxy (front door). Defaults to 10240",
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=2,
        help="Number of backend replicas to launch",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=20240,
        help="Starting port for backend replicas. i-th replica uses base-port+i",
    )
    parser.add_argument(
        "--backend-host",
        type=str,
        default="127.0.0.1",
        help="Host interface for backend replicas",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Uvicorn workers per backend replica",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set logging level for backend replicas and proxy",
    )
    parser.add_argument(
        "--cors-allow-origins",
        type=str,
        default="",
        help='Apply origins to CORSMiddleware. CSV, or "*". Defaults to disabled',
    )
    parser.add_argument(
        "--max-inflight-per-replica",
        type=int,
        default=1,
        help="Maximum concurrent requests per replica (1 to avoid queuing at backend)",
    )
    return parser


def _split_origins(cors_allow_origins: Optional[str]) -> List[str]:
    if cors_allow_origins is None:
        return []
    cors_allow_origins = cors_allow_origins.strip()
    if not cors_allow_origins:
        return []
    return [origin.strip() for origin in cors_allow_origins.split(",") if origin.strip()]


# ---> start() > [launch_backends] > Popen child processes for replicas
def launch_backends(
    *,
    replicas: int,
    backend_host: str,
    base_port: int,
    workers: int,
    log_level: str,
    cors_allow_origins: str,
) -> List[asyncio.subprocess.Process]:
    """Launch multiple backend server instances as child processes.

    Returns a list of asyncio subprocess handles for later cleanup.
    """
    processes: List[asyncio.subprocess.Process] = []
    # Use the package module entry to ensure identical behavior
    cmd_base = [
        sys.executable,
        "-m",
        "mlx_omni_server.main",
        "--host",
        backend_host,
        "--workers",
        str(workers),
        "--log-level",
        log_level,
        "--cors-allow-origins",
        cors_allow_origins,
    ]

    async def _spawn_one(port: int) -> asyncio.subprocess.Process:
        cmd = cmd_base + ["--port", str(port)]
        # Start in new session to manage signals cleanly
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=None,
            stderr=None,
            start_new_session=True,
        )
        return proc

    async def _spawn_all() -> List[asyncio.subprocess.Process]:
        tasks = []
        for i in range(replicas):
            port = base_port + i
            tasks.append(_spawn_one(port))
        return await asyncio.gather(*tasks)

    loop = asyncio.get_event_loop()
    processes = loop.run_until_complete(_spawn_all())

    # Ensure child processes are terminated on exit
    def _cleanup_children() -> None:
        for proc in processes:
            try:
                if proc.returncode is None:
                    # Send SIGTERM to process
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass

    atexit.register(_cleanup_children)
    return processes


async def _stream_upstream_response(upstream: httpx.Response) -> Iterable[bytes]:
    async for chunk in upstream.aiter_raw():
        if chunk:
            yield chunk


# ---> start() > [make_proxy_app] > uvicorn serves app; requests forwarded to backends
def make_proxy_app(
    backend_urls: List[str],
    cors_allow_origins: str,
    max_inflight_per_replica: int = 1,
) -> FastAPI:
    app = FastAPI(title="MLX Omni Cluster Proxy")

    origins = _split_origins(cors_allow_origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(
        RequestResponseLoggingMiddleware,
        exclude_paths=["/health", "/metrics"]
    )

    # ---> make_proxy_app > [ReplicaScheduler] > used by _proxy_request for selection & queueing
    class ReplicaScheduler:
        def __init__(self, urls: List[str], max_inflight: int) -> None:
            self.urls = urls
            self.max_inflight = max(1, int(max_inflight))
            self._inflight: List[int] = [0] * len(urls)
            self._rr_index: int = 0
            self._cond = asyncio.Condition()

        def _next_indices_rr(self) -> List[int]:
            n = len(self.urls)
            start = self._rr_index % n
            return [(start + i) % n for i in range(n)]

        # ---> _proxy_request > [acquire] > blocks until free slot
        async def acquire(self) -> int:
            start_wait = asyncio.get_event_loop().time()
            async with self._cond:
                while True:
                    for idx in self._next_indices_rr():
                        if self._inflight[idx] < self.max_inflight:
                            self._inflight[idx] += 1
                            self._rr_index = (idx + 1) % len(self.urls)
                            wait_ms = (asyncio.get_event_loop().time() - start_wait) * 1000
                            print(f"Acquired replica {idx} ({self.urls[idx]}), wait {wait_ms:.0f}ms, in-flight: {self._inflight}")
                            return idx
                    await self._cond.wait()

        # ---> _proxy_request > [try_acquire_excluding] > immediate if free, else None
        async def try_acquire_excluding(self, exclude: List[int]) -> Optional[int]:
            async with self._cond:
                for idx in self._next_indices_rr():
                    if idx in exclude: continue
                    if self._inflight[idx] < self.max_inflight:
                        self._inflight[idx] += 1
                        self._rr_index = (idx + 1) % len(self.urls)
                        return idx
                return None

        # ---> _proxy_request > [release] > decrement and notify
        async def release(self, idx: int) -> None:
            async with self._cond:
                if 0 <= idx < len(self._inflight):
                    self._inflight[idx] = max(0, self._inflight[idx] - 1)
                    print(f"Released replica {idx} ({self.urls[idx]}), in-flight: {self._inflight}")
                self._cond.notify_all()

    scheduler = ReplicaScheduler(backend_urls, max_inflight_per_replica)

    client = httpx.AsyncClient(timeout=None)

    @asynccontextmanager
    async def lifespan(app_: FastAPI):  # Note: app_ to avoid shadowing
        yield
        await client.aclose()

    app.router.lifespan = lifespan

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "backends": backend_urls}

    @app.get("/metrics")
    async def metrics() -> dict:
        return {
            "replicas": [
                {"url": url, "inflight": scheduler._inflight[i]}
                for i, url in enumerate(backend_urls)
            ]
        }

    # ---> app.api_route > [_proxy_request] > acquire slot, proxy, release
    async def _proxy_request(request: Request) -> Response:
        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("content-length", None)
        headers.pop("accept-encoding", None)

        body = await request.body()
        method = request.method

        last_exc: Optional[Exception] = None
        tried_indices: List[int] = []

        # Acquire first slot (queue if needed)
        idx = await scheduler.acquire()
        tried_indices.append(idx)
        upstream_base = backend_urls[idx]
        upstream_url = httpx.URL(upstream_base).join(request.url.path)

        try:
            req = client.build_request(
                method=method,
                url=str(upstream_url),
                headers=headers,
                params=list(request.query_params.multi_items()),
                content=body if body else None,
            )
            upstream = await client.send(req, stream=True)

            excluded_headers = {
                "content-encoding", "content-length", "transfer-encoding", "connection"
            }
            response_headers = [(k, v) for k, v in upstream.headers.items() if k.lower() not in excluded_headers]
            response_headers.append(("X-Upstream-Replica", backend_urls[idx]))

            async def _iterator():
                try:
                    async for chunk in upstream.aiter_raw():
                        if chunk: yield chunk
                finally:
                    await upstream.aclose()
                    await scheduler.release(idx)

            return StreamingResponse(
                _iterator(),
                status_code=upstream.status_code,
                headers=dict(response_headers),
            )
        except Exception as exc:
            last_exc = exc
            await scheduler.release(idx)

            # Retry others if immediate capacity
            for _ in range(len(backend_urls) - 1):
                next_idx = await scheduler.try_acquire_excluding(tried_indices)
                if next_idx is None: break
                tried_indices.append(next_idx)
                upstream_base = backend_urls[next_idx]
                upstream_url = httpx.URL(upstream_base).join(request.url.path)
                try:
                    req = client.build_request(
                        method=method,
                        url=str(upstream_url),
                        headers=headers,
                        params=list(request.query_params.multi_items()),
                        content=body if body else None,
                    )
                    upstream = await client.send(req, stream=True)

                    excluded_headers = {
                        "content-encoding", "content-length", "transfer-encoding", "connection"
                    }
                    response_headers = [(k, v) for k, v in upstream.headers.items() if k.lower() not in excluded_headers]
                    response_headers.append(("X-Upstream-Replica", backend_urls[next_idx]))

                    async def _iterator2():
                        try:
                            async for chunk in upstream.aiter_raw():
                                if chunk: yield chunk
                        finally:
                            await upstream.aclose()
                            await scheduler.release(next_idx)

                    return StreamingResponse(
                        _iterator2(),
                        status_code=upstream.status_code,
                        headers=dict(response_headers),
                    )
                except Exception as exc2:
                    last_exc = exc2
                    await scheduler.release(next_idx)
                    continue

            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=502,
                content={"error": "Bad Gateway", "detail": str(last_exc) if last_exc else None},
            )

    # Register catch-all route for all methods
    @app.api_route("/{full_path:path}", methods=[
        "GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"
    ])
    async def catch_all(full_path: str, request: Request) -> Response:  # noqa: ARG001
        return await _proxy_request(request)

    return app


# ---> CLI entry `mlx-omni-cluster` > [start] > uvicorn runs proxy, child procs run backends
def start() -> None:
    """Start the MLX Omni cluster proxy and backend replicas."""
    parser = build_parser()
    args = parser.parse_args()

    set_proxy_log_level = args.log_level

    # Derive backend URLs and launch replicas
    backend_ports = [args.base_port + i for i in range(args.replicas)]

    # Launch backends as subprocesses
    launch_backends(
        replicas=args.replicas,
        backend_host=args.backend_host,
        base_port=args.base_port,
        workers=args.workers,
        log_level=args.log_level,
        cors_allow_origins=args.cors_allow_origins,
    )

    backend_urls = [
        f"http://{args.backend_host}:{port}"
        for port in backend_ports
    ]

    app = make_proxy_app(
        backend_urls,
        args.cors_allow_origins,
        max_inflight_per_replica=args.max_inflight_per_replica,
    )

    # Run proxy server
    import logging
    logging.basicConfig(level="DEBUG" if args.log_level == "debug" else "INFO")
    logging.getLogger().setLevel(logging.DEBUG if args.log_level == "debug" else logging.INFO)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=set_proxy_log_level,
        use_colors=True,
        workers=1,
    )


# Expose `app` for uvicorn dotted-path reference
app = make_proxy_app(
    ["http://127.0.0.1:20240"],
    os.environ.get("MLX_OMNI_CORS", ""),
    max_inflight_per_replica=int(os.environ.get("MLX_OMNI_MAX_INFLIGHT", "1")),
)


if __name__ == "__main__":
    start()


