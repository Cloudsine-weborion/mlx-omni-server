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
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
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
def make_proxy_app(backend_urls: List[str], cors_allow_origins: str) -> FastAPI:
    app = FastAPI(title="MLX Omni Cluster Proxy")

    origins = _split_origins(cors_allow_origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    rr = cycle(backend_urls)
    client = httpx.AsyncClient(timeout=None)

    @app.on_event("shutdown")
    async def _close_client() -> None:
        await client.aclose()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "backends": backend_urls}

    async def _proxy_request(request: Request) -> Response:
        # Choose next backend; on failure try all backends once
        last_exc: Optional[Exception] = None
        for _ in range(len(backend_urls)):
            upstream_base = next(rr)
            upstream_url = httpx.URL(upstream_base).join(request.url.path)

            # Prepare headers, dropping hop-by-hop
            headers = dict(request.headers)
            headers.pop("host", None)
            headers.pop("content-length", None)
            headers.pop("accept-encoding", None)

            try:
                # Read body only once
                body = await request.body()
                method = request.method

                # Use streaming send to avoid consuming the response prematurely
                req = client.build_request(
                    method=method,
                    url=str(upstream_url),
                    headers=headers,
                    params=list(request.query_params.multi_items()),
                    content=body if body else None,
                )
                upstream = await client.send(req, stream=True)

                excluded_headers = {
                    "content-encoding",
                    "content-length",
                    "transfer-encoding",
                    "connection",
                }
                response_headers = [
                    (k, v)
                    for k, v in upstream.headers.items()
                    if k.lower() not in excluded_headers
                ]

                async def _iterator():
                    try:
                        async for chunk in upstream.aiter_raw():
                            if chunk:
                                yield chunk
                    finally:
                        await upstream.aclose()

                return StreamingResponse(
                    _iterator(),
                    status_code=upstream.status_code,
                    headers=dict(response_headers),
                )
            except Exception as exc:  # try next backend
                last_exc = exc
                continue

        # If we get here, all backends failed
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

    app = make_proxy_app(backend_urls, args.cors_allow_origins)

    # Run proxy server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=set_proxy_log_level,
        use_colors=True,
        workers=1,
    )


# Expose `app` for uvicorn dotted-path reference
app = make_proxy_app(["http://127.0.0.1:20240"], os.environ.get("MLX_OMNI_CORS", ""))


if __name__ == "__main__":
    start()


