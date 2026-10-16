from __future__ import annotations

import asyncio
import atexit
import os
import re
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Coroutine, Iterable

import aiohttp
from sanic import HTTPResponse, Request
from sanic.log import logger

from api import Package


def _find_free_port():
    with socket.socket() as s:
        s.bind(("", 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.


def _port_in_use(port: int):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


SANIC_LOG_REGEX = re.compile(r"^\s*\[[^\[\]]*\] \[\d*\] \[(\w*)\] (.*)")

# backslashreplace: a log line with a lone surrogate (an undecodable Windows file
# name) is written escaped instead of being dropped.
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8:backslashreplace"}


class _WorkerProcess:
    def __init__(self, flags: list[str]):
        server_file = os.path.join(os.path.dirname(__file__), "server.py")
        python_location = sys.executable

        self._process = subprocess.Popen(
            # A host that is killed or crashes stops nothing: the worker watches it.
            [python_location, server_file, *flags, "--host-pid", str(os.getpid())],
            shell=False,
            stdin=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            # A byte that is not UTF-8 (native code, a tool the worker runs) is shown
            # escaped instead of ending the reader threads.
            errors="backslashreplace",
            env=ENV,
        )
        self._stop_event = threading.Event()

        # Create a separate thread to read and print the output of the subprocess
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            daemon=True,
            name="stdout reader",
        )
        self._stdout_thread.start()
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            daemon=True,
            name="stderr reader",
        )
        self._stderr_thread.start()

        atexit.register(self.close)

    def close(self):
        if self._process is None:
            # already closed
            return

        logger.info("Closing worker process...")
        self._stop_event.set()
        try:
            self._process.terminate()
            self._process.kill()
        except Exception:
            logger.error("Failed to terminate worker process", exc_info=True)
        self._process = None
        atexit.unregister(self.close)

        self._stdout_thread = None
        self._stderr_thread = None

    def _read_stdout(self):
        p = self._process
        if p is None or p.stdout is None:
            return

        stopped = False
        for line in p.stdout:
            stopped = self._stop_event.is_set()
            if stopped:
                break

            stripped_line = line.rstrip()
            match_obj = re.match(SANIC_LOG_REGEX, stripped_line)
            if match_obj is not None:
                log_level, message = match_obj.groups()
                message = f"[Worker] {message}"
                if log_level == "DEBUG":
                    logger.debug(message)
                elif log_level == "INFO":
                    logger.info(message)
                elif log_level == "WARNING":
                    logger.warning(message)
                elif log_level == "ERROR":
                    logger.error(message)
                elif log_level == "CRITICAL":
                    logger.critical(message)
                else:
                    logger.info(message)
            else:
                logger.info(f"[Worker] {stripped_line}")

        cause = "stop event" if stopped else "stdout ending"
        logger.info(f"Stopped reading worker stdout due to {cause}")

        stopped = self._stop_event.is_set()
        if not stopped:
            # the worker ended on its own, so it likely crashed
            returncode = p.wait()
            if returncode == 0:
                logger.info("Worker process ended normally")
            else:
                logger.error(
                    f"Worker process ended with non-zero return code {returncode}"
                )

    def _read_stderr(self):
        p = self._process
        if p is None or p.stderr is None:
            return

        stopped = False
        for line in p.stderr:
            stopped = self._stop_event.is_set()
            if stopped:
                break

            stripped_line = line.rstrip()
            logger.error(f"[Worker] {stripped_line}")

        cause = "stop event" if stopped else "stderr ending"
        logger.info(f"Stopped reading worker stderr due to {cause}")


def _log_pip_outcome(job: asyncio.Task[object]) -> None:
    """Reports how a pip run ended after the request that started it closed."""
    if job.cancelled():
        logger.error("Dependency change was cancelled after its request closed")
    elif (error := job.exception()) is not None:
        logger.error(f"Dependency change failed after its request closed: {error}")
    else:
        logger.info("Dependency change finished after its request closed")


class WorkerServer:
    def __init__(self, flags: Iterable[str] = []):
        self._process = None

        self._port = _find_free_port()
        self._base_url = f"http://127.0.0.1:{self._port}"
        self._flags = list(flags)
        self._session = None
        self._is_ready = False
        self._is_checking_ready = False
        self._manually_close: set[aiohttp.ClientResponse] = set()
        # The pip runs that change the worker's packages (run_pip_job).
        self._pip_jobs: set[asyncio.Task[object]] = set()

    async def run_pip_job(self, job: Coroutine[Any, Any, object]) -> None:
        """Runs a pip install or uninstall as a task of its own. A cancelled caller
        (the app's request, closed by a reload) neither stops pip mid-install nor
        lets the worker start under it: start() waits for the job, and the job's
        outcome is logged once it ends."""
        task = asyncio.ensure_future(job)
        self._pip_jobs.add(task)
        task.add_done_callback(self._pip_jobs.discard)
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            task.add_done_callback(_log_pip_outcome)
            raise

    async def start(self, extra_flags: Iterable[str] = []):
        # Never while pip changes the packages: the worker would import them half
        # installed.
        while self._pip_jobs:
            await asyncio.wait(set(self._pip_jobs))
        # Overlapping restarts (a second /kill while the first restarts) each start a
        # worker: one started meanwhile is stopped, not left running owned by nothing.
        # No await between this check and the assignments below.
        while self._process is not None or self._session is not None:
            await self.stop()
        logger.info(f"Starting worker process on port {self._port}...")
        self._process = _WorkerProcess([str(self._port), *self._flags, *extra_flags])
        self._session = aiohttp.ClientSession(base_url=self._base_url)
        self._is_ready = False
        self._is_checking_ready = False
        await self.wait_for_ready()
        logger.info("Worker process started")

    async def stop(self):
        process, self._process = self._process, None
        if process:
            process.close()
        session, self._session = self._session, None
        if session:
            for resp in self._manually_close:
                resp.close()
            self._manually_close.clear()
            await session.close()
        logger.info("Worker process stopped")

    async def restart(self, extra_flags: Iterable[str] = []):
        logger.info("Restarting worker...")
        await self.stop()
        await self.start(extra_flags)

    async def wait_for_ready(self, timeout: float = 300):
        if self._is_ready:
            return

        async def test_connection(session: aiohttp.ClientSession):
            async with session.get(
                "/health", timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                resp.raise_for_status()

        start = time.time()
        while self._is_checking_ready and time.time() - start < timeout:
            await asyncio.sleep(0.1)

        if self._is_ready:
            return

        try:
            self._is_checking_ready = True

            while time.time() - start < timeout:
                if (
                    self._process is not None
                    and self._session is not None
                    and _port_in_use(self._port)
                ):
                    try:
                        if not self._is_ready:
                            await test_connection(self._session)
                            self._is_ready = True
                        return
                    except asyncio.TimeoutError:
                        logger.warn("Server not ready yet due to timeout")
                    except Exception as e:
                        logger.warn("Server not ready yet", exc_info=e)

                await asyncio.sleep(0.1)

            raise TimeoutError("Server did not start in time")
        finally:
            self._is_checking_ready = False

    async def proxy_request(self, request: Request, timeout: int | None = 300):
        if request.route is None:
            raise ValueError("Route not found")
        await self.wait_for_ready()
        assert self._session is not None
        async with self._session.request(
            request.method,
            f"/{request.route.path}",
            headers=request.headers,
            data=request.body,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            headers = resp.headers
            status = resp.status
            body = await resp.read()
            return HTTPResponse(
                body,
                status=status,
                headers=dict(headers),
                content_type=request.content_type,
            )

    async def get_sse(self, request: Request):
        await self.wait_for_ready()
        assert self._session is not None
        async with self._session.request(
            request.method,
            "/sse",
            headers=request.headers,
            data=request.body,
            timeout=aiohttp.ClientTimeout(total=60 * 60, connect=5),
        ) as resp:
            self._manually_close.add(resp)
            try:
                async for data, _ in resp.content.iter_chunks():
                    yield data
            finally:
                self._manually_close.remove(resp)

    async def get_packages(self):
        await self.wait_for_ready()
        assert self._session is not None
        logger.debug("Fetching packages...")
        async with self._session.get(
            "/packages", params={"hideInternal": "false"}
        ) as packages_resp:
            packages_json = await packages_resp.json()
            packages = [Package.from_dict(p) for p in packages_json]
            return packages
