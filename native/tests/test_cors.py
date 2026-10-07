"""nodes.impl.cors sends sanic-cors 2.2.0's default CORS(app) headers.

The expected headers are sanic-cors's (core.get_cors_headers with DEFAULT_OPTIONS):
the wildcard origin without an Origin, the origin echoed with Vary: Origin with
one, and on a preflight the requested headers sorted, the literal "None" max age
and the sorted default methods. Each request goes over HTTP to a Sanic 25 server
run in this process.
"""

from __future__ import annotations

import asyncio
import socket

import aiohttp
import pytest
from multidict import CIMultiDict
from sanic import Request, Sanic
from sanic.response import json

from nodes.impl.cors import METHODS, add_cors

ORIGIN = "http://localhost:3000"

Response = tuple[int, CIMultiDict[str], bytes]


def free_port() -> int:
    # Sanic reads port 0 as its default port, 8000, which a backend may hold.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def exchange(
    requests: list[tuple[str, str, dict[str, str]]],
) -> list[Response]:
    app = Sanic("cors_test")
    add_cors(app)

    @app.get("/data")
    async def data(_request: Request):
        return json({"ok": True})

    @app.get("/vary")
    async def vary(_request: Request):
        return json({}, headers={"Vary": "Accept-Encoding"})

    port = free_port()
    server = await app.create_server(
        host="127.0.0.1", port=port, return_asyncio_server=True
    )
    assert server is not None
    await server.startup()
    await server.before_start()
    await server.after_start()
    results: list[Response] = []
    try:
        async with aiohttp.ClientSession(f"http://127.0.0.1:{port}") as session:
            for method, path, headers in requests:
                async with session.request(method, path, headers=headers) as r:
                    results.append((r.status, r.headers.copy(), await r.read()))
    finally:
        await server.before_stop()
        closing = server.close()
        assert closing is not None
        await closing
        await server.after_stop()
    return results


PREFLIGHT = {
    "Origin": ORIGIN,
    "Access-Control-Request-Method": "post",
    "Access-Control-Request-Headers": "content-type, X-Custom",
}
REQUESTS = [
    ("GET", "/data", {}),
    ("GET", "/data", {"Origin": ORIGIN}),
    ("OPTIONS", "/data", PREFLIGHT),
    ("OPTIONS", "/data", {"Origin": ORIGIN, "Access-Control-Request-Method": "TRACE"}),
    ("GET", "/missing", {"Origin": ORIGIN}),
    ("GET", "/vary", {"Origin": ORIGIN}),
]
CORS = (
    "Access-Control-Allow-Origin",
    "Access-Control-Allow-Headers",
    "Access-Control-Max-Age",
    "Access-Control-Allow-Methods",
    "Access-Control-Expose-Headers",
    "Access-Control-Allow-Credentials",
    "Vary",
)


@pytest.fixture(scope="module")
def responses() -> list[Response]:
    return asyncio.run(exchange(REQUESTS))


def cors(headers: CIMultiDict[str]) -> dict[str, list[str]]:
    return {name: headers.getall(name) for name in CORS if name in headers}


def test_get_without_origin_allows_any_origin(responses: list[Response]):
    status, headers, body = responses[0]
    assert status == 200
    assert body == b'{"ok":true}'
    assert cors(headers) == {"Access-Control-Allow-Origin": ["*"]}


def test_get_with_origin_echoes_it(responses: list[Response]):
    status, headers, _ = responses[1]
    assert status == 200
    assert cors(headers) == {
        "Access-Control-Allow-Origin": [ORIGIN],
        "Vary": ["Origin"],
    }


def test_preflight(responses: list[Response]):
    # /data has no OPTIONS route: the preflight answers through the 405 path.
    status, headers, body = responses[2]
    assert status == 200
    assert body == b""
    assert cors(headers) == {
        "Access-Control-Allow-Origin": [ORIGIN],
        "Access-Control-Allow-Headers": ["X-Custom, content-type"],
        "Access-Control-Max-Age": ["None"],
        "Access-Control-Allow-Methods": [METHODS],
        "Vary": ["Origin"],
    }
    assert METHODS == "DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT"


def test_preflight_of_another_method_sends_only_the_origin(responses: list[Response]):
    status, headers, body = responses[3]
    assert status == 200
    assert body == b""
    assert cors(headers) == {
        "Access-Control-Allow-Origin": [ORIGIN],
        "Vary": ["Origin"],
    }


def test_error_responses_carry_the_headers(responses: list[Response]):
    status, headers, _ = responses[4]
    assert status == 404
    assert cors(headers) == {
        "Access-Control-Allow-Origin": [ORIGIN],
        "Vary": ["Origin"],
    }


def test_an_existing_vary_is_extended(responses: list[Response]):
    status, headers, _ = responses[5]
    assert status == 200
    assert cors(headers) == {
        "Access-Control-Allow-Origin": [ORIGIN],
        "Vary": ["Accept-Encoding, Origin"],
    }
