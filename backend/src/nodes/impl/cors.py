"""The CORS headers sanic-cors 2.2.0 sent for a default ``CORS(app)``, on Sanic 25.

sanic-cors is abandoned: it supports no Sanic from 24.12 on. Its defaults let every
path answer any origin, every method and any request header, without credentials.
This emits the same headers on the same responses: the response middleware adds
them to every response, error responses included, and every OPTIONS request gets
an empty preflight response. Sanic routes before the request middleware runs, so
an OPTIONS request to a route without OPTIONS reaches the preflight through the
405 error path, as it reached sanic-cors's.
"""

from __future__ import annotations

from sanic import HTTPResponse, Request, Sanic
from sanic.response import BaseHTTPResponse

# sanic-cors's default methods, sorted and joined; it tests a preflight's requested
# method as a substring of this text, so this does too.
METHODS = "DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT"


def cors_headers(request: Request) -> list[tuple[str, str]]:
    """The headers sanic-cors's defaults set for request, in its order."""
    origins = request.headers.getall("Origin", None)
    origin = ", ".join(origins) if origins else ""
    # Without an Origin, the wildcard; with one, that origin echoed.
    allow_origin = origin or "*"
    headers = [("Access-Control-Allow-Origin", allow_origin)]
    if request.method == "OPTIONS":
        method = request.headers.get("Access-Control-Request-Method", "").upper()
        if method and method in METHODS:
            requested = request.headers.getall("Access-Control-Request-Headers", None)
            joined = ", ".join(requested) if requested else ""
            if joined:
                names = sorted(name.strip() for name in joined.split(","))
                allowed = ", ".join(names)
                if allowed:
                    headers.append(("Access-Control-Allow-Headers", allowed))
            # The default max_age is None, which sanic-cors sends as its text.
            headers.append(("Access-Control-Max-Age", "None"))
            headers.append(("Access-Control-Allow-Methods", METHODS))
    # The allowed origin varies with the request unless it is the wildcard.
    if allow_origin != "*":
        headers.append(("Vary", "Origin"))
    return headers


def _apply(request: Request, response: BaseHTTPResponse) -> None:
    # Added, not set, as sanic-cors did; an existing Vary is extended instead.
    for name, value in cors_headers(request):
        if name == "Vary" and "vary" in response.headers:
            values = response.headers.popall("vary")
            response.headers.add("Vary", ", ".join([*values, value]))
        else:
            response.headers.add(name, value)
    request.ctx.cors_applied = True


async def _preflight(request: Request) -> HTTPResponse | None:
    if request.method != "OPTIONS":
        return None
    response = HTTPResponse()
    _apply(request, response)
    return response


async def _response(request: Request, response: BaseHTTPResponse | None) -> None:
    # None for websockets; the preflight already carries its headers.
    if response is None or getattr(request.ctx, "cors_applied", False):
        return
    _apply(request, response)


def add_cors(app: Sanic) -> None:
    """Gives every response of app sanic-cors 2.2.0's default CORS(app) headers."""
    # sanic-cors's middleware priorities.
    app.on_request(_preflight, priority=99)
    app.on_response(_response, priority=999)
