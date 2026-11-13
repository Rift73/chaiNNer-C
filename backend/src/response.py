from __future__ import annotations

from typing import Literal, TypedDict

from sanic.exceptions import ServerError

from events import ExecutionErrorSource

# A worker's /run or /kill response has this header when the GPU's CUDA context
# is broken (a sticky error): nothing in it can be freed or run again, so the
# host restarts the worker (server.cuda_context_broken).
RESTART_WORKER_HEADER = "X-chaiNNer-Restart-Worker"

# What writing an event stream raises once its client has gone (the app's window closed, a
# reset connection), the quiet end of the stream: the socket's error, or Sanic's ServerError
# once it has ended the stream. A closing transport raises RequestCancelled, a
# CancelledError, which ends the handler by itself.
CLIENT_GONE = (ConnectionError, ServerError)


class SuccessResponse(TypedDict):
    type: Literal["success"]


class ErrorResponse(TypedDict):
    type: Literal["error"]
    message: str
    exception: str
    source: ExecutionErrorSource | None


class NoExecutorResponse(TypedDict):
    type: Literal["no-executor"]


class AlreadyRunningResponse(TypedDict):
    type: Literal["already-running"]
    message: str


def success_response() -> SuccessResponse:
    return {"type": "success"}


def error_response(
    message: str,
    exception: str | Exception,
    source: ExecutionErrorSource | None = None,
) -> ErrorResponse:
    # Imported here: the host imports this module at its start, and the executor's
    # module (with NumPy, about 100 ms) is the worker's.
    from process import NodeExecutionError

    if source is None and isinstance(exception, NodeExecutionError):
        source = {
            "nodeId": exception.node_id,
            "schemaId": exception.node_data.schema_id,
            "inputs": exception.inputs,
        }
    return {
        "type": "error",
        "message": message,
        "exception": str(exception),
        "source": source,
    }


def no_executor_response() -> NoExecutorResponse:
    return {"type": "no-executor"}


def already_running_response(message: str) -> AlreadyRunningResponse:
    return {"type": "already-running", "message": message}
