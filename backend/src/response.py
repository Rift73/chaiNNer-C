from __future__ import annotations

from typing import Literal, TypedDict

from events import ExecutionErrorSource
from process import NodeExecutionError

# A worker's /run or /kill response has this header when the GPU's CUDA context
# is broken (a sticky error): nothing in it can be freed or run again, so the
# host restarts the worker (server.cuda_context_broken).
RESTART_WORKER_HEADER = "X-chaiNNer-Restart-Worker"


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
