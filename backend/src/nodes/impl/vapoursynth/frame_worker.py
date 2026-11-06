"""The process that evaluates one chain run's image scripts one after another (runner.py
starts it and kills it when the run ends or stops): VapourSynth, its plugins and the
vs-jetpack modules a script imports load once, where a vspipe per image loads them each
time. A plugin's crash costs this process and the script it was evaluating.

Run as a script, with the Python that has VapourSynth. Each line on stdin is a JSON
object with a script: {"source": its text, "script": the path its tracebacks name}. Each
script gets its own namespace, as in vspipe, and its clips go when it ends. Each answer is
a JSON line on stdout: {"error": null}, or {"error": the traceback}."""

from __future__ import annotations

import gc
import json
import os
import sys
import traceback
from typing import Any


def evaluate(source: str, script: str, vs: Any) -> str | None:
    """Runs the script; the traceback of what it raised, or None."""
    search_path = list(sys.path)
    scope = {"__name__": "__vapoursynth__", "__file__": script}
    try:
        exec(compile(source, script, "exec"), scope)
        return None
    except BaseException:
        return traceback.format_exc()
    finally:
        vs.clear_outputs()
        scope.clear()
        sys.path[:] = search_path
        gc.collect()


def main() -> None:
    import vapoursynth as vs

    # The answers keep the real stdout; what a plugin prints goes to stderr.
    answers = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    for line in sys.stdin.buffer:
        request = json.loads(line)
        error = evaluate(request["source"], request["script"], vs)
        answers.write(json.dumps({"error": error}) + "\n")
        answers.flush()


if __name__ == "__main__":
    main()
