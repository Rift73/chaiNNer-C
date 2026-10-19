import importlib
import io
import sys

from server_config import ServerConfig
from worker_spawn import find_free_port, spawn_worker

# The app reads this process's output as UTF-8, but piped output defaults to the ANSI
# code page, and logging drops a whole line that has a character outside it (e.g. a
# Japanese file name).
for stream in (sys.stdout, sys.stderr):
    if isinstance(stream, io.TextIOWrapper):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")

# The worker starts first: it loads its nodes while this process checks the server
# dependencies and imports its own modules.
worker_port = find_free_port()
worker = spawn_worker([str(worker_port), *ServerConfig.parse_argv().worker_flags()])

# Install server dependencies. Can't start the server without them, but we don't want to install the other deps yet.
install_server_deps = importlib.import_module("dependencies.install_server_deps")
if install_server_deps.installed > 0:
    # A first start: the worker started without them. The host starts another.
    worker.kill()

# Start the host server
server_host = importlib.import_module("server_host")
server_host.main(None if install_server_deps.installed > 0 else (worker_port, worker))
