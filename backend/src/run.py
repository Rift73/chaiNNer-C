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

# The server dependencies: the host can't start without them (the other packages'
# come later). Installed before the worker starts, as pip can't replace a module a
# running worker has loaded (NumPy's, psutil's .pyd); none is missing but on a first
# start, which the installed packages' metadata tells without pip.
install_server_deps = importlib.import_module("dependencies.install_server_deps")
if install_server_deps.missing:
    install_server_deps.install()

# The worker starts next: it loads its nodes while this process imports its own modules.
worker_port = find_free_port()
worker = spawn_worker([str(worker_port), *ServerConfig.parse_argv().worker_flags()])

# Start the host server
server_host = importlib.import_module("server_host")
server_host.main((worker_port, worker))
