import importlib
import io
import sys

# The app reads this process's output as UTF-8, but piped output defaults to the ANSI
# code page, and logging drops a whole line that has a character outside it (e.g. a
# Japanese file name).
for stream in (sys.stdout, sys.stderr):
    if isinstance(stream, io.TextIOWrapper):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")

# Install server dependencies. Can't start the server without them, but we don't want to install the other deps yet.
importlib.import_module("dependencies.install_server_deps")

# Start the host server
server_host = importlib.import_module("server_host")
server_host.main()
