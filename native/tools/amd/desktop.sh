#!/usr/bin/env bash
# Optional: a remote desktop in the browser to use chaiNNer-C's window on the rented machine. A virtual display (Xvfb
# :1, 1920x1080), a light window manager (Openbox), a VNC server on it (x11vnc) and noVNC, its web client, all bound to
# 127.0.0.1 and without a password: only an SSH tunnel reaches them. On your computer:
#   ssh -p <ssh port> -L 6080:127.0.0.1:6080 root@<host>
# then open http://127.0.0.1:6080/vnc.html. chaiNNer starts there through its launcher (as root without Chromium's
# sandbox, which the launcher decides). `desktop.sh stop` ends it all.
# Usage: native/tools/amd/desktop.sh [start | stop]
set -uo pipefail
# shellcheck source=common.sh
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

pids=$WORK/desktop.pids
stop() {
    [ -f "$pids" ] || return 0
    while read -r pid; do
        kill "$pid" 2>/dev/null
    done <"$pids"
    rm -f -- "$pids"
    say "remote desktop stopped"
}

case "${1:-start}" in
    stop)
        stop
        exit 0
        ;;
    start) ;;
    *) die "usage: desktop.sh [start | stop]" ;;
esac
command -v apt-get >/dev/null 2>&1 || die "the remote desktop installs its packages with apt (Ubuntu, Debian)"
[ -x "$APP/chaiNNer.sh" ] || die "no app at $APP: run setup.sh first"
stop
export DEBIAN_FRONTEND=noninteractive
run apt-get install -y -qq --no-install-recommends xvfb openbox x11vnc novnc websockify xterm \
    >"$LOGS/desktop-packages.log" 2>&1 || die "installing the remote desktop failed ($LOGS/desktop-packages.log)"

start_background() {
    local name=$1
    shift
    if [ "$DRY_RUN" = 1 ]; then
        say "[dry run] $*"
        return 0
    fi
    "$@" >"$LOGS/desktop-$name.log" 2>&1 </dev/null &
    echo $! >>"$pids"
}
start_background xvfb Xvfb :1 -screen 0 1920x1080x24 -nolisten tcp
sleep 2
export DISPLAY=:1
start_background openbox openbox
start_background x11vnc x11vnc -display :1 -localhost -rfbport 5901 -forever -shared -nopw
sleep 2
start_background novnc websockify --web /usr/share/novnc 127.0.0.1:6080 127.0.0.1:5901
start_background chainner "$APP/chaiNNer.sh"
say "remote desktop on 127.0.0.1:6080. On your computer: ssh -p <ssh port> -L 6080:127.0.0.1:6080 root@<host>, then open http://127.0.0.1:6080/vnc.html"
