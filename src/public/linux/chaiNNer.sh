#!/bin/sh
# Starts chaiNNer-C from the folder it was unzipped to. `./chaiNNer.sh` opens the editor and
# `./chaiNNer.sh run chain.chn` runs a chain without a window; the script becomes the program, so a
# shell waits for it and gets its exit code (chaiNNer-cli.cmd does this on Windows).
#
# Chromium isolates its renderer processes in a sandbox. A zip cannot give chrome-sandbox the
# setuid bit that the old kind of sandbox needs, so the sandbox runs on unprivileged user
# namespaces, which some systems do not allow: Ubuntu 23.10 and later restrict them with AppArmor,
# Debian 10 and hardened kernels turn them off, and Docker blocks them. Chromium also exits at
# once when it runs as root with its sandbox. There the program starts without the sandbox, which
# this script says once. CHAINNER_SANDBOX=1 keeps the sandbox whatever the test finds (e.g. with
# an AppArmor profile for this folder's chainner), CHAINNER_SANDBOX=0 starts without it, and so
# does --no-sandbox among the arguments.
#
# Electron cannot start without a display, not even to run a chain. On a server without one,
# `run` uses a virtual display from xvfb-run (with xauth).
#
# Neither Electron nor the Python wheels (OpenCV, VapourSynth) carry every system library they
# link against, and a minimal install lacks some. The table resources/src/linux_libraries.txt
# names them and the package of each distribution family that provides them; a missing one ends
# the launch with the command that installs it. CHAINNER_SKIP_LIBRARY_CHECK=1 skips the check, for
# a system whose dynamic loader finds libraries that neither ldconfig nor LD_LIBRARY_PATH show.

here=$(dirname "$(readlink -f "$0")")
tab=$(printf '\t')

# Whether the dynamic loader finds the library $1, or the program $1 is on the PATH. A library is
# in ldconfig's cache (64-bit only), in a directory of LD_LIBRARY_PATH, or, where there is no
# ldconfig, in one of the usual directories.
have_file() {
    case $1 in
        *.so*)
            case $library_cache in
                *"$tab$1 (libc6,x86-64"*) return 0 ;;
            esac
            (
                IFS=:
                for directory in $library_dirs; do
                    [ -e "$directory/$1" ] && exit 0
                done
                exit 1
            )
            ;;
        *) command -v "$1" >/dev/null 2>&1 ;;
    esac
}

# The package manager of this distribution's family: from ID and ID_LIKE in os-release.
package_manager() {
    release=/etc/os-release
    [ -r "$release" ] || release=/usr/lib/os-release
    [ -r "$release" ] || return 0
    for word in $(. "$release"; echo "${ID:-} ${ID_LIKE:-}"); do
        case $word in
            debian | ubuntu) echo apt; return ;;
            rhel | fedora | centos | rocky | almalinux) echo dnf; return ;;
            arch) echo pacman; return ;;
            suse | sles) echo zypper; return ;;
        esac
    done
}

# Ends the launch with the install command when a file of the table is missing: the files of
# the groups base and electron, and of display when $1 is display.
check_system_files() {
    table="$here/resources/src/linux_libraries.txt"
    if [ ! -r "$table" ] || [ -n "${CHAINNER_SKIP_LIBRARY_CHECK:-}" ]; then
        return 0
    fi
    library_cache=$(ldconfig -p 2>/dev/null || /sbin/ldconfig -p 2>/dev/null || /usr/sbin/ldconfig -p 2>/dev/null)
    library_dirs=${LD_LIBRARY_PATH:-}
    if [ -z "$library_cache" ]; then
        library_dirs=$library_dirs:/lib:/lib64:/usr/lib:/usr/lib64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/usr/local/lib
    fi
    manager=$(package_manager)
    # Debian 13 and Ubuntu 24.04 (glibc 2.38 and newer) renamed some packages with a t64 suffix
    suffix=
    glibc=$(getconf GNU_LIBC_VERSION 2>/dev/null)
    glibc=${glibc#glibc }
    minor=${glibc#2.}
    if [ "$minor" != "$glibc" ] && [ "${minor%%.*}" -ge 38 ] 2>/dev/null; then
        suffix=t64
    fi

    missing=
    packages=
    while read -r file groups apt dnf pacman zypper; do
        case $file in '' | '#'*) continue ;; esac
        case ",$groups," in
            *,base,* | *,electron,*) ;;
            *,display,*) [ "$1" = display ] || continue ;;
            *) continue ;;
        esac
        have_file "$file" && continue
        missing="$missing $file"
        case $manager in
            apt) package=$apt ;;
            dnf) package=$dnf ;;
            pacman) package=$pacman ;;
            zypper) package=$zypper ;;
            *) package= ;;
        esac
        case $package in
            *'(t64)') package=${package%'(t64)'}$suffix ;;
        esac
        case " $packages " in
            *" $package "*) ;;
            *) packages="$packages $package" ;;
        esac
    done <"$table"
    [ -n "$missing" ] || return 0

    echo "chaiNNer-C cannot start: this system lacks$missing." >&2
    case $manager in
        apt) install="apt-get install" ;;
        dnf) install="dnf install" ;;
        pacman) install="pacman -S --needed" ;;
        zypper) install="zypper install" ;;
    esac
    if [ -n "$manager" ]; then
        if [ "$(id -u 2>/dev/null)" != 0 ]; then
            install="sudo $install"
        fi
        echo "Install them with:" >&2
        echo >&2
        echo "  $install$packages" >&2
    else
        echo "Install the packages of this distribution that provide them." >&2
    fi
    exit 1
}

# Whether Chromium's sandbox can start here: a setuid chrome-sandbox owned by root (what an install
# from a package has), or a user namespace in which the program has its capabilities. The namespace
# is tried, not just the settings that govern it: AppArmor and seccomp leave it created but empty.
# As root it cannot start at all: Chromium exits with "Running as root without --no-sandbox".
sandbox_works() {
    [ "$(id -u 2>/dev/null)" = 0 ] && return 1
    if [ -u "$here/chrome-sandbox" ] && [ "$(stat -c %u "$here/chrome-sandbox")" = 0 ]; then
        return 0
    fi
    [ "$(cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns 2>/dev/null)" = 1 ] && return 1
    [ "$(cat /proc/sys/kernel/unprivileged_userns_clone 2>/dev/null)" = 0 ] && return 1
    [ "$(cat /proc/sys/user/max_user_namespaces 2>/dev/null)" = 0 ] && return 1
    if command -v unshare >/dev/null 2>&1; then
        unshare --user --map-root-user --net --pid --fork true >/dev/null 2>&1 || return 1
    fi
    return 0
}

command_name=
for argument in "$@"; do
    case $argument in
        -*) ;;
        *)
            command_name=$argument
            break
            ;;
    esac
done

display_needed=
if [ "$command_name" = run ] && [ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
    display_needed=display
fi

check_system_files "$display_needed"

case "${CHAINNER_SANDBOX:-}" in
    0)
        set -- --no-sandbox "$@"
        ;;
    1) ;;
    *)
        case " $* " in
            *" --no-sandbox "*) ;;
            *)
                if ! sandbox_works; then
                    echo "chaiNNer-C: starting without the Chromium sandbox, because this system does not allow it here (CHAINNER_SANDBOX=1 forces it)." >&2
                    set -- --no-sandbox "$@"
                fi
                ;;
        esac
        ;;
esac

if [ -n "$display_needed" ]; then
    exec xvfb-run -a "$here/chainner" "$@"
fi

exec "$here/chainner" "$@"
