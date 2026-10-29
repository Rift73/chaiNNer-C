#!/bin/sh
# fetch.sh URL SHA256 OUTPUT: downloads URL to OUTPUT and fails, removing OUTPUT, unless its SHA-256 is SHA256.
set -eu
curl -fsSL --retry 3 -o "$3" "$1"
if ! echo "$2  $3" | sha256sum -c --quiet -; then
    rm -f "$3"
    echo "fetch.sh: unexpected SHA-256 for $1" >&2
    exit 1
fi
