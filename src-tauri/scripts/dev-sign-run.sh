#!/bin/sh
# Cargo's runner for this Mac (see .cargo/config.toml): signs the app it is
# about to run with a real signing identity and a FIXED identifier, then runs
# it.
#
# Why: a debug build is signed ad hoc, and macOS ties Screen Recording and
# Input Monitoring to that signature — so every rebuild looked like a new app,
# and a permission granted to the last build silently stopped applying (the
# switch in System Settings still showed "on"). Signed with a certificate and
# a fixed identifier, the permission follows the app across rebuilds.
#
# Only the app itself is signed; test binaries and anything else run as they
# are. With no signing identity on the Mac, it runs unsigned, as before.
bin="$1"
case "$(basename "$bin")" in
  airoplane-flight-school)
    id="${FS_SIGN_IDENTITY:-$(security find-identity -v -p codesigning 2>/dev/null | sed -n 's/^ *1) \([0-9A-F]*\) .*/\1/p')}"
    if [ -n "$id" ]; then
      codesign --force --sign "$id" --identifier mode.airoplane.flightschool.dev "$bin" 2>/dev/null \
        || echo "[dev] could not sign the app; screen permissions will not survive a rebuild" >&2
    fi
    ;;
esac
exec "$@"
