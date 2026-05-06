#!/usr/bin/env bash

die() {
    echo "ERROR: $*" >&2
    exit 1
}

discover_fuzz_targets() {
    local manifest="$1"

    [[ -r "$manifest" ]] || {
        echo "cannot read fuzz manifest: $manifest" >&2
        return 2
    }

    awk '
        function fail(message) {
            print message > "/dev/stderr"
            failed = 1
            exit 2
        }

        /^[[:space:]]*\[\[bin\]\][[:space:]]*(#.*)?$/ {
            if (in_bin && !seen_name) {
                fail("missing name in [[bin]] entry")
            }
            in_bin = 1
            seen_name = 0
            next
        }

        /^[[:space:]]*\[\[bin\]\]/ {
            fail("malformed [[bin]] header in " manifest ": " $0)
        }

        /^\[/ {
            if (in_bin && !seen_name) {
                fail("missing name in [[bin]] entry")
            }
            in_bin = 0
            seen_name = 0
            next
        }

        in_bin && /^[[:space:]]*name[[:space:]]*=/ {
            line = $0
            sub(/^[^=]*=[[:space:]]*/, "", line)
            sub(/[[:space:]]*(#.*)?$/, "", line)
            if (line !~ /^"[^"]+"$/) {
                fail("malformed fuzz target name in " manifest ": " $0)
            }
            sub(/^"/, "", line)
            sub(/"$/, "", line)
            targets[++target_count] = line
            seen_name = 1
        }

        END {
            if (failed) {
                exit 2
            }
            if (in_bin && !seen_name) {
                print "missing name in [[bin]] entry" > "/dev/stderr"
                exit 2
            }
            for (i = 1; i <= target_count; i++) {
                print targets[i]
            }
        }
    ' manifest="$manifest" "$manifest"
}
