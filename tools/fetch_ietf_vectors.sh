#!/usr/bin/env bash
# Fetch the IETF RFC 6716 / RFC 8251 Opus test vectors.
#
# Downloads the canonical archive from opus-codec.org, verifies its
# SHA-256, and extracts it to `tests/vectors/ietf/` at the repo root.
# Safe to re-run: if the target directory already contains the complete
# expected vector set, we skip the download. If SHA-256 verification fails we
# abort without touching the destination.
#
# Used by the `conformance` crate's `ietf_vectors` test. That test panics
# with a helpful message if this script hasn't been run.

set -euo pipefail

URL="https://opus-codec.org/static/testvectors/opus_testvectors-rfc8251.tar.gz"
EXPECTED_SIZE=74624664
EXPECTED_SHA256="6b26a22f9ba87b2b836906a9bb7afec5f8e54d49553b1200382520ee6fedfa55"
DOWNLOAD_CONNECT_TIMEOUT=30
DOWNLOAD_TIMEOUT=900

# Locate the repo root: the directory containing the top-level Cargo.toml
# with `[workspace]`. This script may be invoked from anywhere.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

if [[ ! -f "${repo_root}/Cargo.toml" ]]; then
    echo "ERROR: could not locate repo root (no Cargo.toml at ${repo_root})" >&2
    exit 1
fi

dest_dir="${repo_root}/tests/vectors/ietf"
tmp_dir=""
archive=""

cleanup() {
    if [[ -n "${tmp_dir}" ]]; then
        rm -rf "${tmp_dir}"
    fi
}
trap cleanup EXIT

vector_set_complete() {
    local dir="$1"
    local missing=0
    local n stem

    if [[ ! -d "${dir}" ]]; then
        return 1
    fi

    for n in 01 02 03 04 05 06 07 08 09 10 11 12; do
        stem="testvector${n}"
        if [[ ! -f "${dir}/${stem}.bit" ]]; then
            missing=$((missing + 1))
        fi
        if [[ ! -f "${dir}/${stem}.dec" && ! -f "${dir}/${stem}m.dec" ]]; then
            missing=$((missing + 1))
        fi
    done

    [[ "${missing}" -eq 0 ]]
}

report_missing_vectors() {
    local dir="$1"
    local n stem

    for n in 01 02 03 04 05 06 07 08 09 10 11 12; do
        stem="testvector${n}"
        if [[ ! -f "${dir}/${stem}.bit" ]]; then
            echo "WARNING: missing ${dir}/${stem}.bit" >&2
        fi
        if [[ ! -f "${dir}/${stem}.dec" && ! -f "${dir}/${stem}m.dec" ]]; then
            echo "WARNING: missing ${dir}/${stem}.dec or ${dir}/${stem}m.dec" >&2
        fi
    done
}

download_with_timeout() {
    if command -v timeout >/dev/null 2>&1; then
        timeout "${DOWNLOAD_TIMEOUT}s" "$@"
    else
        "$@"
    fi
}

# Already-extracted? Skip only when the complete set is present.
if vector_set_complete "${dest_dir}"; then
    echo "IETF vectors already present at ${dest_dir}; nothing to do."
    exit 0
fi

tmp_dir="$(mktemp -d -t ropus_ietf.XXXXXX)"
archive="${tmp_dir}/opus_testvectors-rfc8251.tar.gz"
staging_dir="${tmp_dir}/staging"

echo "Downloading IETF vectors from ${URL}..."
if command -v curl >/dev/null 2>&1; then
    download_with_timeout curl --connect-timeout "${DOWNLOAD_CONNECT_TIMEOUT}" --max-time "${DOWNLOAD_TIMEOUT}" -fL -o "${archive}" "${URL}"
elif command -v wget >/dev/null 2>&1; then
    download_with_timeout wget --timeout="${DOWNLOAD_CONNECT_TIMEOUT}" --tries=3 -O "${archive}" "${URL}"
else
    echo "ERROR: need curl or wget to download the vectors" >&2
    exit 1
fi

# Size check first — cheap, catches a truncated download before SHA.
actual_size="$(wc -c < "${archive}")"
if [[ "${actual_size}" != "${EXPECTED_SIZE}" ]]; then
    echo "ERROR: downloaded size ${actual_size} != expected ${EXPECTED_SIZE}" >&2
    echo "       archive: ${archive}" >&2
    exit 2
fi

echo "Verifying SHA-256..."
if command -v sha256sum >/dev/null 2>&1; then
    actual_sha="$(sha256sum "${archive}" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
    actual_sha="$(shasum -a 256 "${archive}" | awk '{print $1}')"
else
    echo "ERROR: need sha256sum or shasum to verify the archive" >&2
    exit 1
fi

if [[ "${actual_sha}" != "${EXPECTED_SHA256}" ]]; then
    echo "ERROR: SHA-256 mismatch" >&2
    echo "  expected: ${EXPECTED_SHA256}" >&2
    echo "  actual:   ${actual_sha}" >&2
    echo "       archive: ${archive}" >&2
    echo "Aborting without touching ${dest_dir}." >&2
    exit 3
fi

echo "Extracting to ${dest_dir}..."
# The tarball has one top-level directory `opus_newvectors/`. We
# strip that and land files directly under `ietf/` so
# `tests/vectors/ietf/testvectorNN.bit` resolves without an extra
# subdirectory indirection.
tar -xzf "${archive}" -C "${tmp_dir}"

# Move .bit / .dec files up, tolerating the upstream directory name.
extracted_dir=""
for candidate in "${tmp_dir}"/opus_newvectors "${tmp_dir}"/opus_testvectors; do
    if [[ -d "${candidate}" ]]; then
        extracted_dir="${candidate}"
        break
    fi
done
if [[ -z "${extracted_dir}" ]]; then
    echo "ERROR: tarball layout unexpected; no opus_newvectors/ or opus_testvectors/ subdir" >&2
    exit 4
fi

mkdir -p "${staging_dir}"
mv -f "${extracted_dir}"/* "${staging_dir}/"

# Sanity-check staging before publishing anything into the workspace.
if ! vector_set_complete "${staging_dir}"; then
    report_missing_vectors "${staging_dir}"
    echo "ERROR: expected vector files missing after extraction" >&2
    exit 5
fi

mkdir -p "${dest_dir}"
mv -f "${staging_dir}"/* "${dest_dir}/"

echo "IETF vectors installed in ${dest_dir}."
