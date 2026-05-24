#!/usr/bin/env bash
# Verify the byte-frozen slot1 submission file by SHA-256.
# Run from this directory: ./verify.sh
set -euo pipefail
cd "$(dirname "$0")"
sha256sum -c SHA256SUMS
