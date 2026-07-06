#!/usr/bin/env bash
set -euo pipefail

# Cohort handles and access terms live in terrain-sources.txt. Each registry is
# downloaded under its own data-use agreement into the directory passed as $1.
DEST="${1:-data}"
mkdir -p "${DEST}"

echo "Open-layer cohorts to stage under ${DEST}:"
grep -E '^\[' terrain-sources.txt
echo
echo "See terrain-sources.txt for portal URLs, scope, and license per cohort."
