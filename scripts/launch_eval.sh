#!/usr/bin/env bash
set -euo pipefail

ROUTE="${1:-beta/standard.yaml}"
CKPT="${2:-runs/standard/final.ckpt}"

python -m cairn.routes.evaluate "${ROUTE}" "${CKPT}" auto
