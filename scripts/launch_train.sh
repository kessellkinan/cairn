#!/usr/bin/env bash
set -euo pipefail

ROUTE="${1:-beta/standard.yaml}"
WORKDIR="${2:-runs/standard}"

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m cairn.routes.train "${ROUTE}" auto "${WORKDIR}"
