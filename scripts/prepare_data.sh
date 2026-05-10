#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== Fintech PII — Data Preparation ==="

python3 -m src.data.build_dataset \
  --data-config configs/data_config.yaml \
  "$@"

echo "=== Done ==="
