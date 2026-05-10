#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== Fintech PII — Evaluation ==="

python3 -m src.eval.run_eval \
  --eval-config configs/eval_config.yaml \
  "$@"

echo "=== Done ==="
