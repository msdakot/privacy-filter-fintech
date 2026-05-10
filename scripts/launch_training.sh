#!/usr/bin/env bash
# NOTE: Training runs in Google Colab, not locally.
# This script is a reference for what the Colab notebook executes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== Fintech PII — Training (reference) ==="
echo "Open notebooks/02_train_colab.ipynb in Google Colab to run training."
echo "This script documents the opf train command used:"
echo ""
echo "  opf train \\"
echo "    --model OpenMed/privacy-filter-nemotron \\"
echo "    --dataset msdakot/fintech-privacy-pii \\"
echo "    --output msdakot/fintech-privacy-filter-v0 \\"
echo "    --config configs/train_config.yaml"
