#!/usr/bin/env bash
set -euo pipefail

FORMAT="${1:-svg}"
OUT_DIR="${2:-diagrams/architecture}"

if ! command -v dot >/dev/null 2>&1; then
  echo "Graphviz 'dot' not found."
  echo "Install it (macOS: brew install graphviz) then re-run."
  exit 1
fi

mkdir -p "${OUT_DIR}"

cmd_base=(dot -T"${FORMAT}")
if [[ "${FORMAT}" == "png" ]]; then
  cmd_base+=( -Gdpi=300 )
fi

"${cmd_base[@]}" docs/architecture/acagn_training_pipeline.dot -o "${OUT_DIR}/diagram1_system_overview.${FORMAT}"
"${cmd_base[@]}" docs/architecture/acagn_inference.dot -o "${OUT_DIR}/diagram2_acagn_gate_detail.${FORMAT}"

echo "Wrote:"
echo "  ${OUT_DIR}/diagram1_system_overview.${FORMAT}"
echo "  ${OUT_DIR}/diagram2_acagn_gate_detail.${FORMAT}"
