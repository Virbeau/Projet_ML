#!/bin/bash
set -euo pipefail

LOCAL_PROJECT_DIR="/users/local/l24virbe/Projet_ML"
RUN_ID="${1:-v3_10000}"
LOCAL_OUT_DIR="${LOCAL_PROJECT_DIR}/distributed_runs/${RUN_ID}"
FINAL_DATASET="${LOCAL_PROJECT_DIR}/dataset_${RUN_ID}.json"

if [[ ! -d "${LOCAL_OUT_DIR}" ]]; then
  echo "Output directory not found: ${LOCAL_OUT_DIR}"
  exit 1
fi

# Count parts
PART_COUNT=$(find "${LOCAL_OUT_DIR}" -maxdepth 1 -name "part_*.json" | wc -l)
if [[ "${PART_COUNT}" -eq 0 ]]; then
  echo "No part files found in ${LOCAL_OUT_DIR}"
  exit 1
fi

echo "Merging ${PART_COUNT} part files..."
source "${LOCAL_PROJECT_DIR}/env_projet/bin/activate"
python3 "${LOCAL_PROJECT_DIR}/merge_v3_parts.py" \
  --parts-dir "${LOCAL_OUT_DIR}" \
  --output "${FINAL_DATASET}"

# Validate
if [ -f "${FINAL_DATASET}" ]; then
  SIZE=$(ls -lh "${FINAL_DATASET}" | awk '{print $5}')
  INSTANCE_COUNT=$(python3 -c "import json; d=json.load(open('${FINAL_DATASET}')); print(len(d.get('instances',[])))")
  echo ""
  echo "✅ Merge complete"
  echo "  File: ${FINAL_DATASET}"
  echo "  Size: ${SIZE}"
  echo "  Instances: ${INSTANCE_COUNT}"
else
  echo "❌ Merge failed"
  exit 1
fi
