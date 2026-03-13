#!/bin/bash
set -euo pipefail

LOCAL_PROJECT_DIR="/users/local/l24virbe/Projet_ML"
RUN_ID="${1:-v3_10000}"

echo "=== v3 Merge Pipeline ==="
echo ""

# Step 1: Wait for completion
echo "📊 STEP 1: Monitoring generation completion..."
while true; do
  RUNNING=$(ssh -n -o BatchMode=yes -o ConnectTimeout=3 fl-tp-br-632 \
    "ps aux | grep -v grep | grep 'main_production.py' | wc -l" 2>/dev/null || echo 1)
  
  if [ "${RUNNING}" -eq 0 ]; then
    echo "✅ Generation done!"
    break
  fi
  
  SAMPLE_LOG=$(ssh -n -o BatchMode=yes -o ConnectTimeout=3 fl-tp-br-632 \
    "tail -3 /users/local/l24virbe/Projet_ML/distributed_runs/${RUN_ID}/logs/progress_0_*.log 2>/dev/null | tail -1" 2>/dev/null || echo "...")
  
  echo "  Still running... ${SAMPLE_LOG}"
  sleep 30
done

echo ""
echo "🔄 STEP 2: Retrieving part files from all hosts..."
"${LOCAL_PROJECT_DIR}/retrieve_v3_parts.sh" "${RUN_ID}"

echo ""
echo "🔗 STEP 3: Finalizing merge..."
"${LOCAL_PROJECT_DIR}/finalize_v3_merge.sh" "${RUN_ID}"

echo ""
echo "✅ Merge pipeline complete!"
echo ""
echo "Next steps:"
echo "  1. Run analysis: python3 analyze_dataset_${RUN_ID}.py"
echo "  2. Implement train/val/test split"
echo "  3. Train ML models"
