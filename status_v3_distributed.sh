#!/bin/bash
set -uo pipefail

PROJECT_DIR="/users/local/l24virbe/Projet_ML"
HOSTS_FILE="${PROJECT_DIR}/hosts_v3.txt"
RUN_ID="${1:-v3_60000}"
REMOTE_LOG_DIR="${PROJECT_DIR}/distributed_runs/${RUN_ID}/logs"

mapfile -t HOSTS < "${HOSTS_FILE}"

echo "Status for run: ${RUN_ID}"
for host in "${HOSTS[@]}"; do
  echo "--- ${host} ---"
  if ! ssh -n -o BatchMode=yes "${host}" "
    if ls '${REMOTE_LOG_DIR}'/pid_*.txt >/dev/null 2>&1; then
      for p in '${REMOTE_LOG_DIR}'/pid_*.txt; do
        pid=\$(cat \"\$p\")
        if ps -p \"\$pid\" >/dev/null 2>&1; then
          echo \"running pid=\$pid file=\$p\"
        else
          echo \"stopped pid=\$pid file=\$p\"
        fi
      done
      echo 'recent logs:'
      ls -1t '${REMOTE_LOG_DIR}'/progress_*.log 2>/dev/null | head -n 2 | xargs -r -I{} sh -c 'echo "  {}"; tail -n 3 "{}"'
    else
      echo 'no pid files found'
    fi
  "; then
    echo "unreachable or permission denied"
  fi
done
