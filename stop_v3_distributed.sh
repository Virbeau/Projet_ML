#!/bin/bash
set -uo pipefail

PROJECT_DIR="/users/local/l24virbe/Projet_ML"
HOSTS_FILE="${PROJECT_DIR}/hosts_v3.txt"
RUN_ID="${1:-v3_60000}"
REMOTE_LOG_DIR="${PROJECT_DIR}/distributed_runs/${RUN_ID}/logs"

mapfile -t HOSTS < "${HOSTS_FILE}"

for host in "${HOSTS[@]}"; do
  echo "Stopping on ${host}"
  if ! ssh -n -o BatchMode=yes "${host}" "
    if ls '${REMOTE_LOG_DIR}'/pid_*.txt >/dev/null 2>&1; then
      for p in '${REMOTE_LOG_DIR}'/pid_*.txt; do
        pid=\$(cat \"\$p\")
        kill -TERM \"\$pid\" >/dev/null 2>&1 || true
      done
      sleep 1
      for p in '${REMOTE_LOG_DIR}'/pid_*.txt; do
        pid=\$(cat \"\$p\")
        if ps -p \"\$pid\" >/dev/null 2>&1; then
          kill -KILL \"\$pid\" >/dev/null 2>&1 || true
        fi
      done
      echo 'stopped'
    else
      echo 'nothing to stop'
    fi
  "; then
    echo "unreachable or permission denied"
  fi
done
