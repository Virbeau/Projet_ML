#!/bin/bash
set -euo pipefail

LOCAL_PROJECT_DIR="/users/local/l24virbe/Projet_ML"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-/users/local/l24virbe/Projet_ML}"
HOSTS_FILE="${HOSTS_FILE:-${LOCAL_PROJECT_DIR}/hosts_v3.txt}"
RUN_ID="${1:-v3_10000}"
SSH_OPTS=(
  -n
  -o BatchMode=yes
  -o ConnectTimeout=3
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
)
RSYNC_SSH="ssh -o BatchMode=yes -o ConnectTimeout=3 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

REMOTE_OUT_DIR="${REMOTE_PROJECT_DIR}/distributed_runs/${RUN_ID}"
LOCAL_OUT_DIR="${LOCAL_PROJECT_DIR}/distributed_runs/${RUN_ID}"

mkdir -p "${LOCAL_OUT_DIR}"

echo "Retrieving part_*.json from all hosts..."

while read -r host; do
  [[ -z "${host}" ]] && continue

  if ! ssh "${SSH_OPTS[@]}" "${host}" "test -d '${REMOTE_OUT_DIR}'" >/dev/null 2>&1; then
    echo "[${host}] skip (unreachable or no run dir)"
    continue
  fi

  echo "[${host}] syncing parts..."
  rsync -az -e "${RSYNC_SSH}" --include="part_*.json" --exclude="*" \
    "${host}:${REMOTE_OUT_DIR}/" \
    "${LOCAL_OUT_DIR}/" || true

done < "${HOSTS_FILE}"

# Count retrieved files
COUNT=$(find "${LOCAL_OUT_DIR}" -maxdepth 1 -name "part_*.json" | wc -l)
echo ""
echo "Retrieved ${COUNT} part files"

# List them
if [ "${COUNT}" -gt 0 ]; then
  echo "  Files:"
  find "${LOCAL_OUT_DIR}" -maxdepth 1 -name "part_*.json" -exec ls -lh {} \; | awk '{print "    " $9 " (" $5 ")"}'
fi
