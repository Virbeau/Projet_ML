#!/bin/bash
set -euo pipefail

LOCAL_PROJECT_DIR="/users/local/l24virbe/Projet_ML"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-/users/local/l24virbe/Projet_ML}"
HOSTS_FILE="${HOSTS_FILE:-${LOCAL_PROJECT_DIR}/hosts_v3.txt}"
SSH_OPTS=(
  -n
  -o BatchMode=yes
  -o ConnectTimeout=3
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
)
RSYNC_SSH="ssh -o BatchMode=yes -o ConnectTimeout=3 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

if [[ ! -f "${HOSTS_FILE}" ]]; then
  echo "Missing hosts file: ${HOSTS_FILE}"
  exit 1
fi

SYNC_EXCLUDES=(
  "--exclude=.git/"
  "--exclude=env_projet/"
  "--exclude=__pycache__/"
  "--exclude=analysis_visuals_1k/"
  "--exclude=analysis_visuals_50k/"
  "--exclude=budget_analysis/"
  "--exclude=distributed_runs/"
  "--exclude=*.json"
  "--exclude=*.log"
)

while read -r host; do
  [[ -z "${host}" ]] && continue
  echo "=== ${host} ==="

  if ! ssh "${SSH_OPTS[@]}" "${host}" "echo ok" >/dev/null 2>&1; then
    echo "skip: unreachable"
    continue
  fi

  ssh "${SSH_OPTS[@]}" "${host}" "mkdir -p '${REMOTE_PROJECT_DIR}'"

  rsync -az --delete -e "${RSYNC_SSH}" "${SYNC_EXCLUDES[@]}" \
    "${LOCAL_PROJECT_DIR}/" "${host}:${REMOTE_PROJECT_DIR}/"

  ssh "${SSH_OPTS[@]}" "${host}" "bash -lc '
    set -e
    cd "${REMOTE_PROJECT_DIR}"
    if [ ! -x env_projet/bin/python3 ]; then
      python3 -m venv env_projet
    fi
    env_projet/bin/python3 -m pip install --upgrade pip >/dev/null
    env_projet/bin/python3 -m pip install -r requirements_generation.txt >/dev/null
    echo ready
  '"

done < "${HOSTS_FILE}"

echo "Remote setup completed."
