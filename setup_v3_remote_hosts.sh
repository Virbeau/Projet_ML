#!/bin/bash
set -euo pipefail

LOCAL_PROJECT_DIR="/users/local/l24virbe/Projet_ML"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-/users/local/l24virbe/Projet_ML}"
HOSTS_FILE="${LOCAL_PROJECT_DIR}/hosts_v3.txt"

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

  if ! ssh -n -o BatchMode=yes -o ConnectTimeout=3 "${host}" "echo ok" >/dev/null 2>&1; then
    echo "skip: unreachable"
    continue
  fi

  ssh -n -o BatchMode=yes "${host}" "mkdir -p '${REMOTE_PROJECT_DIR}'"

  rsync -az --delete "${SYNC_EXCLUDES[@]}" \
    "${LOCAL_PROJECT_DIR}/" "${host}:${REMOTE_PROJECT_DIR}/"

  ssh -n -o BatchMode=yes "${host}" "bash -lc '
    set -e
    cd "${REMOTE_PROJECT_DIR}"
    if [ ! -x env_projet/bin/python3 ]; then
      python3 -m venv env_projet
    fi
    env_projet/bin/python3 -m pip install --upgrade pip >/dev/null
    env_projet/bin/python3 -m pip install -r requirements.txt >/dev/null
    echo ready
  '"

done < "${HOSTS_FILE}"

echo "Remote setup completed."
