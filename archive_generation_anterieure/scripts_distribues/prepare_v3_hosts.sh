#!/bin/bash
set -euo pipefail

LOCAL_PROJECT_DIR="/users/local/l24virbe/Projet_ML"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-/homes/l24virbe/Projet_ML}"
HOSTS_FILE="${LOCAL_PROJECT_DIR}/hosts_v3.txt"

if [[ ! -f "${HOSTS_FILE}" ]]; then
  echo "Hosts file missing: ${HOSTS_FILE}"
  exit 1
fi

mapfile -t HOSTS < "${HOSTS_FILE}"

for host in "${HOSTS[@]}"; do
  [[ -z "${host}" ]] && continue
  echo "=== ${host} ==="

  if ! ssh -n -o BatchMode=yes -o ConnectTimeout=5 "${host}" "echo ok" >/dev/null 2>&1; then
    echo "[SKIP] unreachable"
    continue
  fi

  ssh -n -o BatchMode=yes "${host}" "mkdir -p '${REMOTE_PROJECT_DIR}'"

  # Sync code needed for generation; exclude large/generated artifacts and local venv.
  rsync -az --delete \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude 'env_projet/' \
    --exclude 'analysis_visuals_1k/' \
    --exclude 'analysis_visuals_50k/' \
    --exclude 'JSON/' \
    --exclude 'dataset*.json' \
    --exclude '*_analysis.json' \
    --exclude '*_analysis_summary.txt' \
    --exclude 'distributed_runs/' \
    --exclude '*.log' \
    "${LOCAL_PROJECT_DIR}/" "${host}:${REMOTE_PROJECT_DIR}/"

  ssh -n -o BatchMode=yes "${host}" "
    cd '${REMOTE_PROJECT_DIR}' &&
    python3 -m venv env_projet &&
    source env_projet/bin/activate &&
    python3 -m pip install --upgrade pip &&
    python3 -m pip install -r requirements.txt
  "

  echo "[OK] prepared"
done
