#!/bin/bash
set -euo pipefail

LOCAL_PROJECT_DIR="/users/local/l24virbe/Projet_ML"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-/users/local/l24virbe/Projet_ML}"
ENV_ACTIVATE="${REMOTE_PROJECT_DIR}/env_projet/bin/activate"
HOSTS_FILE="${HOSTS_FILE:-${LOCAL_PROJECT_DIR}/hosts_v3.txt}"
RUN_ID="${1:-v3_60000}"
TOTAL_INSTANCES="${2:-60000}"
WORKERS_PER_HOST="${3:-3}"
BASE_SEED="${4:-20260311}"
SSH_OPTS=(
  -n
  -o BatchMode=yes
  -o ConnectTimeout=3
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
)

REMOTE_OUT_DIR="${REMOTE_PROJECT_DIR}/distributed_runs/${RUN_ID}"
REMOTE_LOG_DIR="${REMOTE_PROJECT_DIR}/distributed_runs/${RUN_ID}/logs"

if [[ ! -f "${HOSTS_FILE}" ]]; then
  echo "Hosts file missing: ${HOSTS_FILE}"
  exit 1
fi

mapfile -t HOSTS < "${HOSTS_FILE}"
N_HOSTS_RAW="${#HOSTS[@]}"
if [[ "${N_HOSTS_RAW}" -eq 0 ]]; then
  echo "No hosts found in ${HOSTS_FILE}"
  exit 1
fi

# Pré-filtre des hôtes joignables avec le bon répertoire distant
READY_HOSTS=()
for host in "${HOSTS[@]}"; do
  [[ -z "${host}" ]] && continue
  if ssh "${SSH_OPTS[@]}" "${host}" "test -d '${REMOTE_PROJECT_DIR}'" >/dev/null 2>&1; then
    READY_HOSTS+=("${host}")
  else
    echo "[${host}] skipped (unreachable or missing ${REMOTE_PROJECT_DIR})"
  fi
done

N_HOSTS="${#READY_HOSTS[@]}"
if [[ "${N_HOSTS}" -eq 0 ]]; then
  echo "No ready hosts available."
  exit 1
fi

BASE_PER_HOST=$((TOTAL_INSTANCES / N_HOSTS))
REMAINDER=$((TOTAL_INSTANCES % N_HOSTS))

echo "Run ID: ${RUN_ID}"
echo "Total instances: ${TOTAL_INSTANCES}"
echo "Hosts ready: ${N_HOSTS}/${N_HOSTS_RAW}"
echo "Instances/host: ${BASE_PER_HOST} (+1 for first ${REMAINDER} host(s))"

i=0
launched=0
failed=0
failed_hosts=()
for host in "${READY_HOSTS[@]}"; do
  count="${BASE_PER_HOST}"
  if [[ "${i}" -lt "${REMAINDER}" ]]; then
    count=$((count + 1))
  fi

  seed=$((BASE_SEED + i))
  part_name="part_${i}_$(echo "${host}" | tr '-' '_').json"
  out_path="${REMOTE_OUT_DIR}/${part_name}"
  log_path="${REMOTE_LOG_DIR}/progress_${i}_$(echo "${host}" | tr '-' '_').log"
  stdout_path="${REMOTE_LOG_DIR}/stdout_${i}_$(echo "${host}" | tr '-' '_').log"

  echo "[${host}] launch count=${count} seed=${seed}"

  if ssh "${SSH_OPTS[@]}" "${host}" "bash -lc '
    set -e
    mkdir -p "${REMOTE_OUT_DIR}" "${REMOTE_LOG_DIR}"
    cd "${REMOTE_PROJECT_DIR}"
    RUNPY="python3"
    if [ -x "${REMOTE_PROJECT_DIR}/env_projet/bin/python3" ]; then
      RUNPY="${REMOTE_PROJECT_DIR}/env_projet/bin/python3"
    fi
    nohup "\${RUNPY}" main_production.py \
      --n-instances "${count}" \
      --seed "${seed}" \
      --workers "${WORKERS_PER_HOST}" \
      --out-path "${out_path}" \
      --log-file "${log_path}" \
      > "${stdout_path}" 2>&1 &
    pid=\$!
    echo "\${pid}" > "${REMOTE_LOG_DIR}/pid_${i}_$(echo "${host}" | tr '-' '_').txt"
  '"; then
    launched=$((launched + 1))
  else
    echo "[${host}] launch failed (ssh or remote command)"
    failed=$((failed + 1))
    failed_hosts+=("${host}")
  fi

  i=$((i + 1))
done

echo ""
echo "Launch complete for all hosts."
echo "Launched: ${launched} | Failed: ${failed}"
if [[ "${failed}" -gt 0 ]]; then
  echo "Failed hosts: ${failed_hosts[*]}"
fi
echo "Monitor with: ./status_v3_distributed.sh ${RUN_ID}"
