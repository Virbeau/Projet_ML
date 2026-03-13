#!/bin/bash
set -euo pipefail

PROJECT_DIR="/users/local/l24virbe/Projet_ML"
HOSTS_FILE="${PROJECT_DIR}/hosts_v3.txt"
KEY_PATH="${HOME}/.ssh/id_ed25519"

if [[ ! -f "${KEY_PATH}" ]]; then
  echo "No SSH key found. Creating ${KEY_PATH}..."
  ssh-keygen -t ed25519 -N "" -f "${KEY_PATH}"
fi

if [[ ! -f "${HOSTS_FILE}" ]]; then
  echo "Missing hosts file: ${HOSTS_FILE}"
  exit 1
fi

echo "Ensuring SSH config has ControlMaster and key settings..."
mkdir -p "${HOME}/.ssh"
touch "${HOME}/.ssh/config"
chmod 600 "${HOME}/.ssh/config"

if ! grep -q "# v3-distributed" "${HOME}/.ssh/config"; then
  cat >> "${HOME}/.ssh/config" <<'EOF'
# v3-distributed
Host fl-tp-br-*
  User l24virbe
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
  ServerAliveInterval 30
  ServerAliveCountMax 3
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 10m
EOF
fi

echo "Copying key to all hosts (one-time password may be required per host)..."
mapfile -t HOSTS < "${HOSTS_FILE}"

for host in "${HOSTS[@]}"; do
  [[ -z "${host}" ]] && continue
  echo "  -> ${host}"
  ssh-copy-id -i "${KEY_PATH}.pub" "${host}" < /dev/null || true
done

echo "Testing BatchMode access..."
for host in "${HOSTS[@]}"; do
  [[ -z "${host}" ]] && continue
  if ssh -n -o BatchMode=yes "${host}" "echo ok" >/dev/null 2>&1; then
    echo "  [OK] ${host}"
  else
    echo "  [KO] ${host}"
  fi
done

echo "SSH setup complete."
