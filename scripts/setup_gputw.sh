#!/usr/bin/env bash

# Bootstrap this project on a GPUtw instance.  GPUtw only guarantees that
# /vault persists, so the repository must be cloned or uploaded below it.

set -Eeuo pipefail

PYTHON_VERSION="3.12"
UV_VERSION="${GPUTW_UV_VERSION:-0.11.32}"
VAULT_DIR="${GPUTW_VAULT_DIR:-/vault}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
ENV_FILE="${PROJECT_ROOT}/.gputw-env"

log() {
    printf '[gputw-setup] %s\n' "$*"
}

die() {
    printf '[gputw-setup] ERROR: %s\n' "$*" >&2
    exit 1
}

run_as_root() {
    if [[ "${EUID}" -eq 0 ]]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        die "Installing system packages requires root access or sudo."
    fi
}

install_system_tools() {
    local missing=()
    command -v btop >/dev/null 2>&1 || missing+=(btop)
    command -v tmux >/dev/null 2>&1 || missing+=(tmux)

    if [[ "${#missing[@]}" -eq 0 ]]; then
        log "System tools already installed: btop, tmux."
        return
    fi

    log "Installing system tools: ${missing[*]}..."
    if command -v apt-get >/dev/null 2>&1; then
        run_as_root apt-get update
        run_as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y "${missing[@]}"
    elif command -v dnf >/dev/null 2>&1; then
        run_as_root dnf install -y "${missing[@]}"
    elif command -v yum >/dev/null 2>&1; then
        run_as_root yum install -y "${missing[@]}"
    elif command -v apk >/dev/null 2>&1; then
        run_as_root apk add --no-cache "${missing[@]}"
    else
        die "No supported package manager was found for btop and tmux."
    fi
}

command -v uname >/dev/null 2>&1 || die "uname is required."
[[ "$(uname -s)" == "Linux" ]] || die "This script must run inside a Linux GPUtw instance."
[[ -d "${VAULT_DIR}" ]] || die "Persistent storage was not found at ${VAULT_DIR}. Attach the GPUtw Vault and retry."
[[ -w "${VAULT_DIR}" ]] || die "Persistent storage is not writable: ${VAULT_DIR}"

VAULT_REAL="$(cd "${VAULT_DIR}" && pwd -P)"
case "${PROJECT_ROOT}/" in
    "${VAULT_REAL}/"*) ;;
    *)
        die "The repository is outside persistent storage (${PROJECT_ROOT}). Clone or upload it below ${VAULT_REAL}, then run scripts/setup_gputw.sh there."
        ;;
esac

[[ -f "${PROJECT_ROOT}/pyproject.toml" ]] || die "pyproject.toml is missing from ${PROJECT_ROOT}."
[[ -f "${PROJECT_ROOT}/uv.lock" ]] || die "uv.lock is missing from ${PROJECT_ROOT}."

PERSIST_ROOT="${VAULT_REAL}/.net-packages"
BIN_DIR="${PERSIST_ROOT}/bin"
CACHE_DIR="${PERSIST_ROOT}/cache"
PYTHON_DIR="${PERSIST_ROOT}/python"

mkdir -p \
    "${BIN_DIR}" \
    "${CACHE_DIR}/uv" \
    "${CACHE_DIR}/huggingface/hub" \
    "${CACHE_DIR}/sentence-transformers" \
    "${CACHE_DIR}/torch" \
    "${CACHE_DIR}/matplotlib" \
    "${PYTHON_DIR}" \
    "${PROJECT_ROOT}/Year=2022" \
    "${PROJECT_ROOT}/outputs"

# Every large or reusable artifact is rooted in the persistent Vault.
export PATH="${BIN_DIR}:${PATH}"
export UV_CACHE_DIR="${CACHE_DIR}/uv"
export UV_PYTHON_INSTALL_DIR="${PYTHON_DIR}"
export UV_PYTHON_BIN_DIR="${BIN_DIR}"
export UV_PROJECT_ENVIRONMENT="${PROJECT_ROOT}/.venv"
export HF_HOME="${CACHE_DIR}/huggingface"
export HF_HUB_CACHE="${CACHE_DIR}/huggingface/hub"
export SENTENCE_TRANSFORMERS_HOME="${CACHE_DIR}/sentence-transformers"
export TORCH_HOME="${CACHE_DIR}/torch"
export MPLCONFIGDIR="${CACHE_DIR}/matplotlib"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export PYTHONUNBUFFERED="1"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    die "nvidia-smi is unavailable. Redeploy with a GPUtw CUDA or PyTorch/Jupyter template; the container script cannot install the host NVIDIA driver."
fi

log "Detected NVIDIA driver and GPU:"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

install_system_tools
log "btop: $(btop --version | head -n 1)"
log "tmux: $(tmux -V)"

INSTALLED_UV_VERSION=""
if [[ -x "${BIN_DIR}/uv" ]]; then
    INSTALLED_UV_VERSION="$("${BIN_DIR}/uv" --version | awk '{print $2}')"
fi

if [[ "${INSTALLED_UV_VERSION}" != "${UV_VERSION}" ]]; then
    log "Installing uv ${UV_VERSION} into persistent storage..."
    UV_INSTALL_URL="https://astral.sh/uv/${UV_VERSION}/install.sh"
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf "${UV_INSTALL_URL}" | env \
            UV_INSTALL_DIR="${BIN_DIR}" \
            UV_NO_MODIFY_PATH=1 \
            sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- "${UV_INSTALL_URL}" | env \
            UV_INSTALL_DIR="${BIN_DIR}" \
            UV_NO_MODIFY_PATH=1 \
            sh
    else
        die "curl or wget is required to install uv."
    fi
fi

log "Using $(uv --version)."
log "Installing managed Python ${PYTHON_VERSION} into ${PYTHON_DIR}..."
uv python install "${PYTHON_VERSION}"

log "Installing all locked project dependencies (including parquet and notebook extras)..."
cd "${PROJECT_ROOT}"
uv sync --frozen --all-extras --python "${PYTHON_VERSION}"
log "gdown: $(uv run --frozen gdown --version)"

cat >"${ENV_FILE}" <<EOF
# Generated by scripts/setup_gputw.sh. Source this file in each new shell.
export PATH="${BIN_DIR}:\${PATH}"
export UV_CACHE_DIR="${CACHE_DIR}/uv"
export UV_PYTHON_INSTALL_DIR="${PYTHON_DIR}"
export UV_PYTHON_BIN_DIR="${BIN_DIR}"
export UV_PROJECT_ENVIRONMENT="${PROJECT_ROOT}/.venv"
export HF_HOME="${CACHE_DIR}/huggingface"
export HF_HUB_CACHE="${CACHE_DIR}/huggingface/hub"
export SENTENCE_TRANSFORMERS_HOME="${CACHE_DIR}/sentence-transformers"
export TORCH_HOME="${CACHE_DIR}/torch"
export MPLCONFIGDIR="${CACHE_DIR}/matplotlib"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export PYTHONUNBUFFERED="1"
cd "${PROJECT_ROOT}"
EOF
chmod 600 "${ENV_FILE}"

log "Verifying Python, PyTorch, CUDA, and an actual GPU operation..."
uv run --frozen python - <<'PY'
import sys

import torch

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA verification failed: torch.cuda.is_available() is False. "
        "Check the GPUtw template and NVIDIA driver compatibility with the locked PyTorch build."
    )

for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    print(
        f"GPU {index}: {properties.name}; "
        f"compute capability={properties.major}.{properties.minor}; "
        f"VRAM={properties.total_memory / 1024**3:.1f} GiB"
    )

device = torch.device("cuda:0")
left = torch.randn((512, 512), device=device)
right = torch.randn((512, 512), device=device)
checksum = (left @ right).sum().item()
torch.cuda.synchronize(device)
print(f"CUDA matrix multiplication: OK (checksum={checksum:.6f})")
PY

log "Persistent filesystem usage:"
df -h "${VAULT_REAL}"

cat <<EOF

Setup complete.

In every new SSH/Jupyter terminal, run:
  source "${ENV_FILE}"

Persistent locations:
  input data : ${PROJECT_ROOT}/Year=2022
  run outputs: ${PROJECT_ROOT}/outputs
  environment: ${PROJECT_ROOT}/.venv
  caches     : ${CACHE_DIR}

Example training command:
  uv run python experiments/disentangled_cvae_step1/run_experiment.py \\
    --config experiments/disentangled_cvae_step1/configs/default.yaml \\
    --stage all
EOF
