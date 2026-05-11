#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/install.sh [options]

Options:
  --gpu-backend {rocm72|rocm71|cuda}   GPU runtime to install. Default: rocm72
  --python <version>                   Python version for the uv-managed venv. Default: 3.12
  --with-rwkv-kernels                  Install flash-linear-attention for rwkv7 and other FLA mixers
  --with-flash-attention               Install flash-attn for FlashAttention layers
  -h, --help                           Show this help text

Examples:
  scripts/install.sh
  scripts/install.sh --gpu-backend rocm71 --with-rwkv-kernels
  scripts/install.sh --gpu-backend rocm72 --with-flash-attention
  scripts/install.sh --gpu-backend cuda
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_BACKEND="rocm72"
PYTHON_VERSION="3.12"
WITH_RWKV_KERNELS=0
WITH_FLASH_ATTENTION=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-backend)
      GPU_BACKEND="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="${2:-}"
      shift 2
      ;;
    --with-rwkv-kernels)
      WITH_RWKV_KERNELS=1
      shift
      ;;
    --with-flash-attention)
      WITH_FLASH_ATTENTION=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "$GPU_BACKEND" in
  rocm71)
    BACKEND_EXTRA="rocm71"
    ;;
  rocm72)
    BACKEND_EXTRA="rocm72"
    ;;
  cuda)
    BACKEND_EXTRA="cu128"
    ;;
  *)
    echo "Unsupported --gpu-backend: $GPU_BACKEND" >&2
    usage >&2
    exit 1
    ;;
esac

if [[ "$BACKEND_EXTRA" == rocm71 || "$BACKEND_EXTRA" == rocm72 ]]; then
  if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "ROCm installs in this script target Linux x86_64 only." >&2
    exit 1
  fi
fi

SYNC_ARGS=(
  --python "$PYTHON_VERSION"
  --extra "$BACKEND_EXTRA"
)

if [[ "$WITH_RWKV_KERNELS" -eq 1 ]]; then
  SYNC_ARGS+=(--extra rwkv-kernels)
fi

if [[ "$WITH_FLASH_ATTENTION" -eq 1 ]]; then
  SYNC_ARGS+=(--extra flash-attention)
fi

cd "$ROOT_DIR"

uv venv --clear --python "$PYTHON_VERSION" .venv
uv sync "${SYNC_ARGS[@]}"

uv run python -c "import torch; print('torch', torch.__version__); print('hip', getattr(torch.version, 'hip', None)); print('cuda', getattr(torch.version, 'cuda', None)); print('cuda_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

cat <<EOF

Environment ready in $ROOT_DIR/.venv

If you plan to use rwkv7 or other FLA-based mixers, rerun this script with --with-rwkv-kernels.
If you plan to use FlashAttention layers, rerun this script with --with-flash-attention.
EOF
