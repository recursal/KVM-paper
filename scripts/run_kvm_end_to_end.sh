#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

python_bin="$repo_root/.venv/bin/python"
gpu_slots=${2:-8}
run_stamp=$(date -u +%Y%m%d_%H%M%S)
output_root=${1:-"$repo_root/experiments/kvm_benchmark_$run_stamp"}

if [[ ! -x "$python_bin" ]]; then
    echo "missing virtual-environment interpreter: $python_bin" >&2
    exit 2
fi
if [[ -e "$output_root" ]]; then
    echo "output path already exists: $output_root" >&2
    exit 2
fi

benchmark_root="$output_root/fixed_context"
decode_series_root="$output_root/decode_series"

"$python_bin" scripts/benchmark_kvm.py submit \
    --max-parallel "$gpu_slots" \
    --root "$benchmark_root"
"$python_bin" scripts/benchmark_kvm.py analyze \
    --root "$benchmark_root"

"$python_bin" scripts/benchmark_kvm.py decode-series-submit \
    --max-parallel "$gpu_slots" \
    --root "$decode_series_root"
"$python_bin" scripts/benchmark_kvm.py decode-series-analyze \
    --root "$decode_series_root"

printf 'KVM benchmark results: %s\n' "$output_root"
