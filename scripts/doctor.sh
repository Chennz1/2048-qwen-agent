#!/bin/bash
# Environment diagnostics for this project.
# ======================= 参数说明区 =======================
# 用法:
#   bash scripts/doctor.sh
#
# 参数:
#   本脚本不接收命令行参数。
#   作用是检测:
#   1) 常用命令是否存在
#   2) data/checkpoints/scripts 目录是否可写
#   3) Python/ML 依赖和 CUDA 可用性
#
# 示例:
#   bash scripts/doctor.sh
# ========================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

print_kv() {
  printf "%-26s %s\n" "$1" "$2"
}

check_cmd() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    print_kv "$name" "ok ($(command -v "$name"))"
  else
    print_kv "$name" "missing"
  fi
}

echo "=== Project Doctor ==="
print_kv "project_root" "$ROOT_DIR"
print_kv "date" "$(date '+%Y-%m-%d %H:%M:%S %z')"
echo

echo "[Commands]"
for cmd in python pip git pytest ruff black isort mypy shellcheck; do
  check_cmd "$cmd"
done
echo

echo "[Writable Paths]"
for path in data checkpoints scripts; do
  if mkdir -p "$path" 2>/dev/null; then
    print_kv "$path" "writable"
  else
    print_kv "$path" "not writable"
  fi
done
echo

echo "[Python/ML Stack]"
python - <<'PY'
import importlib
import platform
import sys

MODULES = [
    "torch",
    "transformers",
    "datasets",
    "trl",
    "peft",
    "vllm",
    "bitsandbytes",
]

def version_of(mod):
    return getattr(mod, "__version__", "unknown")

print(f"{'python':<26} {sys.version.split()[0]}")
print(f"{'platform':<26} {platform.platform()}")

for name in MODULES:
    try:
        mod = importlib.import_module(name)
        print(f"{name:<26} ok ({version_of(mod)})")
    except Exception as e:
        print(f"{name:<26} missing ({type(e).__name__})")

try:
    trl_mod = importlib.import_module("trl")
    has_top = hasattr(trl_mod, "GRPOConfig") and hasattr(trl_mod, "GRPOTrainer")
    if has_top:
        print(f"{'trl_grpo_api':<26} ok (top-level)")
    else:
        importlib.import_module("trl.trainer.grpo_config")
        importlib.import_module("trl.trainer.grpo_trainer")
        print(f"{'trl_grpo_api':<26} ok (submodule)")
except Exception as e:
    print(f"{'trl_grpo_api':<26} missing ({type(e).__name__})")

try:
    import torch
    print(f"{'cuda_available':<26} {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"{'cuda_device_count':<26} {torch.cuda.device_count()}")
        print(f"{'cuda_device_0':<26} {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"{'cuda_probe':<26} failed ({type(e).__name__})")
PY

echo
echo "Doctor check completed."
