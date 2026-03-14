#!/usr/bin/env bash
set -euo pipefail

COMFY_DIR="/opt/ComfyUI"
PORT="${COMFYUI_PORT:-8188}"
FLAGS="${COMFYUI_FLAGS:---listen 0.0.0.0 --port ${PORT}}"

echo "[entrypoint] Python: $(python --version)"
echo "[entrypoint] Torch:  $(python -c 'import torch; print(torch.__version__)')"
echo "[entrypoint] CUDA:   $(python -c 'import torch; print(torch.version.cuda)')"
echo "[entrypoint] Flags:  ${FLAGS}"

# Ensure ComfyUI-Manager exists in mounted custom_nodes
# Check for __init__.py to detect corrupted/partial installs
if [[ ! -f "${COMFY_DIR}/custom_nodes/ComfyUI-Manager/__init__.py" ]]; then
    echo "[entrypoint] ComfyUI-Manager missing or corrupted, cloning latest..."
    rm -rf "${COMFY_DIR}/custom_nodes/ComfyUI-Manager" 2>/dev/null || true
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git \
        "${COMFY_DIR}/custom_nodes/ComfyUI-Manager" || true
fi

# Install any requirements from custom nodes
for req in "${COMFY_DIR}"/custom_nodes/*/requirements.txt; do
    if [[ -f "$req" ]]; then
        echo "[entrypoint] Installing deps from: $req"
        pip install -q -r "$req" || true
    fi
done

exec python "${COMFY_DIR}/main.py" ${FLAGS}
