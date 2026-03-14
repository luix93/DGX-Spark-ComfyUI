#!/bin/bash
# Copy only the wheels needed for ComfyUI
# Container already includes: PyTorch, TorchVision, TorchAudio, Triton, ONNX

set -euo pipefail

MLWHEEL_SOURCE="${1:-$HOME/comfyui/wheels}"
COMFYUI_DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/wheels"

echo "Copying custom wheels from: $MLWHEEL_SOURCE"
echo "Destination: $COMFYUI_DEST"

mkdir -p "$COMFYUI_DEST"

# Only copy what the container doesn't have:
# 1. Flash Attention 3 (sm_121 patches)
# 2. Comfy Kitchen (NVFP4 quantization)

if ls "$MLWHEEL_SOURCE"/flash_attn_3*.whl 1>/dev/null 2>&1; then
    echo "✓ Copying Flash Attention 3..."
    cp "$MLWHEEL_SOURCE"/flash_attn_3*.whl "$COMFYUI_DEST/"
else
    echo "⚠ Warning: flash_attn_3*.whl not found in $MLWHEEL_SOURCE"
fi

if ls "$MLWHEEL_SOURCE"/comfy*.whl 1>/dev/null 2>&1; then
    echo "✓ Copying Comfy Kitchen..."
    cp "$MLWHEEL_SOURCE"/comfy*.whl "$COMFYUI_DEST/"
else
    echo "⚠ Warning: comfy*.whl not found in $MLWHEEL_SOURCE"
fi

echo ""
echo "Wheels copied successfully:"
ls -lh "$COMFYUI_DEST"/*.whl 2>/dev/null || echo "No wheels found!"
echo ""
echo "NGC PyTorch container will provide:"
echo "  - PyTorch 2.6+ (NVIDIA optimized)"
echo "  - TorchVision, TorchAudio"
echo "  - Triton, ONNX"
echo "  - CUDA 13.0, cuDNN 9, NCCL"
echo "  - TransformerEngine (FP8)"
