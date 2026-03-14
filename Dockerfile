# CUDA 13.0 for Blackwell GB10 (sm_121 / compute_121)
# CUDA 12.8 only supports up to sm_120, but GB10 is sm_121.
# "devel" includes nvcc so we can compile CUDA extensions like SageAttention.
FROM nvidia/cuda:13.1.1-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG COMFYUI_REF=master
ARG SAGEATTN_REF=main

# Base system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    python3 python3-pip python3-venv python3-dev \
    build-essential ninja-build cmake pkg-config \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxcb1 \
    libtcmalloc-minimal4 \
    && rm -rf /var/lib/apt/lists/*

# Create venv (keeps python deps isolated inside container)
ENV VENV=/opt/venv
RUN python3 -m venv $VENV
ENV PATH="$VENV/bin:$PATH"

# Upgrade packaging tools
RUN pip install -U pip setuptools wheel pynvml

# ---- PyTorch (ARM64 + CUDA 13.0) ----
# PyTorch cu130 wheels work with CUDA 13.0.x runtime.
RUN pip install --index-url https://download.pytorch.org/whl/cu130 \
    torch torchvision

# ---- ComfyUI ----
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI && \
    cd /opt/ComfyUI && \
    git checkout ${COMFYUI_REF} || true

# ---- Patch utils.py to set tensor.to() to False if --disable-mmap enabled - Fixes double RAM/VRAM usage bug ----
RUN python - <<'PY'
from pathlib import Path

path = Path("/opt/ComfyUI/comfy/utils.py")
text = path.read_text()
old = "tensor = tensor.to(device=device, copy=True)"
new = "tensor = tensor.to(device=device, copy=False)"
if old not in text:
    raise SystemExit("Expected pattern not found in comfy/utils.py")
path.write_text(text.replace(old, new, 1))
PY

RUN pip install -r /opt/ComfyUI/requirements.txt

# ---- Comfy Kitchen Blackwell Optimization ----
# Copy your local wheel into the build context
COPY wheels/comfy_kitchen-0.2.7-cp312-abi3-linux_aarch64.whl /tmp/

# Force install the local wheel
# We use --force-reinstall to ensure it replaces any version installed by the requirements.txt
RUN pip install --no-cache-dir --force-reinstall /tmp/comfy_kitchen-0.2.7-cp312-abi3-linux_aarch64.whl

# Cleanup the wheel from the layer to save space
RUN rm /tmp/comfy_kitchen-0.2.7-cp312-abi3-linux_aarch64.whl

# ---- ComfyUI-Manager ----
# Handled at runtime by entrypoint.sh (clones if missing in mounted volume)
# This ensures latest version on each container start

# ---- SageAttention ----
# GB10 is compute capability 12.1 (sm_121).
# CUDA 13.0 NVCC supports sm_121, so we compile directly for it.
ENV TORCH_CUDA_ARCH_LIST="12.1"
ENV CUDA_HOME=/usr/local/cuda

# Build/install SageAttention from repo with sm_121 support
RUN pip install --no-build-isolation "git+https://github.com/thu-ml/SageAttention@${SAGEATTN_REF}" || true

# Expose ComfyUI
EXPOSE 8188

# Entry script handles runtime updates / flags
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
