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

# ---- Patch model_management.py for Grace-Blackwell unified memory ----
# On unified memory systems (GB10/GB200), VRAM and RAM are the same physical
# pool. ComfyUI's defaults treat them as separate, causing pointless CPU
# offloading and cache thrashing. Four changes:
#   1. Detect unified memory (VRAM ≈ RAM ratio > 0.95, or GB device name)
#   2. maximum_vram_for_weights(): 95% instead of 88% (no separate VRAM pool)
#   3. intermediate_device(): return GPU instead of CPU (same physical pool)
#   4. soft_empty_cache(): skip empty_cache() to avoid page faults on re-alloc
RUN python - <<'PY'
from pathlib import Path

path = Path("/opt/ComfyUI/comfy/model_management.py")
text = path.read_text()
applied = 0

# 1. Insert unified memory detection after VRAMState.SHARED block
old1 = (
    "if cpu_state == CPUState.MPS:\n"
    "    vram_state = VRAMState.SHARED\n"
    "\n"
    'logging.info(f"Set vram state to: {vram_state.name}")'
)
new1 = (
    "if cpu_state == CPUState.MPS:\n"
    "    vram_state = VRAMState.SHARED\n"
    "\n"
    "# --- Grace-Blackwell Unified Memory Detection (Sparky) ---\n"
    "# On unified memory systems (Grace-Blackwell), VRAM and RAM are the same\n"
    "# physical memory. ComfyUI's default treats them as separate pools, causing\n"
    "# pointless CPU offloading and cache thrashing. Detect and optimize.\n"
    "def _is_unified_memory():\n"
    '    """Detect if GPU and CPU share the same physical memory pool."""\n'
    "    if cpu_state == CPUState.MPS:\n"
    "        return False  # MPS handles unified memory via VRAMState.SHARED\n"
    "    if cpu_state != CPUState.GPU:\n"
    "        return False\n"
    "    if not torch.cuda.is_available():\n"
    "        return False\n"
    "    try:\n"
    "        vram_bytes = torch.cuda.get_device_properties(0).total_memory\n"
    "        ram_bytes = psutil.virtual_memory().total\n"
    "        ratio = vram_bytes / ram_bytes if ram_bytes > 0 else 0\n"
    "        device_name = torch.cuda.get_device_properties(0).name.lower()\n"
    "        is_gb = 'gb10' in device_name or 'gb200' in device_name or 'grace' in device_name\n"
    "        if ratio > 0.95 or is_gb:\n"
    "            return True\n"
    "    except Exception:\n"
    "        pass\n"
    "    return False\n"
    "\n"
    "UNIFIED_MEMORY = _is_unified_memory()\n"
    "\n"
    "if UNIFIED_MEMORY:\n"
    "    if not (args.highvram or args.gpu_only):\n"
    '        logging.info("[Sparky] Grace-Blackwell unified memory detected — "\n'
    '                     "keeping NORMAL_VRAM mode (allows layer offloading)")\n'
    "\n"
    'logging.info(f"Set vram state to: {vram_state.name}")'
)
if old1 in text:
    text = text.replace(old1, new1, 1)
    applied += 1
else:
    print("WARNING: unified memory detection pattern not found")

# 2. maximum_vram_for_weights(): 95% on unified memory
old2 = (
    "def maximum_vram_for_weights(device=None):\n"
    "    return (get_total_memory(device) * 0.88 - minimum_inference_memory())"
)
new2 = (
    "def maximum_vram_for_weights(device=None):\n"
    "    if UNIFIED_MEMORY:\n"
    "        return (get_total_memory(device) * 0.95 - 2 * 1024 * 1024 * 1024)\n"
    "    return (get_total_memory(device) * 0.88 - minimum_inference_memory())"
)
if old2 in text:
    text = text.replace(old2, new2, 1)
    applied += 1
else:
    print("WARNING: maximum_vram_for_weights pattern not found")

# 3. intermediate_device(): return GPU on unified memory
old3 = (
    "def intermediate_device():\n"
    "    if args.gpu_only:\n"
    "        return get_torch_device()\n"
    "    else:\n"
    '        return torch.device("cpu")'
)
new3 = (
    "def intermediate_device():\n"
    "    if args.gpu_only or UNIFIED_MEMORY:\n"
    "        return get_torch_device()\n"
    "    else:\n"
    '        return torch.device("cpu")'
)
if old3 in text:
    text = text.replace(old3, new3, 1)
    applied += 1
else:
    print("WARNING: intermediate_device pattern not found")

# 4. soft_empty_cache(): skip empty_cache() on unified memory
old4 = (
    "def soft_empty_cache(force=False):\n"
    "    if cpu_mode():\n"
    "        return\n"
    "    global cpu_state\n"
    "    if cpu_state == CPUState.MPS:\n"
    "        torch.mps.empty_cache()\n"
    "    elif is_intel_xpu():"
)
new4 = (
    "def soft_empty_cache(force=False):\n"
    "    if cpu_mode():\n"
    "        return\n"
    "    global cpu_state\n"
    "    if cpu_state == CPUState.MPS:\n"
    "        torch.mps.empty_cache()\n"
    "        return\n"
    "    if UNIFIED_MEMORY and not force:\n"
    "        if torch.cuda.is_available():\n"
    "            torch.cuda.synchronize()\n"
    "        return\n"
    "    elif is_intel_xpu():"
)
if old4 in text:
    text = text.replace(old4, new4, 1)
    applied += 1
else:
    print("WARNING: soft_empty_cache pattern not found")

if applied == 0:
    raise SystemExit("No patches applied to model_management.py — all patterns missing")
path.write_text(text)
print(f"Applied {applied}/4 patches to model_management.py")
PY

RUN pip install -r /opt/ComfyUI/requirements.txt

# ---- Comfy Kitchen Blackwell Optimization ----
# Copy your local wheel into the build context
##COPY wheels/comfy_kitchen-0.2.7-cp312-abi3-linux_aarch64.whl /tmp/

# Force install the local wheel
# We use --force-reinstall to ensure it replaces any version installed by the requirements.txt
##RUN pip install --no-cache-dir --force-reinstall /tmp/comfy_kitchen-0.2.7-cp312-abi3-linux_aarch64.whl

# Cleanup the wheel from the layer to save space
##RUN rm /tmp/comfy_kitchen-0.2.7-cp312-abi3-linux_aarch64.whl

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
