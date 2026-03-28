# 🚀 ComfyUI on DGX Spark (Blackwell GB10)

A Docker Compose setup for running [ComfyUI](https://github.com/comfyanonymous/ComfyUI) on the **NVIDIA DGX Spark** (Grace-Blackwell GB10), with a mobile-friendly UI included.

Built specifically to handle the quirks of the **sm_121 / compute 12.1** architecture and its **unified CPU-GPU memory fabric**.

---

## ✨ Features

- **CUDA 13.1 base** — full `nvcc` support for GB10 (`sm_121`), enabling CUDA extension compilation
- **PyTorch cu130** — prebuilt ARM64 wheels from PyTorch's cu130 index
- **SageAttention 2** — compiled from source directly against sm_121 for full hardware attention acceleration
- **Comfy Kitchen** (`comfy_kitchen`) — NVFP4 quantization support for Blackwell
- **Unified-memory optimized flags** — carefully tuned `COMFYUI_FLAGS` that avoid fighting the Grace-Blackwell memory fabric
- **Double-VRAM bug fix** — patches `comfy/utils.py` to set `copy=False` in `tensor.to()`, fixing the double memory usage on unified memory systems with `--disable-mmap`
- **Disabled dynamic vram** — uses `--disable-dynamic-vram` as it doesn't work properly on the Spark, if models fit in memory they won't be unloaded, faster prompt changes to final image/video
- **ComfyUI-Manager** — auto-installed at container startup into the mounted `custom_nodes` volume
- **ComfyUIMini** — lightweight mobile/tablet UI proxying to the ComfyUI backend (optional second service)
- **Health checks** — both services expose health check endpoints for reliable `depends_on` startup ordering
- **Persistent volumes** — models, custom nodes, outputs, inputs, user settings, and workflows are all mounted from the host

---

## 🗂️ Repo Structure

```
.
├── Dockerfile            # Main ComfyUI image (CUDA 13.1, PyTorch, SageAttention, Comfy Kitchen)
├── docker-compose.yml    # Orchestrates comfyui + comfyuimini services
├── entrypoint.sh         # Runtime startup: installs ComfyUI-Manager, custom node deps, launches ComfyUI
├── copy-wheels.sh        # Helper to stage local .whl files into the build context
├── .env.example          # Example environment file — copy to .env and customize
├── wheels/               # Place comfy_kitchen*.whl here before building
└── comfyuimini/
    └── Dockerfile        # Lightweight Node.js image for ComfyUIMini mobile UI
```

---

## ⚙️ Prerequisites

- NVIDIA DGX Spark (or any Grace-Blackwell system with `sm_121`)
- Docker with the NVIDIA Container Toolkit installed
- `comfy_kitchen-0.2.7-cp312-abi3-linux_aarch64.whl` — place in `~/comfyui/wheels/` (or wherever you keep your ML wheels)

---

## 🛠️ Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/DGX-Spark-ComfyUI.git
cd DGX-Spark-ComfyUI
```

### 2. Stage wheels - Not required if downloading the whole repo as the custom built comfy_kitchen wheel is included

Copy the required wheels into the build context:

```bash
bash copy-wheels.sh ~/comfyui/wheels
```

This copies `comfy_kitchen*.whl` into the local `wheels/` directory used by the Docker build.

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your paths:

```env
# Where your ComfyUI models live on the host
COMFYUI_HOST_PATH=/home/user/comfyui

# Where ComfyUI data lives (custom_nodes, output, input, etc.)
COMFYUI_DATA_PATH=/home/user/comfyui
```

### 4. Build and start

```bash
docker compose up --build -d
```

ComfyUI will be available at **`http://<host-ip>:8188`**.  
ComfyUIMini (mobile UI) will be available at **`http://<host-ip>:3000`**.

---

## 🔧 Optimized Flags Explained

The default `COMFYUI_FLAGS` are tuned for the Grace-Blackwell unified memory architecture:

| Flag | Reason |
|---|---|
| `--normalvram` | Enforces normal vram mode |
| `--disable-dynamic-vram` | Disables dynamic vram, keeps models in memory as long as they fit, faster prompt changes to final output |
| `--reserve-vram 1` | Reserves 1gb of vram to the system, i found it works a bit better with the other flags |
| `--disable-pinned-memory` | Reduces overhead on the unified memory fabric; pinned memory is counterproductive here |
| `--use-sage-attention` | Enables SageAttention compiled for sm_121 |
| `--force-fp16` | Enables Flash Attention path in PyTorch |
| `--bf16-unet --bf16-vae --bf16-text-enc` | Keeps models in BF16 — Blackwell's native precision |
| `--dont-upcast-attention` | Keeps attention ops in FP16/BF16 for speed |
| `--disable-mmap` | Combined with the `copy=False` patch, fixes double memory usage on unified memory |

> **Note:** Do **not** use `--gpu-only`. It forces a split memory model that fights the unified memory fabric on Grace-Blackwell systems.
> **Note:** Do **not** use `--high-vram`. It pins every model in memory causing OOM issues.

### Environment variables

| Variable | Purpose |
|---|---|
| `PYTORCH_NO_CUDA_MEMORY_CACHING=1` | Disables PyTorch CUDA caching allocator — lets the unified memory manager handle allocations |
| `CUDA_MANAGED_FORCE_DEVICE_ALLOC=1` | Forces managed memory to prefer device allocation |
| `TORCH_COMPILE_DISABLE=1` / `TORCHDYNAMO_DISABLE=1` | Disables `torch.compile` — Triton does not support `sm_121a` yet |
| `CUDA_MODULE_LOADING=LAZY` | Loads CUDA modules lazily for faster model loading and more memory efficiency |
| `OMP_NUM_THREADS=20` | Tuned for the Grace CPU core count |

---

## 📦 Volumes

| Host path | Container path | Purpose |
|---|---|---|
| `${COMFYUI_HOST_PATH}/models` | `/opt/ComfyUI/models` | Model files (shared with host) |
| `${COMFYUI_DATA_PATH}/custom_nodes` | `/opt/ComfyUI/custom_nodes` | Custom nodes (incl. ComfyUI-Manager) |
| `${COMFYUI_DATA_PATH}/user` | `/opt/ComfyUI/user` | User settings & ComfyUI-Manager config |
| `${COMFYUI_DATA_PATH}/output` | `/opt/ComfyUI/output` | Generated images/videos |
| `${COMFYUI_DATA_PATH}/input` | `/opt/ComfyUI/input` | Input files |
| `${COMFYUI_DATA_PATH}/workflows` | `/opt/ComfyUI/workflows` | Saved workflows |

---

## 📱 ComfyUIMini (Mobile UI)

[ComfyUIMini](https://github.com/ImDarkTom/ComfyUIMini) is included as an optional second service. It provides a lightweight, mobile-friendly interface that proxies requests to the ComfyUI backend over the internal Docker network.

It starts automatically after ComfyUI passes its health check and shares the `output` directory for gallery access.

---

## 🩹 Patches Applied

### `comfy/utils.py` — `copy=False` fix

The Dockerfile patches one line in ComfyUI's source at build time:

```python
# Before
tensor = tensor.to(device=device, copy=True)

# After
tensor = tensor.to(device=device, copy=False)
```

On unified memory systems, `copy=True` causes tensors to be duplicated unnecessarily, effectively doubling VRAM usage when `--disable-mmap` is active. Setting `copy=False` eliminates this.

---

## 🔄 Updating ComfyUI

The `COMFYUI_REF` variable (in `.env` or at build time) controls which commit/tag/branch of ComfyUI is checked out. Pin it for reproducibility:

```env
COMFYUI_REF=v0.3.43
```

ComfyUI-Manager is **not** baked into the image — it is cloned fresh at container startup into the mounted `custom_nodes` volume, so you always get the latest version.

---

## 📄 License

MIT
