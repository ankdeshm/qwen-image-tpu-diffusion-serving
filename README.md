# `Qwen/Qwen-Image` (20B MMDiT) & `Qwen/Qwen-Image-2.1` (`Qwen3-VL`) Diffusion Serving on Cloud TPU `v5p` & `v7x`

Qwen/Qwen-Image (20B MMDiT) & Qwen/Qwen-Image-2.1 / Qwen-Image-2.1-Turbo (Qwen3-VL Text-Encoded Image Diffusion) Serving on Cloud TPU v5p & v7x with vLLM-Omni, TorchTPU, & Disaggregated Image Encoding.

---

## 1. Model & System Architecture

| Component | Specification |
|---|---|
| **Checkpoints** | [`Qwen/Qwen-Image`](https://huggingface.co/Qwen/Qwen-Image) (`20B` MMDiT), [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) (`28`-step `Qwen3-VL`), & [`Qwen/Qwen-Image-2.1-Turbo`](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) (`8`-step `Qwen3-VL`) |
| **Architecture** | Family A: `Qwen2.5-VL` + `QwenImageTransformer2DModel` (`20B` MMDiT) · Family B (`Qwen 3 Image`): `Qwen3VLForConditionalGeneration` + `QwenImage21Transformer2DModel` + `AutoencoderKLQwenImage21` |
| **Serving Engines** | `vllm-omni` HTTP (`vllm serve --omni --ulysses-degree {1,2,4}`) + `Diffusers` `QwenImage21Pipeline` + Compiled `TorchTPU` (`BF16` & `FP8` MXU) |
| **Resolutions** | `512x512`, `1024x1024`, `1328x1328`, `1536x1024`, `1664x928`, and `2048x2048` (`2K` native) |
| **Disaggregated CPU Offload** | `multimodal-cpu-offload-svc` (`c3-standard-44` AVX-512): `POST /encode_image` parallel PNG/WebP encoding & TI2I `POST /preprocess_vision` |

---

## 2. Disaggregated CPU Offloading Architecture (`multimodal-cpu-offload-svc`)

All non-TPU pre-processing and media encoding work is offloaded over intra-cluster `ClusterIP` HTTP (`port 8090`) to dedicated `c3-standard-44` (`44` Sapphire Rapids vCPUs, AVX-512, `176 GiB` DDR5) CPU pods (`scripts/multimodal_cpu_offload_server.py`), keeping TPU host CPUs and Python threads dedicated to XLA/PJRT dispatch:

- `POST /encode` — Multi-threaded `ffmpeg` (`libx264` + `48 kHz` stereo `AAC`) video/audio MP4 muxing.
- `POST /encode_image` — Thread-pooled `Pillow` `PNG` / `WebP` / `JPEG` batch image encoding.
- `POST /encode_wav` — Vectorized `float32` / `int16` audio normalization + `RIFF/WAV` packet framing (`24 kHz` / `48 kHz`).
- `POST /preprocess_vision` — Multi-camera RGB image decode, bicubic resize, and `bfloat16`/`float32` normalization.

---

## 3. Hardware Platforms Evaluated

| Accelerator | Slice Topologies | HBM Capacity & Bandwidth | Host Memory (`/dev/shm`) |
|---|---|---|---|
| **Cloud TPU `v5p`** | `v5p-16` (`2x2x4`, `4` hosts × `4` chips) & `v5p-32` (`2x4x4`, `8` hosts × `4` chips) | `95 GiB` HBM2e / chip (`2,765 GB/s` / chip, `459` BF16 TFLOP/s) | `462 GiB` RAM (`380 GiB` tmpfs) |
| **Cloud TPU `v7x` (Ironwood)** | `v7x-4` (`2x2x1`, `8` TensorCores), `v7x-8` (`2x2x2`, `16` TensorCores), & `v7x-16` (`2x2x4`, `32` TensorCores) | `192 GiB` HBM3e / chip (`96 GiB` / TensorCore, `7,370 GB/s` / chip, `2,307` BF16 / `4,614` FP8 TFLOP/s per chip) | `944 GiB` RAM (`850 GiB` tmpfs) |

---

## 4. Benchmark Results (`v5p` & `v7x`)

Measured results linked 1:1 to raw JSON and Markdown summaries under [`results/`](./results/):

- **`v5p-16`** (`0` benchmark rows, updated `2026-11-02T19:10:04.446185+00:00`): [`results/v5p-16/SUMMARY.md`](./results/v5p-16/SUMMARY.md) · [`results/v5p-16/summary.json`](./results/v5p-16/summary.json)

# Qwen-Image (`20B` MMDiT) & Qwen-Image-2.1 / 2.1-Turbo (`Qwen3-VL`) — `v5p-16` (`v5p`)

- **Updated (UTC):** `2026-11-02T19:10:04.446185+00:00`
- **Hardware Topology:** `4` hosts x `4` TPU cores/host = `16` TPU cores
- **Models Covered:** `Qwen/Qwen-Image` (`Qwen2.5-VL` + 60-block MMDiT), `Qwen/Qwen-Image-2.1-Turbo` (`8`-step `Qwen3-VL`), `Qwen/Qwen-Image-2.1` (`28`-step `Qwen3-VL`)

## Disaggregated CPU PNG / WebP Image Encoding Ablation (`multimodal-cpu-offload-svc:8090`)

| Resolution | Offload Active | On-Host CPU PNG (ms) | Disagg AVX-512 PNG RTT (ms) | Disagg WebP RTT (ms) | PNG Offload Speedup |
|---|:---:|---:|---:|---:|---:|
| `512x512` | `True` | 38.87 | **41.62** | 41.67 | `0.93x` |
| `1024x1024` | `True` | 155.28 | **144.85** | 144.12 | `1.07x` |
| `1328x1328` | `True` | 261.07 | **247.68** | 239.1 | `1.05x` |
| `1664x928` | `True` | 228.64 | **212.28** | 208.14 | `1.08x` |
| `1536x1024` | `True` | 232.42 | **222.32** | 208.74 | `1.05x` |
| `2048x2048` | `True` | 615.21 | **583.98** | 554.44 | `1.05x` |

## End-to-End Diffusion Serving & TorchTPU Compilation Results

| Model | Engine | Prec | TP/Ulysses | Cluster DP | Task / Resolution | Steps | Conc | Stage-1 VL (ms) | MMDiT Step (ms) | VAE (ms) | E2E Local CPU (s) | E2E P50 (s) | Cluster img/min | Status |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|

- **`v7x-16`** (`3` benchmark rows, updated `2026-11-02T19:07:03.733583+00:00`): [`results/v7x-16/SUMMARY.md`](./results/v7x-16/SUMMARY.md) · [`results/v7x-16/summary.json`](./results/v7x-16/summary.json)

# Qwen-Image (`20B` MMDiT) & Qwen-Image-2.1 / 2.1-Turbo (`Qwen3-VL`) — `v7x-16` (`v7x`)

- **Updated (UTC):** `2026-11-02T19:07:03.733583+00:00`
- **Hardware Topology:** `4` hosts x `8` TPU cores/host = `32` TPU cores
- **Models Covered:** `Qwen/Qwen-Image` (`Qwen2.5-VL` + 60-block MMDiT), `Qwen/Qwen-Image-2.1-Turbo` (`8`-step `Qwen3-VL`), `Qwen/Qwen-Image-2.1` (`28`-step `Qwen3-VL`)

## Disaggregated CPU PNG / WebP Image Encoding Ablation (`multimodal-cpu-offload-svc:8090`)

| Resolution | Offload Active | On-Host CPU PNG (ms) | Disagg AVX-512 PNG RTT (ms) | Disagg WebP RTT (ms) | PNG Offload Speedup |
|---|:---:|---:|---:|---:|---:|
| `512x512` | `True` | 27.99 | **44.75** | 37.5 | `0.63x` |
| `1024x1024` | `True` | 118.09 | **144.6** | 143.75 | `0.82x` |
| `1328x1328` | `True` | 200.01 | **247.89** | 237.71 | `0.81x` |
| `1664x928` | `True` | 165.42 | **217.74** | 204.44 | `0.76x` |
| `1536x1024` | `True` | 189.17 | **217.5** | 209.24 | `0.87x` |
| `2048x2048` | `True` | 455.69 | **572.24** | 554.01 | `0.8x` |

## End-to-End Diffusion Serving & TorchTPU Compilation Results

| Model | Engine | Prec | TP/Ulysses | Cluster DP | Task / Resolution | Steps | Conc | Stage-1 VL (ms) | MMDiT Step (ms) | VAE (ms) | E2E Local CPU (s) | E2E P50 (s) | Cluster img/min | Status |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `8` | `T2I-OpenAI-HTTP-512x512` | `20` | `1` | - | - | - | - | **37.707** | **12.73** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `8` | `T2I-OpenAI-HTTP-1024x1024` | `20` | `1` | - | - | - | - | **43.041** | **11.14** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `8` | `T2I-OpenAI-HTTP-1024x1024` | `20` | `2` | - | - | - | - | **7.129** | **101.18** | `PASS` |

- **`v7x-spot-4c`** (`8` benchmark rows, updated `2026-11-02T19:03:48.814991+00:00`): [`results/v7x-spot-4c/SUMMARY.md`](./results/v7x-spot-4c/SUMMARY.md) · [`results/v7x-spot-4c/summary.json`](./results/v7x-spot-4c/summary.json)

# Qwen-Image (`20B` MMDiT) & Qwen-Image-2.1 / 2.1-Turbo (`Qwen3-VL`) — `v7x-spot-4c` (`v7x`)

- **Updated (UTC):** `2026-11-02T19:03:48.814991+00:00`
- **Hardware Topology:** `1` hosts x `8` TPU cores/host = `8` TPU cores
- **Models Covered:** `Qwen/Qwen-Image` (`Qwen2.5-VL` + 60-block MMDiT), `Qwen/Qwen-Image-2.1-Turbo` (`8`-step `Qwen3-VL`), `Qwen/Qwen-Image-2.1` (`28`-step `Qwen3-VL`)

## Disaggregated CPU PNG / WebP Image Encoding Ablation (`multimodal-cpu-offload-svc:8090`)

| Resolution | Offload Active | On-Host CPU PNG (ms) | Disagg AVX-512 PNG RTT (ms) | Disagg WebP RTT (ms) | PNG Offload Speedup |
|---|:---:|---:|---:|---:|---:|
| `512x512` | `True` | 32.28 | **76.49** | 139.15 | `0.42x` |
| `1024x1024` | `True` | 114.86 | **148.75** | 147.02 | `0.77x` |
| `1328x1328` | `True` | 201.42 | **248.87** | 238.45 | `0.81x` |
| `1664x928` | `True` | 165.59 | **217.95** | 210.2 | `0.76x` |
| `1536x1024` | `True` | 173.81 | **221.78** | 209.05 | `0.78x` |
| `2048x2048` | `True` | 455.06 | **573.35** | 557.77 | `0.79x` |

## End-to-End Diffusion Serving & TorchTPU Compilation Results

| Model | Engine | Prec | TP/Ulysses | Cluster DP | Task / Resolution | Steps | Conc | Stage-1 VL (ms) | MMDiT Step (ms) | VAE (ms) | E2E Local CPU (s) | E2E P50 (s) | Cluster img/min | Status |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `2` | `T2I-OpenAI-HTTP-512x512` | `20` | `1` | - | - | - | - | **37.344** | **3.21** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `2` | `T2I-OpenAI-HTTP-1024x1024` | `20` | `1` | - | - | - | - | **44.658** | **2.68** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `2` | `T2I-OpenAI-HTTP-1024x1024` | `20` | `2` | - | - | - | - | **7.364** | **24.46** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `2` | `T2I-OpenAI-HTTP-1024x1024` | `20` | `4` | - | - | - | - | **12.22** | **4.68** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `2` | `T2I-OpenAI-HTTP-1024x1024` | `50` | `1` | - | - | - | - | **11.683** | **10.24** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `2` | `T2I-OpenAI-HTTP-1328x1328` | `20` | `1` | - | - | - | - | **36.6** | **3.27** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `2` | `T2I-OpenAI-HTTP-1328x1328` | `20` | `2` | - | - | - | - | **8.912** | **19.85** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `2` | `T2I-OpenAI-HTTP-1328x1328` | `50` | `1` | - | - | - | - | **13.902** | **8.59** | `PASS` |

---

## 5. Repository Layout & Reproduction

- [`scripts/run_qwen_image_tpu_bench.py`](./scripts/run_qwen_image_tpu_bench.py) — Main TPU `v5p` / `v7x` benchmark driver (cooperative zero-disk HF-to-GCS staging, `/dev/shm` hydration, single-host PJRT isolation, `TorchTPU` / `vLLM` execution, and A/B CPU offload ablation).
- [`scripts/multimodal_cpu_offload_server.py`](./scripts/multimodal_cpu_offload_server.py) & [`scripts/cpu_offload_client.py`](./scripts/cpu_offload_client.py) — Disaggregated CPU offload HTTP server and client.
- [`scripts/gcs_stager.py`](./scripts/gcs_stager.py) & [`scripts/tpu_single_host_setup.py`](./scripts/tpu_single_host_setup.py) — Multi-host GCS shard stager and single-host `libtpu` metadata hook.
- [`k8s/`](./k8s/) — Kubernetes Deployment & Indexed Job manifests for CPU offload pool, TPU `v5p`, and TPU `v7x`.

```bash
# 1. Deploy the disaggregated CPU offload service
kubectl apply -f k8s/multimodal-cpu-offload-service.yaml

# 2. Launch the TPU v5p / v7x Indexed Job
kubectl apply -f k8s/v5p16-gr00t-and-qwen-image.yaml
```
