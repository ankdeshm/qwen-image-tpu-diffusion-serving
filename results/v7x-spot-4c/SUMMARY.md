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
