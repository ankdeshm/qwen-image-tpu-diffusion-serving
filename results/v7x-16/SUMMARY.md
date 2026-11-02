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
