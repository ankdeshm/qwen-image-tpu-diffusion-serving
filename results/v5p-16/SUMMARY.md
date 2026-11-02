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
