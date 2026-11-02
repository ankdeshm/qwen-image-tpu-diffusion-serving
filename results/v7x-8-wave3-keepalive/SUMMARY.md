# `Qwen/Qwen-Image` (20B MMDiT) & `Qwen/Qwen-Image-2.1` (`Qwen3-VL`) Diffusion Serving on Cloud TPU `v5p` & `v7x` — Slice `v7x-8-wave3-keepalive`

### Slice `v7x-8-wave3-keepalive` (`v7x`, `1` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v7x-8-wave3-keepalive/SUMMARY.md`](./SUMMARY.md) · [`results/v7x-8-wave3-keepalive/summary.json`](./summary.json) *(updated `2026-11-02T21:22:11.533253+00:00`)*

#### Track A — OpenAI-Compatible `vllm-omni` HTTP Serving (`Qwen/Qwen-Image` 20B MMDiT)

| Engine | Precision | `Ulysses × Cluster DP` | Resolution | Steps | Conc | Warmup (s) | Median E2E (s) | P95 E2E (s) | Replica Img/min | Cluster Img/min |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `512x512` | `20` | `1` | `71.98` | **`37.683`** | `71.708` | `1.59` | **`6.37`** |

#### Track D — Disaggregated CPU Image Encoding (`PNG` & `WebP`) Ablation (`POST /encode_image`)

| Resolution & Format | On-Host CPU (ms) | Disagg CPU RTT (ms) | Remote AVX-512 Compute (ms) | Remote Rate (img/s) | Speedup vs On-Host |
|---|---:|---:|---:|---:|---:|
| `512x512_png` | `28.01` | **`36.05`** | `34.86` | `28.7` | **`0.78x`** |
| `512x512_webp` | `27.87` | **`34.93`** | `33.92` | `29.5` | **`0.80x`** |
| `1024x1024_png` | `123.63` | **`142.05`** | `140.39` | `7.1` | **`0.87x`** |
| `1024x1024_webp` | `114.16` | **`136.30`** | `134.95` | `7.4` | **`0.84x`** |
| `1328x1328_png` | `198.73` | **`238.87`** | `236.41` | `4.2` | **`0.83x`** |
| `1328x1328_webp` | `189.44` | **`231.07`** | `228.15` | `4.4` | **`0.82x`** |
| `1664x928_png` | `165.84` | **`209.23`** | `207.01` | `4.8` | **`0.79x`** |
| `1664x928_webp` | `169.34` | **`201.26`** | `198.97` | `5.0` | **`0.84x`** |
| `1536x1024_png` | `178.49` | **`212.62`** | `210.67` | `4.7` | **`0.84x`** |
| `1536x1024_webp` | `168.74` | **`205.13`** | `202.72` | `4.9` | **`0.82x`** |
| `2048x2048_png` | `452.78` | **`567.80`** | `562.76` | `1.8` | **`0.80x`** |
| `2048x2048_webp` | `450.75` | **`545.52`** | `540.74` | `1.8` | **`0.83x`** |
