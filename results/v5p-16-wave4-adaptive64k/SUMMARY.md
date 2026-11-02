# `Qwen/Qwen-Image` (20B MMDiT) & `Qwen/Qwen-Image-2.1` (`Qwen3-VL`) Diffusion Serving on Cloud TPU `v5p` & `v7x` — Slice `v5p-16-wave4-adaptive64k`

### Slice `v5p-16-wave4-adaptive64k` (`v5p`, `5` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v5p-16-wave4-adaptive64k/SUMMARY.md`](./SUMMARY.md) · [`results/v5p-16-wave4-adaptive64k/summary.json`](./summary.json) *(updated `2026-11-02T21:22:16.697207+00:00`)*

#### Track A — OpenAI-Compatible `vllm-omni` HTTP Serving (`Qwen/Qwen-Image` 20B MMDiT)

| Engine | Precision | `Ulysses × Cluster DP` | Resolution | Steps | Conc | Warmup (s) | Median E2E (s) | P95 E2E (s) | Replica Img/min | Cluster Img/min |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `512x512` | `20` | `1` | `117.22` | **`62.261`** | `119.606` | `0.96` | **`3.85`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `1` | `118.32` | **`63.087`** | `120.096` | `0.95` | **`3.80`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `2` | `118.32` | **`9.367`** | `12.653` | `9.45` | **`37.81`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `4` | `118.32` | **`75.161`** | `144.301` | `1.66` | **`6.65`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1024x1024` | `50` | `1` | `14.68` | **`14.769`** | `14.800` | `4.05` | **`16.20`** |

#### Track D — Disaggregated CPU Image Encoding (`PNG` & `WebP`) Ablation (`POST /encode_image`)

| Resolution & Format | On-Host CPU (ms) | Disagg CPU RTT (ms) | Remote AVX-512 Compute (ms) | Remote Rate (img/s) | Speedup vs On-Host |
|---|---:|---:|---:|---:|---:|
| `512x512_png` | `38.67` | **`35.78`** | `34.96` | `28.6` | **`1.08x`** |
| `512x512_webp` | `38.47` | **`34.66`** | `33.90` | `29.5` | **`1.11x`** |
| `1024x1024_png` | `155.83` | **`140.96`** | `139.76` | `7.2` | **`1.11x`** |
| `1024x1024_webp` | `154.94` | **`136.47`** | `135.17` | `7.4` | **`1.14x`** |
| `1328x1328_png` | `261.05` | **`238.98`** | `236.55` | `4.2` | **`1.09x`** |
| `1328x1328_webp` | `260.52` | **`229.98`** | `227.72` | `4.4` | **`1.13x`** |
| `1664x928_png` | `227.90` | **`209.06`** | `206.99` | `4.8` | **`1.09x`** |
| `1664x928_webp` | `227.88` | **`201.15`** | `199.02` | `5.0` | **`1.13x`** |
| `1536x1024_png` | `231.73` | **`212.40`** | `210.42` | `4.8` | **`1.09x`** |
| `1536x1024_webp` | `232.21` | **`205.08`** | `203.10` | `4.9` | **`1.13x`** |
| `2048x2048_png` | `625.72` | **`567.54`** | `563.06` | `1.8` | **`1.10x`** |
| `2048x2048_webp` | `613.55` | **`545.01`** | `540.20` | `1.9` | **`1.13x`** |
