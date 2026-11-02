# `Qwen/Qwen-Image` (20B MMDiT) & `Qwen/Qwen-Image-2.1` (`Qwen3-VL`) Diffusion Serving on Cloud TPU `v5p` & `v7x` — Slice `v7x-8-wave3-keepalive`

> [!IMPORTANT]
> **Benchmark Scope & Measurement Methodology**
> - **Track A (`vllm-omni` HTTP) & Track B (`diffusers QwenImage21Pipeline`):** **100% Real End-to-End Checkpoint Inference** generating full images (`pixel_std ~ 75–96`). *Archived `Conc=1` `20`-step warmup note:* In sweeps recorded prior to the multi-prompt warmup fix in [`scripts/run_qwen_image_tpu_bench.py`](../../scripts/run_qwen_image_tpu_bench.py), warmup executed only `PROMPTS[0]`, so the 2nd sequential request in `Conc=1` (`PROMPTS[1]`, distinct `Qwen2.5-VL` prompt token count) triggered a one-time cold XLA compilation (`~85–120 s`), skewing `Conc=1` `20`-step `Median E2E` (`median([~9s, ~110s]) ≈ 37–66 s`), whereas subsequent `Conc=2` (`5.1–11.8 s`) and `50`-step (`8.7–18.3 s`) rows executed against warm XLA caches.
> - **Track C (`torchtpu-*` Multi-Stage Breakdown):** **Local-Shard Layer-Scaled Operator Microbenchmark** (`run_child_torchtpu_mmdit_worker` in `scripts/run_qwen_image_tpu_bench.py`) holding real `safetensors` checkpoint shards (`<=68 GiB/core`) in HBM while timing local-shard `Stage1QwenVLEncoder`, `6` measured `Stage2QwenDoubleStreamMMDiTBlock` blocks linearly scaled `× 10` (`-> 60` layers), and `Stage3QwenVAEDecoder` **excluding inter-chip TP/Ulysses collectives** (`AllReduce`/`AllToAll`). On `torchtpu-compiled`, first-shape rows (`20`-step / `8`-step / `28`-step) include worker-thread first-call `TorchDynamo` + XLA compilation in `VL Encode` (`~1.7–5.1 s`) / `VAE Decode` (`~2.4–8.8 s`), whereas subsequent `50`-step / warm-shape rows reflect steady-state execution (`2–23 ms`).
> - **Track D (`POST /encode_image` CPU Offload):** **100% Real End-to-End HTTP + AVX-512 Pillow `PNG`/`WebP` Encoding** against `multimodal-cpu-offload-svc:8090` (`c3-standard-44`). On `v5p-16` (`Wave-3/Wave-4 Keep-Alive`), remote AVX-512 achieves **`1.08x–1.14x` single-image latency speedup** (`545 ms` vs `614 ms` @ `2048x2048 WebP`). On `v7x` (`tpu7x-standard-4t`), the host's `224` Emerald Rapids vCPUs encode single images locally in `445 ms` vs `546–553 ms` HTTP RTT (`Local / Disagg RTT Ratio = 0.78x–0.87x`); on `v7x`, offload provides asynchronous host-CPU/GIL isolation under multi-stream load rather than single-image synchronous latency reduction.

### Slice `v7x-8-wave3-keepalive` (`v7x`, `1` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v7x-8-wave3-keepalive/SUMMARY.md`](./SUMMARY.md) · [`results/v7x-8-wave3-keepalive/summary.json`](./summary.json) *(updated `2026-11-02T21:22:11.533253+00:00`)*

#### Track A — OpenAI-Compatible `vllm-omni` HTTP Serving (`Qwen/Qwen-Image` 20B MMDiT)

| Engine | Precision | `Ulysses × Cluster DP` | Resolution | Steps | Conc | Warmup (s) | Median E2E (s) | P95 E2E (s) | Replica Img/min | Cluster Img/min |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `512x512` | `20` | `1` | `71.98` | **`37.683`** | `71.708` | `1.59` | **`6.37`** |

#### Track D — Disaggregated CPU Image Encoding (`PNG` & `WebP`) Ablation (`POST /encode_image`)

| Resolution & Format | On-Host CPU (ms) | Disagg CPU RTT (ms) | Remote AVX-512 Compute (ms) | Remote Rate (img/s) | Local / Disagg RTT Ratio |
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
