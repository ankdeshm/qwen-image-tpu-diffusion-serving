# `Qwen/Qwen-Image` (20B MMDiT) & `Qwen/Qwen-Image-2.1` (`Qwen3-VL`) Diffusion Serving on Cloud TPU `v5p` & `v7x` — Slice `v5p-16-wave4-adaptive64k`

> [!IMPORTANT]
> **Benchmark Scope & Measurement Methodology**
> - **Track A (`vllm-omni` HTTP) & Track B (`diffusers QwenImage21Pipeline`):** **100% Real End-to-End Checkpoint Inference** generating full images (`pixel_std ~ 75–96`). *Archived `Conc=1` `20`-step warmup note:* In sweeps recorded prior to the multi-prompt warmup fix in [`scripts/run_qwen_image_tpu_bench.py`](../../scripts/run_qwen_image_tpu_bench.py), warmup executed only `PROMPTS[0]`, so the 2nd sequential request in `Conc=1` (`PROMPTS[1]`, distinct `Qwen2.5-VL` prompt token count) triggered a one-time cold XLA compilation (`~85–120 s`), skewing `Conc=1` `20`-step `Median E2E` (`median([~9s, ~110s]) ≈ 37–66 s`), whereas subsequent `Conc=2` (`5.1–11.8 s`) and `50`-step (`8.7–18.3 s`) rows executed against warm XLA caches.
> - **Track C (`torchtpu-*` Multi-Stage Breakdown):** **Local-Shard Layer-Scaled Operator Microbenchmark** (`run_child_torchtpu_mmdit_worker` in `scripts/run_qwen_image_tpu_bench.py`) holding real `safetensors` checkpoint shards (`<=68 GiB/core`) in HBM while timing local-shard `Stage1QwenVLEncoder`, `6` measured `Stage2QwenDoubleStreamMMDiTBlock` blocks linearly scaled `× 10` (`-> 60` layers), and `Stage3QwenVAEDecoder` **excluding inter-chip TP/Ulysses collectives** (`AllReduce`/`AllToAll`). On `torchtpu-compiled`, first-shape rows (`20`-step / `8`-step / `28`-step) include worker-thread first-call `TorchDynamo` + XLA compilation in `VL Encode` (`~1.7–5.1 s`) / `VAE Decode` (`~2.4–8.8 s`), whereas subsequent `50`-step / warm-shape rows reflect steady-state execution (`2–23 ms`).
> - **Track D (`POST /encode_image` CPU Offload):** **100% Real End-to-End HTTP + AVX-512 Pillow `PNG`/`WebP` Encoding** against `multimodal-cpu-offload-svc:8090` (`c3-standard-44`). On `v5p-16` (`Wave-3/Wave-4 Keep-Alive`), remote AVX-512 achieves **`1.08x–1.14x` single-image latency speedup** (`545 ms` vs `614 ms` @ `2048x2048 WebP`). On `v7x` (`tpu7x-standard-4t`), the host's `224` Emerald Rapids vCPUs encode single images locally in `445 ms` vs `546–553 ms` HTTP RTT (`Local / Disagg RTT Ratio = 0.78x–0.87x`); on `v7x`, offload provides asynchronous host-CPU/GIL isolation under multi-stream load rather than single-image synchronous latency reduction.

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

| Resolution & Format | On-Host CPU (ms) | Disagg CPU RTT (ms) | Remote AVX-512 Compute (ms) | Remote Rate (img/s) | Local / Disagg RTT Ratio |
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
