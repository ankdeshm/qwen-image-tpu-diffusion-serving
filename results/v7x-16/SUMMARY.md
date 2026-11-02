# `Qwen/Qwen-Image` (20B MMDiT) & `Qwen/Qwen-Image-2.1` (`Qwen3-VL`) Diffusion Serving on Cloud TPU `v5p` & `v7x` — Slice `v7x-16`

> [!IMPORTANT]
> **Benchmark Scope & Measurement Methodology**
> - **Track A (`vllm-omni` HTTP) & Track B (`diffusers QwenImage21Pipeline`):** **100% Real End-to-End Checkpoint Inference** generating full images (`pixel_std ~ 75–96`). *Archived `Conc=1` `20`-step warmup note:* In sweeps recorded prior to the multi-prompt warmup fix in [`scripts/run_qwen_image_tpu_bench.py`](../../scripts/run_qwen_image_tpu_bench.py), warmup executed only `PROMPTS[0]`, so the 2nd sequential request in `Conc=1` (`PROMPTS[1]`, distinct `Qwen2.5-VL` prompt token count) triggered a one-time cold XLA compilation (`~85–120 s`), skewing `Conc=1` `20`-step `Median E2E` (`median([~9s, ~110s]) ≈ 37–66 s`), whereas subsequent `Conc=2` (`5.1–11.8 s`) and `50`-step (`8.7–18.3 s`) rows executed against warm XLA caches.
> - **Track C (`torchtpu-*` Multi-Stage Breakdown):** **Local-Shard Layer-Scaled Operator Microbenchmark** (`run_child_torchtpu_mmdit_worker` in `scripts/run_qwen_image_tpu_bench.py`) holding real `safetensors` checkpoint shards (`<=68 GiB/core`) in HBM while timing local-shard `Stage1QwenVLEncoder`, `6` measured `Stage2QwenDoubleStreamMMDiTBlock` blocks linearly scaled `× 10` (`-> 60` layers), and `Stage3QwenVAEDecoder` **excluding inter-chip TP/Ulysses collectives** (`AllReduce`/`AllToAll`). On `torchtpu-compiled`, first-shape rows (`20`-step / `8`-step / `28`-step) include worker-thread first-call `TorchDynamo` + XLA compilation in `VL Encode` (`~1.7–5.1 s`) / `VAE Decode` (`~2.4–8.8 s`), whereas subsequent `50`-step / warm-shape rows reflect steady-state execution (`2–23 ms`).
> - **Track D (`POST /encode_image` CPU Offload):** **100% Real End-to-End HTTP + AVX-512 Pillow `PNG`/`WebP` Encoding** against `multimodal-cpu-offload-svc:8090` (`c3-standard-44`). On `v5p-16` (`Wave-3/Wave-4 Keep-Alive`), remote AVX-512 achieves **`1.08x–1.14x` single-image latency speedup** (`545 ms` vs `614 ms` @ `2048x2048 WebP`). On `v7x` (`tpu7x-standard-4t`), the host's `224` Emerald Rapids vCPUs encode single images locally in `445 ms` vs `546–553 ms` HTTP RTT (`Local / Disagg RTT Ratio = 0.78x–0.87x`); on `v7x`, offload provides asynchronous host-CPU/GIL isolation under multi-stream load rather than single-image synchronous latency reduction.

### Slice `v7x-16` (`v7x`, `88` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v7x-16/SUMMARY.md`](./SUMMARY.md) · [`results/v7x-16/summary.json`](./summary.json) *(updated `2026-11-02T20:43:46.998504+00:00`)*

#### Track A — OpenAI-Compatible `vllm-omni` HTTP Serving (`Qwen/Qwen-Image` 20B MMDiT)

| Engine | Precision | `Ulysses × Cluster DP` | Resolution | Steps | Conc | Warmup (s) | Median E2E (s) | P95 E2E (s) | Replica Img/min | Cluster Img/min |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=8` | `512x512` | `20` | `1` | `73.16` | **`38.717`** | `73.377` | `1.55` | **`12.39`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=8` | `1024x1024` | `20` | `1` | `106.75` | **`46.700`** | `88.654` | `1.28` | **`10.27`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=8` | `1024x1024` | `20` | `2` | `106.75` | **`7.553`** | `10.193` | `11.73` | **`93.84`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=8` | `1024x1024` | `20` | `4` | `106.75` | **`11.845`** | `105.826` | `2.27` | **`18.14`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=8` | `1024x1024` | `50` | `1` | `11.33` | **`11.294`** | `11.295` | `5.30` | **`42.37`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=8` | `1328x1328` | `20` | `1` | `96.26` | **`36.893`** | `68.091` | `1.62` | **`12.99`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=8` | `1328x1328` | `20` | `2` | `96.26` | **`8.906`** | `11.998` | `9.95` | **`79.58`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=8` | `1328x1328` | `50` | `1` | `13.61` | **`13.662`** | `13.702` | `4.37` | **`34.97`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=8` | `1664x928` | `20` | `1` | `120.83` | **`53.985`** | `102.726` | `1.11` | **`8.88`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=8` | `1664x928` | `20` | `2` | `120.83` | **`7.822`** | `10.365` | `11.51` | **`92.09`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=8` | `1664x928` | `50` | `1` | `12.64` | **`12.550`** | `12.623` | `4.76` | **`38.07`** |
| `vllm-omni-sdpa` | `BF16` | `U=4 × DP=8` | `1024x1024` | `20` | `1` | `9.02` | **`6.054`** | `8.737` | `9.85` | **`78.81`** |
| `vllm-omni-sdpa` | `BF16` | `U=4 × DP=8` | `1024x1024` | `20` | `2` | `9.02` | **`5.116`** | `6.843` | `17.44` | **`139.50`** |
| `vllm-omni-sdpa` | `BF16` | `U=4 × DP=8` | `1328x1328` | `20` | `1` | `8.01` | **`5.794`** | `7.716` | `10.24` | **`81.91`** |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=4` | `512x512` | `20` | `1` | `10.11` | **`6.859`** | `10.055` | `8.73` | **`34.93`** |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=4` | `1024x1024` | `20` | `1` | `46.13` | **`7.025`** | `10.426` | `8.49` | **`33.97`** |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=4` | `1024x1024` | `20` | `2` | `46.13` | **`5.989`** | `8.092` | `14.76` | **`59.05`** |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=4` | `1024x1024` | `20` | `4` | `46.13` | **`9.655`** | `25.832` | `9.28` | **`37.10`** |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=4` | `1024x1024` | `50` | `1` | `8.67` | **`8.716`** | `8.830` | `6.85` | **`27.41`** |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=4` | `1328x1328` | `20` | `1` | `62.53` | **`7.970`** | `12.040` | `7.47` | **`29.87`** |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=4` | `1328x1328` | `20` | `2` | `62.53` | **`6.044`** | `8.219` | `14.48` | **`57.93`** |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=4` | `1328x1328` | `50` | `1` | `9.23` | **`10.034`** | `10.227` | `5.94` | **`23.76`** |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=4` | `1664x928` | `20` | `1` | `54.78` | **`9.034`** | `14.213` | `6.60` | **`26.40`** |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=4` | `1664x928` | `20` | `2` | `54.78` | **`5.610`** | `7.414` | `16.06` | **`64.24`** |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=4` | `1664x928` | `50` | `1` | `9.28` | **`9.366`** | `9.468` | `6.37` | **`25.47`** |

#### Track B — `Qwen-Image-2.1` & `Qwen-Image-2.1-Turbo` (`Qwen3-VL` Text-Encoded Native Pipeline)

| Model Checkpoint | Engine | `TP × Cluster DP` | Resolution | Steps | Weight Load (s) | Median E2E (s) | Pixel Std | Replica Img/min | Cluster Img/min |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=32` | `1024x1024` | `8` | `10.84` | **`3.595`** | `90.06` | `16.69` | **`534.08`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=32` | `1536x1024` | `8` | `10.84` | **`5.378`** | `80.43` | `11.16` | **`357.12`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=32` | `2048x2048` | `8` | `10.84` | **`15.762`** | `86.05` | `3.81` | **`121.92`** |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=32` | `1024x1024` | `28` | `11.97` | **`11.919`** | `76.67` | `5.03` | **`160.96`** |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=32` | `1536x1024` | `28` | `11.97` | **`17.848`** | `86.87` | `3.36` | **`107.52`** |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=32` | `2048x2048` | `28` | `11.97` | **`52.948`** | `79.03` | `1.13` | **`36.16`** |

#### Track C — `TorchTPU` Multi-Stage Breakdown (`Qwen-Image` & `Qwen-Image-2.1`, `BF16` & `FP8` MXU)

| Model | Engine & Kernel | Task & Resolution | Steps | VL Encode (ms) | MMDiT Step (ms) | VAE Decode (ms) | Local PNG (ms) | Disagg PNG (ms) | E2E Disagg (s) | Cluster Img/min |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-50step` | `50` | `0.95` | `52.02` | `5.05` | `117.9` | **`149.9`** | **`2.757`** | **`174.16`** |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1328x1328-50step` | `50` | `1.00` | `61.45` | `30.50` | `189.8` | **`241.9`** | **`3.346`** | **`143.44`** |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1664x928-20step` | `20` | `1.07` | `50.73` | `8,784.34` | `184.1` | **`212.7`** | **`10.013`** | **`47.92`** |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1664x928-50step` | `50` | `0.93` | `55.38` | `75.39` | `184.1` | **`212.7`** | **`3.058`** | **`156.96`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-512x512-20step` | `20` | `1,689.32` | `20.07` | `5,576.75` | `28.1` | **`41.4`** | **`7.709`** | **`31.12`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-512x512-50step` | `50` | `2.39` | `18.25` | `6.73` | `28.1` | **`41.4`** | **`0.963`** | **`249.28`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-20step` | `20` | `2.35` | `52.28` | `6,303.53` | `117.9` | **`149.9`** | **`7.501`** | **`32.00`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-50step` | `50` | `2.54` | `50.16` | `13.56` | `117.9` | **`149.9`** | **`2.674`** | **`89.76`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1328x1328-20step` | `20` | `2.84` | `105.36` | `6,077.44` | `189.8` | **`241.9`** | **`8.429`** | **`28.48`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1328x1328-50step` | `50` | `2.77` | `102.27` | `23.77` | `189.8` | **`241.9`** | **`5.382`** | **`44.60`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1664x928-20step` | `20` | `2.51` | `89.02` | `8,377.05` | `184.1` | **`212.7`** | **`10.373`** | **`23.12`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1664x928-50step` | `50` | `2.81` | `89.87` | `20.71` | `184.1` | **`212.7`** | **`4.730`** | **`50.76`** |
| `Qwen/Qwen-Image` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-20step` | `20` | `12.42` | `78.21` | `15.79` | `117.9` | **`149.9`** | **`1.742`** | **`137.76`** |
| `Qwen/Qwen-Image` | `torchtpu-eager (BF16_MXU)` | `T2I-1328x1328-20step` | `20` | `10.82` | `138.65` | `26.56` | `189.8` | **`241.9`** | **`3.052`** | **`78.64`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-512x512-20step` | `20` | `2,182.76` | `19.68` | `5,203.71` | `28.1` | **`41.4`** | **`7.821`** | **`30.68`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-512x512-50step` | `50` | `1.68` | `18.52` | `6.69` | `28.1` | **`41.4`** | **`0.976`** | **`245.96`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-20step` | `20` | `1.78` | `52.10` | `5,922.07` | `117.9` | **`149.9`** | **`7.116`** | **`33.72`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-50step` | `50` | `2.21` | `51.85` | `14.01` | `117.9` | **`149.9`** | **`2.758`** | **`87.00`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1328x1328-20step` | `20` | `2.01` | `111.17` | `6,308.49` | `189.8` | **`241.9`** | **`8.776`** | **`27.36`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1328x1328-50step` | `50` | `1.81` | `107.09` | `23.05` | `189.8` | **`241.9`** | **`5.621`** | **`42.68`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1664x928-20step` | `20` | `2.41` | `88.91` | `8,178.09` | `184.1` | **`212.7`** | **`10.171`** | **`23.60`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1664x928-50step` | `50` | `2.05` | `89.07` | `20.58` | `184.1` | **`212.7`** | **`4.689`** | **`51.20`** |
| `Qwen/Qwen-Image` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-20step` | `20` | `17.60` | `70.45` | `15.86` | `117.9` | **`149.9`** | **`1.592`** | **`150.72`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-8step` | `8` | `20.06` | `25.78` | `0.82` | `117.9` | **`149.9`** | **`0.377`** | **`1,273.12`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (BF16_MXU)` | `T2I-1536x1024-8step` | `8` | `15.16` | `26.77` | `1.15` | `168.6` | **`218.9`** | **`0.449`** | **`1,068.00`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-8step` | `8` | `18.37` | `33.88` | `0.85` | `117.9` | **`149.9`** | **`0.440`** | **`1,090.72`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-8step` | `8` | `2,303.38` | `22.02` | `6,278.58` | `117.9` | **`149.9`** | **`8.908`** | **`26.96`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `T2I-1536x1024-8step` | `8` | `2.76` | `35.98` | `5,963.76` | `168.6` | **`218.9`** | **`6.473`** | **`37.08`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `T2I-2048x2048-8step` | `8` | `2.75` | `103.93` | `6,931.75` | `447.2` | **`571.8`** | **`8.338`** | **`28.80`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1024x1024-8step` | `8` | `2,250.84` | `28.75` | `13.89` | `117.9` | **`149.9`** | **`2.678`** | **`89.64`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1536x1024-8step` | `8` | `7.02` | `54.90` | `19.77` | `168.6` | **`218.9`** | **`0.718`** | **`334.32`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-8step` | `8` | `15.86` | `30.61` | `15.97` | `117.9` | **`149.9`** | **`0.427`** | **`562.64`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (BF16_MXU)` | `T2I-1536x1024-8step` | `8` | `13.70` | `47.10` | `22.85` | `168.6` | **`218.9`** | **`0.632`** | **`379.60`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-8step` | `8` | `2,840.03` | `18.37` | `6,222.83` | `117.9` | **`149.9`** | **`9.360`** | **`25.64`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1536x1024-8step` | `8` | `2.95` | `29.95` | `6,103.55` | `168.6` | **`218.9`** | **`6.565`** | **`36.56`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-2048x2048-8step` | `8` | `3.14` | `90.95` | `6,983.46` | `447.2` | **`571.8`** | **`8.286`** | **`28.96`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1024x1024-8step` | `8` | `2,939.14` | `25.77` | `14.42` | `117.9` | **`149.9`** | **`3.342`** | **`71.80`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1536x1024-8step` | `8` | `5.05` | `53.00` | `19.39` | `168.6` | **`218.9`** | **`0.700`** | **`342.72`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-8step` | `8` | `20.23` | `36.48` | `16.19` | `117.9` | **`149.9`** | **`0.478`** | **`501.96`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `1.18` | `25.76` | `127.94` | `168.6` | **`218.9`** | **`1.102`** | **`435.52`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-28step` | `28` | `16.33` | `26.05` | `1.31` | `117.9` | **`149.9`** | **`0.897`** | **`535.20`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (BF16_MXU)` | `T2I-1536x1024-28step` | `28` | `14.77` | `29.27` | `1.31` | `168.6` | **`218.9`** | **`1.055`** | **`455.20`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `0.84` | `33.85` | `109.79` | `168.6` | **`218.9`** | **`1.310`** | **`366.32`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-28step` | `28` | `21.10` | `45.33` | `0.96` | `117.9` | **`149.9`** | **`1.441`** | **`333.12`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-28step` | `28` | `2,353.85` | `39.93` | `6,065.42` | `117.9` | **`149.9`** | **`9.687`** | **`24.76`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `T2I-1536x1024-28step` | `28` | `3.18` | `72.99` | `5,795.39` | `168.6` | **`218.9`** | **`8.061`** | **`29.76`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `T2I-2048x2048-28step` | `28` | `2.99` | `204.14` | `6,835.34` | `447.2` | **`571.8`** | **`13.126`** | **`18.28`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1024x1024-28step` | `28` | `2,293.80` | `54.65` | `13.36` | `117.9` | **`149.9`** | **`4.020`** | **`59.68`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `6.94` | `100.65` | `19.76` | `168.6` | **`218.9`** | **`3.097`** | **`77.52`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-28step` | `28` | `16.29` | `54.99` | `15.61` | `117.9` | **`149.9`** | **`1.722`** | **`139.40`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (BF16_MXU)` | `T2I-1536x1024-28step` | `28` | `14.62` | `84.19` | `23.12` | `168.6` | **`218.9`** | **`2.614`** | **`91.80`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-28step` | `28` | `2,882.26` | `37.53` | `5,680.62` | `117.9` | **`149.9`** | **`9.764`** | **`24.60`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1536x1024-28step` | `28` | `2.50` | `66.02` | `6,186.25` | `168.6` | **`218.9`** | **`8.256`** | **`29.08`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-2048x2048-28step` | `28` | `2.82` | `196.31` | `6,689.88` | `447.2` | **`571.8`** | **`12.761`** | **`18.80`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1024x1024-28step` | `28` | `3,034.54` | `50.91` | `13.72` | `117.9` | **`149.9`** | **`4.656`** | **`51.56`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `5.23` | `92.24` | `19.66` | `168.6` | **`218.9`** | **`2.859`** | **`83.92`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-28step` | `28` | `23.91` | `48.10` | `16.20` | `117.9` | **`149.9`** | **`1.537`** | **`156.16`** |

#### Track D — Disaggregated CPU Image Encoding (`PNG` & `WebP`) Ablation (`POST /encode_image`)

| Resolution & Format | On-Host CPU (ms) | Disagg CPU RTT (ms) | Remote AVX-512 Compute (ms) | Remote Rate (img/s) | Local / Disagg RTT Ratio |
|---|---:|---:|---:|---:|---:|
| `512x512_png` | `28.13` | **`41.35`** | `35.16` | `28.4` | **`0.68x`** |
| `512x512_webp` | `28.35` | **`41.69`** | `34.24` | `29.2` | **`0.68x`** |
| `1024x1024_png` | `117.87` | **`149.86`** | `140.70` | `7.1` | **`0.79x`** |
| `1024x1024_webp` | `113.33` | **`140.18`** | `135.48` | `7.4` | **`0.81x`** |
| `1328x1328_png` | `189.75` | **`241.87`** | `237.13` | `4.2` | **`0.78x`** |
| `1328x1328_webp` | `199.65` | **`239.25`** | `228.17` | `4.4` | **`0.83x`** |
| `1664x928_png` | `184.07` | **`212.65`** | `207.56` | `4.8` | **`0.87x`** |
| `1664x928_webp` | `167.63` | **`203.20`** | `198.84` | `5.0` | **`0.82x`** |
| `1536x1024_png` | `168.57` | **`218.94`** | `211.17` | `4.7` | **`0.77x`** |
| `1536x1024_webp` | `168.93` | **`212.11`** | `202.61` | `4.9` | **`0.80x`** |
| `2048x2048_png` | `447.21` | **`571.82`** | `563.83` | `1.8` | **`0.78x`** |
| `2048x2048_webp` | `445.36` | **`553.49`** | `539.54` | `1.9` | **`0.80x`** |
