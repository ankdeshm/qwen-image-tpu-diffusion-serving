# `Qwen/Qwen-Image` (20B MMDiT) & `Qwen/Qwen-Image-2.1` (`Qwen3-VL`) Diffusion Serving on Cloud TPU `v5p` & `v7x` — Slice `v5p-16-wave3-keepalive`

> [!IMPORTANT]
> **Benchmark Scope & Measurement Methodology**
> - **Track A (`vllm-omni` HTTP) & Track B (`diffusers QwenImage21Pipeline`):** **100% Real End-to-End Checkpoint Inference** generating full images (`pixel_std ~ 75–96`). *Archived `Conc=1` `20`-step warmup note:* In sweeps recorded prior to the multi-prompt warmup fix in [`scripts/run_qwen_image_tpu_bench.py`](../../scripts/run_qwen_image_tpu_bench.py), warmup executed only `PROMPTS[0]`, so the 2nd sequential request in `Conc=1` (`PROMPTS[1]`, distinct `Qwen2.5-VL` prompt token count) triggered a one-time cold XLA compilation (`~85–120 s`), skewing `Conc=1` `20`-step `Median E2E` (`median([~9s, ~110s]) ≈ 37–66 s`), whereas subsequent `Conc=2` (`5.1–11.8 s`) and `50`-step (`8.7–18.3 s`) rows executed against warm XLA caches.
> - **Track C (`torchtpu-*` Multi-Stage Breakdown):** **Local-Shard Layer-Scaled Operator Microbenchmark** (`run_child_torchtpu_mmdit_worker` in `scripts/run_qwen_image_tpu_bench.py`) holding real `safetensors` checkpoint shards (`<=68 GiB/core`) in HBM while timing local-shard `Stage1QwenVLEncoder`, `6` measured `Stage2QwenDoubleStreamMMDiTBlock` blocks linearly scaled `× 10` (`-> 60` layers), and `Stage3QwenVAEDecoder` **excluding inter-chip TP/Ulysses collectives** (`AllReduce`/`AllToAll`). On `torchtpu-compiled`, first-shape rows (`20`-step / `8`-step / `28`-step) include worker-thread first-call `TorchDynamo` + XLA compilation in `VL Encode` (`~1.7–5.1 s`) / `VAE Decode` (`~2.4–8.8 s`), whereas subsequent `50`-step / warm-shape rows reflect steady-state execution (`2–23 ms`).
> - **Track D (`POST /encode_image` CPU Offload):** **100% Real End-to-End HTTP + AVX-512 Pillow `PNG`/`WebP` Encoding** against `multimodal-cpu-offload-svc:8090` (`c3-standard-44`). On `v5p-16` (`Wave-3/Wave-4 Keep-Alive`), remote AVX-512 achieves **`1.08x–1.14x` single-image latency speedup** (`545 ms` vs `614 ms` @ `2048x2048 WebP`). On `v7x` (`tpu7x-standard-4t`), the host's `224` Emerald Rapids vCPUs encode single images locally in `445 ms` vs `546–553 ms` HTTP RTT (`Local / Disagg RTT Ratio = 0.78x–0.87x`); on `v7x`, offload provides asynchronous host-CPU/GIL isolation under multi-stream load rather than single-image synchronous latency reduction.

### Slice `v5p-16-wave3-keepalive` (`v5p`, `65` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v5p-16-wave3-keepalive/SUMMARY.md`](./SUMMARY.md) · [`results/v5p-16-wave3-keepalive/summary.json`](./summary.json) *(updated `2026-11-02T20:54:39.543913+00:00`)*

#### Track A — OpenAI-Compatible `vllm-omni` HTTP Serving (`Qwen/Qwen-Image` 20B MMDiT)

| Engine | Precision | `Ulysses × Cluster DP` | Resolution | Steps | Conc | Warmup (s) | Median E2E (s) | P95 E2E (s) | Replica Img/min | Cluster Img/min |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `512x512` | `20` | `1` | `116.41` | **`60.345`** | `115.844` | `0.99` | **`3.98`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `1` | `117.08` | **`61.504`** | `116.994` | `0.97` | **`3.90`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `2` | `117.08` | **`9.258`** | `12.496` | `9.57` | **`38.27`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `4` | `117.08` | **`16.073`** | `144.873` | `1.66` | **`6.62`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1024x1024` | `50` | `1` | `14.64` | **`14.571`** | `14.576` | `4.11` | **`16.42`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1328x1328` | `20` | `1` | `88.12` | **`46.520`** | `85.525` | `1.29` | **`5.15`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1328x1328` | `20` | `2` | `88.12` | **`11.399`** | `15.338` | `7.79` | **`31.15`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1328x1328` | `50` | `1` | `17.98` | **`18.172`** | `18.226` | `3.29` | **`13.15`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1664x928` | `20` | `1` | `127.20` | **`64.846`** | `122.668` | `0.92` | **`3.70`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1664x928` | `20` | `2` | `127.20` | **`10.331`** | `13.708` | `8.71` | **`34.85`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1664x928` | `50` | `1` | `16.79` | **`16.992`** | `17.099` | `3.52` | **`14.07`** |
| `vllm-omni-sdpa` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `1` | `9.21` | **`6.621`** | `9.148` | `9.00` | **`36.01`** |
| `vllm-omni-sdpa` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `2` | `9.21` | **`5.830`** | `7.891` | `15.12` | **`60.50`** |
| `vllm-omni-sdpa` | `BF16` | `U=4 × DP=4` | `1328x1328` | `20` | `1` | `7.29` | **`5.623`** | `7.034` | `10.52` | **`42.08`** |

#### Track B — `Qwen-Image-2.1` & `Qwen-Image-2.1-Turbo` (`Qwen3-VL` Text-Encoded Native Pipeline)

| Model Checkpoint | Engine | `TP × Cluster DP` | Resolution | Steps | Weight Load (s) | Median E2E (s) | Pixel Std | Replica Img/min | Cluster Img/min |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `1024x1024` | `8` | `11.94` | **`3.299`** | `82.45` | `18.19` | **`291.04`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `1536x1024` | `8` | `11.94` | **`4.951`** | `74.66` | `12.12` | **`193.92`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `2048x2048` | `8` | `11.94` | **`14.414`** | `96.06` | `4.16` | **`66.56`** |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `1024x1024` | `28` | `11.11` | **`10.711`** | `77.99` | `5.60` | **`89.60`** |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `1536x1024` | `28` | `11.11` | **`16.180`** | `76.08` | `3.71` | **`59.36`** |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `2048x2048` | `28` | `11.11` | **`47.894`** | `83.47` | `1.25` | **`20.00`** |

#### Track C — `TorchTPU` Multi-Stage Breakdown (`Qwen-Image` & `Qwen-Image-2.1`, `BF16` & `FP8` MXU)

| Model | Engine & Kernel | Task & Resolution | Steps | VL Encode (ms) | MMDiT Step (ms) | VAE Decode (ms) | Local PNG (ms) | Disagg PNG (ms) | E2E Disagg (s) | Cluster Img/min |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-512x512-20step` | `20` | `2,455.61` | `44.79` | `2,567.28` | `39.0` | **`39.0`** | **`5.958`** | **`40.28`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-512x512-50step` | `50` | `5.22` | `44.00` | `2.95` | `39.0` | **`39.0`** | **`2.247`** | **`106.80`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-20step` | `20` | `5.24` | `123.18` | `2,779.49` | `155.8` | **`157.8`** | **`5.406`** | **`44.40`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-50step` | `50` | `5.24` | `121.70` | `6.08` | `155.8` | **`157.8`** | **`6.254`** | **`38.36`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1328x1328-20step` | `20` | `5.26` | `226.71` | `2,801.85` | `262.5` | **`241.5`** | **`7.583`** | **`31.64`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1328x1328-50step` | `50` | `5.20` | `226.38` | `10.63` | `262.5` | **`241.5`** | **`11.576`** | **`20.72`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1664x928-20step` | `20` | `5.45` | `195.06` | `3,041.97` | `229.8` | **`212.1`** | **`7.161`** | **`33.52`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1664x928-50step` | `50` | `5.18` | `193.12` | `8.64` | `229.8` | **`212.1`** | **`9.882`** | **`24.28`** |
| `Qwen/Qwen-Image` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-20step` | `20` | `15.87` | `165.14` | `6.33` | `155.8` | **`157.8`** | **`3.483`** | **`68.92`** |
| `Qwen/Qwen-Image` | `torchtpu-eager (BF16_MXU)` | `T2I-1328x1328-20step` | `20` | `16.39` | `287.65` | `10.70` | `262.5` | **`241.5`** | **`6.022`** | **`39.84`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-512x512-20step` | `20` | `3,637.35` | `116.99` | `2,415.00` | `39.0` | **`39.0`** | **`8.431`** | **`28.48`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-512x512-50step` | `50` | `16.88` | `115.88` | `2.82` | `39.0` | **`39.0`** | **`5.853`** | **`41.00`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-20step` | `20` | `16.96` | `317.91` | `2,670.12` | `155.8` | **`157.8`** | **`9.203`** | **`26.08`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-50step` | `50` | `16.91` | `316.76` | `6.03` | `155.8` | **`157.8`** | **`16.019`** | **`15.00`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1328x1328-20step` | `20` | `16.96` | `553.83` | `2,862.03` | `262.5` | **`241.5`** | **`14.197`** | **`16.92`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1328x1328-50step` | `50` | `16.84` | `554.31` | `10.81` | `262.5` | **`241.5`** | **`27.985`** | **`8.56`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1664x928-20step` | `20` | `17.02` | `406.93` | `3,029.72` | `229.8` | **`212.1`** | **`11.397`** | **`21.04`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1664x928-50step` | `50` | `16.92` | `405.81` | `8.76` | `229.8` | **`212.1`** | **`20.528`** | **`11.68`** |
| `Qwen/Qwen-Image` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-20step` | `20` | `21.71` | `403.32` | `6.22` | `155.8` | **`157.8`** | **`8.252`** | **`29.08`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-8step` | `8` | `3,331.14` | `57.06` | `2,781.34` | `155.8` | **`157.8`** | **`6.727`** | **`35.68`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `T2I-1536x1024-8step` | `8` | `8.37` | `86.10` | `2,716.55` | `233.7` | **`215.2`** | **`3.629`** | **`66.12`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `T2I-2048x2048-8step` | `8` | `8.47` | `248.48` | `3,447.70` | `617.0` | **`572.1`** | **`6.016`** | **`39.88`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1024x1024-8step` | `8` | `3,424.71` | `72.11` | `6.12` | `155.8` | **`157.8`** | **`4.191`** | **`57.28`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1536x1024-8step` | `8` | `21.30` | `129.77` | `8.73` | `233.7` | **`215.2`** | **`1.309`** | **`183.36`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-8step` | `8` | `24.72` | `69.72` | `6.28` | `155.8` | **`157.8`** | **`0.747`** | **`321.44`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (BF16_MXU)` | `T2I-1536x1024-8step` | `8` | `18.49` | `108.63` | `8.72` | `233.7` | **`215.2`** | **`1.111`** | **`215.96`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-8step` | `8` | `4,590.22` | `146.95` | `2,814.72` | `155.8` | **`157.8`** | **`8.738`** | **`27.48`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1536x1024-8step` | `8` | `22.93` | `191.27` | `2,881.84` | `233.7` | **`215.2`** | **`4.650`** | **`51.60`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-2048x2048-8step` | `8` | `22.79` | `536.67` | `3,372.77` | `617.0` | **`572.1`** | **`8.261`** | **`29.04`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1024x1024-8step` | `8` | `4,893.07` | `185.96` | `6.37` | `155.8` | **`157.8`** | **`6.571`** | **`36.52`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1536x1024-8step` | `8` | `46.95` | `299.16` | `8.87` | `233.7` | **`215.2`** | **`2.690`** | **`89.24`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-8step` | `8` | `36.17` | `190.09` | `6.13` | `155.8` | **`157.8`** | **`1.721`** | **`139.48`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-28step` | `28` | `3,268.52` | `105.01` | `2,653.59` | `155.8` | **`157.8`** | **`9.020`** | **`26.60`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `T2I-1536x1024-28step` | `28` | `8.36` | `164.97` | `2,794.61` | `233.7` | **`215.2`** | **`7.637`** | **`31.44`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `T2I-2048x2048-28step` | `28` | `8.36` | `499.12` | `3,244.19` | `617.0` | **`572.1`** | **`17.800`** | **`13.48`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1024x1024-28step` | `28` | `3,400.43` | `136.51` | `6.02` | `155.8` | **`157.8`** | **`7.412`** | **`32.36`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `21.23` | `244.63` | `8.74` | `233.7` | **`215.2`** | **`7.120`** | **`33.72`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-28step` | `28` | `26.42` | `134.41` | `6.27` | `155.8` | **`157.8`** | **`3.954`** | **`60.68`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (BF16_MXU)` | `T2I-1536x1024-28step` | `28` | `25.35` | `205.55` | `8.68` | `233.7` | **`215.2`** | **`6.005`** | **`39.96`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-28step` | `28` | `4,564.53` | `314.78` | `2,781.37` | `155.8` | **`157.8`** | **`16.318`** | **`14.72`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1536x1024-28step` | `28` | `23.00` | `465.72` | `2,857.56` | `233.7` | **`215.2`** | **`16.136`** | **`14.88`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-2048x2048-28step` | `28` | `22.87` | `1,269.66` | `3,448.97` | `617.0` | **`572.1`** | **`39.594`** | **`6.08`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1024x1024-28step` | `28` | `4,684.67` | `402.74` | `6.20` | `155.8` | **`157.8`** | **`16.151`** | **`14.84`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `46.84` | `569.98` | `8.85` | `233.7` | **`215.2`** | **`16.256`** | **`14.76`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-28step` | `28` | `33.23` | `358.46` | `6.30` | `155.8` | **`157.8`** | **`10.234`** | **`23.44`** |

#### Track D — Disaggregated CPU Image Encoding (`PNG` & `WebP`) Ablation (`POST /encode_image`)

| Resolution & Format | On-Host CPU (ms) | Disagg CPU RTT (ms) | Remote AVX-512 Compute (ms) | Remote Rate (img/s) | Local / Disagg RTT Ratio |
|---|---:|---:|---:|---:|---:|
| `512x512_png` | `39.05` | **`38.97`** | `36.68` | `27.3` | **`1.00x`** |
| `512x512_webp` | `39.06` | **`36.27`** | `34.13` | `29.3` | **`1.08x`** |
| `1024x1024_png` | `155.83` | **`157.85`** | `140.10` | `7.1` | **`0.99x`** |
| `1024x1024_webp` | `155.57` | **`138.47`** | `135.60` | `7.4` | **`1.12x`** |
| `1328x1328_png` | `262.52` | **`241.47`** | `236.85` | `4.2` | **`1.09x`** |
| `1328x1328_webp` | `262.71` | **`233.05`** | `228.49` | `4.4` | **`1.13x`** |
| `1664x928_png` | `229.78` | **`212.06`** | `207.64` | `4.8` | **`1.08x`** |
| `1664x928_webp` | `229.50` | **`203.42`** | `199.08` | `5.0` | **`1.13x`** |
| `1536x1024_png` | `233.67` | **`215.15`** | `210.83` | `4.7` | **`1.09x`** |
| `1536x1024_webp` | `233.77` | **`208.15`** | `203.68` | `4.9` | **`1.12x`** |
| `2048x2048_png` | `617.05` | **`572.14`** | `564.25` | `1.8` | **`1.08x`** |
| `2048x2048_webp` | `618.71` | **`549.99`** | `541.30` | `1.8` | **`1.12x`** |
