# `Qwen/Qwen-Image` (20B MMDiT) & `Qwen/Qwen-Image-2.1` (`Qwen3-VL`) Diffusion Serving on Cloud TPU `v5p` & `v7x` — Slice `v7x-spot-4c`

> [!IMPORTANT]
> **Benchmark Scope & Measurement Methodology**
> - **Track A (`vllm-omni` HTTP) & Track B (`diffusers QwenImage21Pipeline`):** **100% Real End-to-End Checkpoint Inference** generating full images (`pixel_std ~ 75–96`). *Archived `Conc=1` `20`-step warmup note:* In sweeps recorded prior to the multi-prompt warmup fix in [`scripts/run_qwen_image_tpu_bench.py`](../../scripts/run_qwen_image_tpu_bench.py), warmup executed only `PROMPTS[0]`, so the 2nd sequential request in `Conc=1` (`PROMPTS[1]`, distinct `Qwen2.5-VL` prompt token count) triggered a one-time cold XLA compilation (`~85–120 s`), skewing `Conc=1` `20`-step `Median E2E` (`median([~9s, ~110s]) ≈ 37–66 s`), whereas subsequent `Conc=2` (`5.1–11.8 s`) and `50`-step (`8.7–18.3 s`) rows executed against warm XLA caches.
> - **Track C (`torchtpu-*` Multi-Stage Breakdown):** **Local-Shard Layer-Scaled Operator Microbenchmark** (`run_child_torchtpu_mmdit_worker` in `scripts/run_qwen_image_tpu_bench.py`) holding real `safetensors` checkpoint shards (`<=68 GiB/core`) in HBM while timing local-shard `Stage1QwenVLEncoder`, `6` measured `Stage2QwenDoubleStreamMMDiTBlock` blocks linearly scaled `× 10` (`-> 60` layers), and `Stage3QwenVAEDecoder` **excluding inter-chip TP/Ulysses collectives** (`AllReduce`/`AllToAll`). On `torchtpu-compiled`, first-shape rows (`20`-step / `8`-step / `28`-step) include worker-thread first-call `TorchDynamo` + XLA compilation in `VL Encode` (`~1.7–5.1 s`) / `VAE Decode` (`~2.4–8.8 s`), whereas subsequent `50`-step / warm-shape rows reflect steady-state execution (`2–23 ms`).
> - **Track D (`POST /encode_image` CPU Offload):** **100% Real End-to-End HTTP + AVX-512 Pillow `PNG`/`WebP` Encoding** against `multimodal-cpu-offload-svc:8090` (`c3-standard-44`). On `v5p-16` (`Wave-3/Wave-4 Keep-Alive`), remote AVX-512 achieves **`1.08x–1.14x` single-image latency speedup** (`545 ms` vs `614 ms` @ `2048x2048 WebP`). On `v7x` (`tpu7x-standard-4t`), the host's `224` Emerald Rapids vCPUs encode single images locally in `445 ms` vs `546–553 ms` HTTP RTT (`Local / Disagg RTT Ratio = 0.78x–0.87x`); on `v7x`, offload provides asynchronous host-CPU/GIL isolation under multi-stream load rather than single-image synchronous latency reduction.

### Slice `v7x-spot-4c` (`v7x`, `87` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v7x-spot-4c/SUMMARY.md`](./SUMMARY.md) · [`results/v7x-spot-4c/summary.json`](./summary.json) *(updated `2026-11-02T20:32:54.561312+00:00`)*

#### Track A — OpenAI-Compatible `vllm-omni` HTTP Serving (`Qwen/Qwen-Image` 20B MMDiT)

| Engine | Precision | `Ulysses × Cluster DP` | Resolution | Steps | Conc | Warmup (s) | Median E2E (s) | P95 E2E (s) | Replica Img/min | Cluster Img/min |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=2` | `512x512` | `20` | `1` | `72.39` | **`38.878`** | `73.758` | `1.54` | **`3.09`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=2` | `1024x1024` | `20` | `1` | `105.43` | **`46.868`** | `88.853` | `1.28` | **`2.56`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=2` | `1024x1024` | `20` | `2` | `105.43` | **`7.719`** | `10.418` | `11.48` | **`22.95`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=2` | `1024x1024` | `20` | `4` | `105.43` | **`12.361`** | `106.212` | `2.26` | **`4.52`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=2` | `1024x1024` | `50` | `1` | `11.37` | **`11.158`** | `11.181` | `5.36` | **`10.72`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=2` | `1328x1328` | `20` | `1` | `94.53` | **`37.234`** | `68.845` | `1.61` | **`3.22`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=2` | `1328x1328` | `20` | `2` | `94.53` | **`8.772`** | `12.061` | `9.90` | **`19.79`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=2` | `1328x1328` | `50` | `1` | `12.82` | **`13.675`** | `13.838` | `4.37` | **`8.74`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=2` | `1664x928` | `20` | `1` | `118.71` | **`54.878`** | `104.394` | `1.09` | **`2.18`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=2` | `1664x928` | `20` | `2` | `118.71` | **`8.299`** | `11.263` | `10.60` | **`21.21`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=2` | `1664x928` | `50` | `1` | `12.80` | **`12.709`** | `12.782` | `4.70` | **`9.40`** |
| `vllm-omni-sdpa` | `BF16` | `U=4 × DP=2` | `1024x1024` | `20` | `1` | `9.22` | **`6.203`** | `9.078` | `9.61` | **`19.22`** |
| `vllm-omni-sdpa` | `BF16` | `U=4 × DP=2` | `1024x1024` | `20` | `2` | `9.22` | **`5.068`** | `6.772` | `17.61` | **`35.22`** |
| `vllm-omni-sdpa` | `BF16` | `U=4 × DP=2` | `1328x1328` | `20` | `1` | `8.03` | **`5.868`** | `7.925` | `10.11` | **`20.21`** |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=1` | `512x512` | `20` | `1` | `10.28` | **`6.940`** | `10.011` | `8.63` | **`8.63`** |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=1` | `1024x1024` | `20` | `1` | `45.54` | **`7.097`** | `10.594` | `8.41` | **`8.41`** |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=1` | `1024x1024` | `20` | `2` | `45.54` | **`5.911`** | `8.138` | `14.67` | **`14.67`** |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=1` | `1024x1024` | `20` | `4` | `45.54` | **`10.954`** | `27.902` | `8.59` | **`8.59`** |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=1` | `1024x1024` | `50` | `1` | `9.69` | **`9.065`** | `9.145` | `6.59` | **`6.59`** |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=1` | `1328x1328` | `20` | `1` | `59.86` | **`8.222`** | `12.556` | `7.24` | **`7.24`** |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=1` | `1328x1328` | `20` | `2` | `59.86` | **`6.148`** | `8.345` | `14.28` | **`14.28`** |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=1` | `1328x1328` | `50` | `1` | `9.33` | **`9.371`** | `9.445` | `6.36` | **`6.36`** |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=1` | `1664x928` | `20` | `1` | `53.35` | **`9.293`** | `14.695` | `6.42` | **`6.42`** |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=1` | `1664x928` | `20` | `2` | `53.35` | **`5.865`** | `7.821` | `15.22` | **`15.22`** |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=1` | `1664x928` | `50` | `1` | `9.37` | **`9.254`** | `9.286` | `6.44` | **`6.44`** |

#### Track B — `Qwen-Image-2.1` & `Qwen-Image-2.1-Turbo` (`Qwen3-VL` Text-Encoded Native Pipeline)

| Model Checkpoint | Engine | `TP × Cluster DP` | Resolution | Steps | Weight Load (s) | Median E2E (s) | Pixel Std | Replica Img/min | Cluster Img/min |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=8` | `1024x1024` | `8` | `9.48` | **`3.623`** | `77.09` | `16.56` | **`132.48`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=8` | `1536x1024` | `8` | `9.48` | **`5.383`** | `82.54` | `11.15` | **`89.20`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=8` | `2048x2048` | `8` | `9.48` | **`15.759`** | `88.00` | `3.81` | **`30.48`** |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=8` | `1024x1024` | `28` | `11.81` | **`11.925`** | `77.42` | `5.03` | **`40.24`** |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=8` | `1536x1024` | `28` | `11.81` | **`17.859`** | `84.38` | `3.36` | **`26.88`** |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=8` | `2048x2048` | `28` | `11.81` | **`52.948`** | `78.06` | `1.13` | **`9.04`** |

#### Track C — `TorchTPU` Multi-Stage Breakdown (`Qwen-Image` & `Qwen-Image-2.1`, `BF16` & `FP8` MXU)

| Model | Engine & Kernel | Task & Resolution | Steps | VL Encode (ms) | MMDiT Step (ms) | VAE Decode (ms) | Local PNG (ms) | Disagg PNG (ms) | E2E Disagg (s) | Cluster Img/min |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-50step` | `50` | `0.80` | `60.11` | `1.61` | `112.6` | **`143.8`** | **`3.152`** | **`38.08`** |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1328x1328-50step` | `50` | `0.79` | `61.77` | `30.60` | `190.6` | **`246.9`** | **`3.367`** | **`35.64`** |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1664x928-20step` | `20` | `1.15` | `61.28` | `8,541.60` | `165.6` | **`212.1`** | **`9.980`** | **`12.02`** |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1664x928-50step` | `50` | `0.96` | `62.64` | `21.63` | `165.6` | **`212.1`** | **`3.367`** | **`35.64`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-512x512-20step` | `20` | `1,762.19` | `20.59` | `5,136.50` | `28.1` | **`37.9`** | **`7.348`** | **`8.17`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-512x512-50step` | `50` | `2.22` | `16.86` | `6.74` | `28.1` | **`37.9`** | **`0.890`** | **`67.43`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-20step` | `20` | `2.78` | `52.47` | `6,248.09` | `112.6` | **`143.8`** | **`7.444`** | **`8.06`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-50step` | `50` | `2.35` | `50.36` | `13.43` | `112.6` | **`143.8`** | **`2.677`** | **`22.41`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1328x1328-20step` | `20` | `2.55` | `105.46` | `6,324.07` | `190.6` | **`246.9`** | **`8.683`** | **`6.91`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1328x1328-50step` | `50` | `2.60` | `103.76` | `23.19` | `190.6` | **`246.9`** | **`5.460`** | **`10.99`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1664x928-20step` | `20` | `2.65` | `87.79` | `8,038.61` | `165.6` | **`212.1`** | **`10.009`** | **`5.99`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1664x928-50step` | `50` | `2.69` | `88.52` | `20.33` | `165.6` | **`212.1`** | **`4.661`** | **`12.87`** |
| `Qwen/Qwen-Image` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-20step` | `20` | `13.23` | `84.06` | `16.08` | `112.6` | **`143.8`** | **`1.854`** | **`32.36`** |
| `Qwen/Qwen-Image` | `torchtpu-eager (BF16_MXU)` | `T2I-1328x1328-20step` | `20` | `12.23` | `138.63` | `26.76` | `190.6` | **`246.9`** | **`3.058`** | **`19.62`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-512x512-20step` | `20` | `2,231.14` | `19.62` | `5,239.29` | `28.1` | **`37.9`** | **`7.901`** | **`7.59`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-512x512-50step` | `50` | `1.84` | `18.56` | `6.58` | `28.1` | **`37.9`** | **`0.974`** | **`61.59`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-20step` | `20` | `1.97` | `53.13` | `5,837.16` | `112.6` | **`143.8`** | **`7.045`** | **`8.52`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-50step` | `50` | `2.12` | `52.03` | `13.74` | `112.6` | **`143.8`** | **`2.761`** | **`21.73`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1328x1328-20step` | `20` | `1.88` | `108.30` | `6,479.12` | `190.6` | **`246.9`** | **`8.894`** | **`6.75`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1328x1328-50step` | `50` | `2.48` | `107.11` | `22.79` | `190.6` | **`246.9`** | **`5.628`** | **`10.66`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1664x928-20step` | `20` | `2.32` | `89.60` | `8,222.84` | `165.6` | **`212.1`** | **`10.229`** | **`5.87`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1664x928-50step` | `50` | `2.04` | `88.35` | `20.75` | `165.6` | **`212.1`** | **`4.653`** | **`12.90`** |
| `Qwen/Qwen-Image` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-20step` | `20` | `19.00` | `72.44` | `15.87` | `112.6` | **`143.8`** | **`1.627`** | **`36.87`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-8step` | `8` | `15.54` | `30.70` | `1.24` | `112.6` | **`143.8`** | **`0.406`** | **`295.42`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (BF16_MXU)` | `T2I-1536x1024-8step` | `8` | `17.38` | `29.96` | `0.90` | `175.8` | **`222.8`** | **`0.481`** | **`249.62`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-8step` | `8` | `18.84` | `35.52` | `0.85` | `112.6` | **`143.8`** | **`0.448`** | **`268.06`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-8step` | `8` | `2,323.10` | `21.81` | `6,085.00` | `112.6` | **`143.8`** | **`8.726`** | **`6.88`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `T2I-1536x1024-8step` | `8` | `3.18` | `35.68` | `6,235.58` | `175.8` | **`222.8`** | **`6.747`** | **`8.89`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `T2I-2048x2048-8step` | `8` | `3.09` | `100.98` | `6,953.58` | `452.3` | **`570.6`** | **`8.335`** | **`7.20`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1024x1024-8step` | `8` | `2,315.71` | `28.74` | `13.76` | `112.6` | **`143.8`** | **`2.729`** | **`21.99`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1536x1024-8step` | `8` | `7.05` | `57.00` | `19.61` | `175.8` | **`222.8`** | **`0.731`** | **`82.07`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-8step` | `8` | `16.81` | `31.45` | `15.83` | `112.6` | **`143.8`** | **`0.428`** | **`140.16`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (BF16_MXU)` | `T2I-1536x1024-8step` | `8` | `12.34` | `45.89` | `23.23` | `175.8` | **`222.8`** | **`0.625`** | **`95.93`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-8step` | `8` | `2,776.27` | `18.95` | `5,681.96` | `112.6` | **`143.8`** | **`8.754`** | **`6.85`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1536x1024-8step` | `8` | `2.78` | `30.57` | `6,215.12` | `175.8` | **`222.8`** | **`6.685`** | **`8.97`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-2048x2048-8step` | `8` | `2.40` | `91.36` | `6,912.62` | `452.3` | **`570.6`** | **`8.217`** | **`7.30`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1024x1024-8step` | `8` | `2,904.81` | `25.37` | `14.07` | `112.6` | **`143.8`** | **`3.291`** | **`18.23`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1536x1024-8step` | `8` | `4.86` | `52.66` | `19.28` | `175.8` | **`222.8`** | **`0.694`** | **`86.48`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-8step` | `8` | `16.73` | `31.13` | `15.86` | `112.6` | **`143.8`** | **`0.425`** | **`141.03`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `0.91` | `26.25` | `125.73` | `175.8` | **`222.8`** | **`1.110`** | **`108.12`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-28step` | `28` | `15.94` | `28.88` | `1.13` | `112.6` | **`143.8`** | **`0.969`** | **`123.78`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (BF16_MXU)` | `T2I-1536x1024-28step` | `28` | `16.28` | `29.74` | `1.49` | `175.8` | **`222.8`** | **`1.073`** | **`111.80`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `1.06` | `35.32` | `109.11` | `175.8` | **`222.8`** | **`1.348`** | **`89.06`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-28step` | `28` | `2,327.00` | `40.08` | `5,863.16` | `112.6` | **`143.8`** | **`9.456`** | **`6.34`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `T2I-1536x1024-28step` | `28` | `2.80` | `71.25` | `6,038.64` | `175.8` | **`222.8`** | **`8.259`** | **`7.26`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `T2I-2048x2048-28step` | `28` | `2.71` | `202.88` | `7,048.35` | `452.3` | **`570.6`** | **`13.302`** | **`4.51`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1024x1024-28step` | `28` | `2,288.68` | `54.43` | `14.17` | `112.6` | **`143.8`** | **`3.996`** | **`15.01`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `6.87` | `100.05` | `19.14` | `175.8` | **`222.8`** | **`3.076`** | **`19.51`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-28step` | `28` | `13.89` | `54.49` | `15.82` | `112.6` | **`143.8`** | **`1.699`** | **`35.31`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (BF16_MXU)` | `T2I-1536x1024-28step` | `28` | `19.82` | `84.97` | `22.71` | `175.8` | **`222.8`** | **`2.644`** | **`22.69`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-28step` | `28` | `2,838.55` | `37.82` | `6,174.55` | `112.6` | **`143.8`** | **`10.216`** | **`5.87`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1536x1024-28step` | `28` | `2.87` | `66.24` | `6,028.80` | `175.8` | **`222.8`** | **`8.109`** | **`7.40`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-2048x2048-28step` | `28` | `2.70` | `195.44` | `6,631.72` | `452.3` | **`570.6`** | **`12.677`** | **`4.73`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1024x1024-28step` | `28` | `3,005.67` | `50.25` | `14.11` | `112.6` | **`143.8`** | **`4.596`** | **`13.05`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `4.93` | `92.10` | `19.83` | `175.8` | **`222.8`** | **`2.852`** | **`21.04`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-28step` | `28` | `22.21` | `48.36` | `15.69` | `112.6` | **`143.8`** | **`1.536`** | **`39.07`** |

#### Track D — Disaggregated CPU Image Encoding (`PNG` & `WebP`) Ablation (`POST /encode_image`)

| Resolution & Format | On-Host CPU (ms) | Disagg CPU RTT (ms) | Remote AVX-512 Compute (ms) | Remote Rate (img/s) | Local / Disagg RTT Ratio |
|---|---:|---:|---:|---:|---:|
| `512x512_png` | `28.14` | **`37.90`** | `34.98` | `28.6` | **`0.74x`** |
| `512x512_webp` | `27.92` | **`40.54`** | `34.20` | `29.2` | **`0.69x`** |
| `1024x1024_png` | `112.61` | **`143.81`** | `139.99` | `7.1` | **`0.78x`** |
| `1024x1024_webp` | `112.52` | **`143.63`** | `135.34` | `7.4` | **`0.78x`** |
| `1328x1328_png` | `190.64` | **`246.92`** | `236.64` | `4.2` | **`0.77x`** |
| `1328x1328_webp` | `191.07` | **`234.02`** | `229.31` | `4.4` | **`0.82x`** |
| `1664x928_png` | `165.64` | **`212.07`** | `207.18` | `4.8` | **`0.78x`** |
| `1664x928_webp` | `167.28` | **`208.82`** | `199.39` | `5.0` | **`0.80x`** |
| `1536x1024_png` | `175.76` | **`222.80`** | `212.14` | `4.7` | **`0.79x`** |
| `1536x1024_webp` | `169.83` | **`213.22`** | `203.54` | `4.9` | **`0.80x`** |
| `2048x2048_png` | `452.28` | **`570.59`** | `563.54` | `1.8` | **`0.79x`** |
| `2048x2048_webp` | `453.21` | **`551.66`** | `541.30` | `1.8` | **`0.82x`** |
