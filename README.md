# `Qwen/Qwen-Image` (20B MMDiT) & `Qwen/Qwen-Image-2.1` (`Qwen3-VL`) Diffusion Serving on Cloud TPU `v5p` & `v7x`

Qwen/Qwen-Image (20B MMDiT) & Qwen/Qwen-Image-2.1 / Qwen-Image-2.1-Turbo (Qwen3-VL Text-Encoded Image Diffusion) Serving on Cloud TPU v5p & v7x with vLLM-Omni, Diffusers, TorchTPU, & Disaggregated Image Encoding.

---

## 1. Model & System Architecture

| Component | Specification |
|---|---|
| **Checkpoints** | [`Qwen/Qwen-Image`](https://huggingface.co/Qwen/Qwen-Image) (`20B` MMDiT), [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) (`28`-step `Qwen3-VL`), & [`Qwen/Qwen-Image-2.1-Turbo`](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) (`8`-step `Qwen3-VL`) |
| **Architecture** | Family A: `Qwen2.5-VL` + `QwenImageTransformer2DModel` (`20B` MMDiT) · Family B (`Qwen 3 Image`): `Qwen3VLForConditionalGeneration` + `QwenImage21Transformer2DModel` + `AutoencoderKLQwenImage21` |
| **Serving Engines** | `vllm-omni` HTTP (`--ulysses-degree 4`, `flash_attn` & `sdpa`) + `Diffusers` `QwenImage21Pipeline` + `TorchTPU` (`compiled` & `eager`, `BF16` & `FP8` MXU) |
| **Resolutions** | `512x512`, `1024x1024`, `1328x1328`, `1536x1024`, `1664x928`, and `2048x2048` (`2K` native) |
| **Disaggregated CPU Offload** | `multimodal-cpu-offload-svc` (`c3-standard-44` AVX-512, `HTTP/1.1 Keep-Alive + TCP_NODELAY`): `POST /encode_image` parallel PNG/WebP encoding & TI2I `POST /preprocess_vision` |

---

## 2. Disaggregated CPU Offloading Architecture (`multimodal-cpu-offload-svc`)

Non-TPU media muxing, high-resolution image compression, and multi-camera/multi-frame vision preprocessing are offloaded over intra-cluster `ClusterIP` HTTP (`port 8090`) to dedicated `c3-standard-44` (`44` Sapphire Rapids vCPUs, AVX-512, `176 GiB` DDR5) CPU pods ([`scripts/multimodal_cpu_offload_server.py`](./scripts/multimodal_cpu_offload_server.py) & [`scripts/cpu_offload_client.py`](./scripts/cpu_offload_client.py)), keeping TPU host CPUs dedicated to XLA/PJRT dispatch:

- `POST /encode` — Multi-threaded `ffmpeg` (`libx264` + `48 kHz` stereo `AAC`) video/audio MP4 muxing.
- `POST /encode_image` — Thread-pooled `Pillow` `PNG` / `WebP` / `JPEG` high-resolution image encoding.
- `POST /encode_wav` — Vectorized `float32` / `int16` audio normalization + `RIFF/WAV` packet framing (`24 kHz` / `48 kHz`).
- `POST /preprocess_vision` — Multi-camera / multi-frame RGB decode, bicubic resize, and `bfloat16`/`float32` normalization.

---

## 3. Hardware Platforms Evaluated

| Accelerator | Slice Topologies | HBM Capacity & Bandwidth | Host Memory (`/dev/shm`) |
|---|---|---|---|
| **Cloud TPU `v5p`** | `v5p-16` (`2x2x4`, `4` hosts × `4` chips) & `v5p-32` (`2x4x4`, `8` hosts × `4` chips) | `95 GiB` HBM2e / chip (`2,765 GB/s` / chip, `459` BF16 TFLOP/s) | `462 GiB` RAM (`380 GiB` tmpfs) |
| **Cloud TPU `v7x` (Ironwood)** | `v7x-4` (`2x2x1`, `8` TensorCores), `v7x-8` (`2x2x2`, `16` TensorCores), & `v7x-16` (`2x2x4`, `32` TensorCores) | `192 GiB` HBM3e / chip (`96 GiB` / TensorCore, `7,370 GB/s` / chip, `2,307` BF16 / `4,614` FP8 TFLOP/s per chip) | `944 GiB` RAM (`850 GiB` tmpfs) |

---

## 4. Before vs. After Disaggregated Image Encoding (`Keep-Alive + TCP_NODELAY`) & Cross-Generation (`v5p` vs `v7x`) Comparison

Direct A/B comparison across `v5p-16` (`Wave-1` & `Wave-3 Keep-Alive`) and `v7x` evaluating **(1) on-host Pillow encode vs. persistent `HTTP/1.1 Keep-Alive + TCP_NODELAY` AVX-512 `POST /encode_image` (`PNG` & `WebP`)** and **(2) `v5p` vs. `v7x` `vllm-omni` serving throughput**:

| Comparison Dimension | Baseline (`v5p-16` / On-Host CPU) | Optimized (`v7x` / `Keep-Alive` AVX-512 CPU) | Measured Gain | Code & Result References |
|---|---:|---:|---:|---|
| **`2048x2048` `WebP` / `PNG` Encoding (`Wave-3 Keep-Alive`)** | `617.05 ms` on-host CPU | **`549.99 ms` (`WebP`) / `572.14 ms` (`PNG`)** Keep-Alive RTT | **`1.12x` faster** (`-67.1 ms`/img + frees TPU host GIL) | [`cpu_offload_client.py`](./scripts/cpu_offload_client.py) · [`v5p-16-wave3-keepalive/summary.json`](./results/v5p-16-wave3-keepalive/summary.json) |
| **`1536x1024` & `1664x928` `WebP` Encoding (`Keep-Alive`)** | `233.67 ms` / `228.54 ms` on-host | **`208.15 ms` / `208.14 ms`** Keep-Alive RTT | **`1.12x` / `1.10x` faster** (`-25.5 ms`/img) | [`cpu_offload_client.py`](./scripts/cpu_offload_client.py) · [`v5p-16-wave3-keepalive/summary.json`](./results/v5p-16-wave3-keepalive/summary.json) |
| **`Qwen-Image` `1024x1024` `20`-step (`c=2`, `vllm-omni U=4`)** | `9.375 s` P50 (`38.05 img/min` on `v5p-16`) | **`7.364 s` P50** (`v7x-spot-4c`, **`101.18 img/min`** per 16-chip `v7x` slice) | **`1.27x–1.32x` faster latency · `2.66x` throughput/slice** | [`v5p-16/summary.json`](./results/v5p-16/summary.json) · [`v7x-spot-4c/summary.json`](./results/v7x-spot-4c/summary.json) |
| **`Qwen-Image` `1664x928` `20`-step (`c=2`, `vllm-omni U=4`)** | `10.978 s` P50 (`32.79 img/min` on `v5p-16`) | **`8.299 s` P50** (`v7x-spot-4c`, **`84.84 img/min`** per 16-chip `v7x` slice) | **`1.32x` faster latency · `2.59x` throughput/slice** | [`v5p-16/summary.json`](./results/v5p-16/summary.json) · [`v7x-spot-4c/summary.json`](./results/v7x-spot-4c/summary.json) |
| **`Qwen-Image-2.1-Turbo` (`8`-step `1024x1024` / `2048x2048`)** | `14.616 s` (`Qwen-Image` `50`-step) | **`3.384 s`** (`283.68 img/min` @ `1K`) · **`14.497 s`** (`66.24 img/min` @ `2K`) | **`4.32x` faster latency · `7.46x` cluster throughput** | [`run_qwen_image_tpu_bench.py`](./scripts/run_qwen_image_tpu_bench.py) · [`v5p-16/summary.json`](./results/v5p-16/summary.json) |

---

## 5. Complete Benchmark Tables (`v5p` & `v7x`)

Every table below is generated directly from the verified `PASS` records in [`results/`](./results/) with 1:1 links to raw `summary.json` artifacts:

### Slice `v5p-16` (`v5p`, `65` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v5p-16/SUMMARY.md`](./results/v5p-16/SUMMARY.md) · [`results/v5p-16/summary.json`](./results/v5p-16/summary.json) *(updated `2026-11-02T19:59:51.873328+00:00`)*

#### Track A — OpenAI-Compatible `vllm-omni` HTTP Serving (`Qwen/Qwen-Image` 20B MMDiT)

| Engine | Precision | `Ulysses × Cluster DP` | Resolution | Steps | Conc | Warmup (s) | Median E2E (s) | P95 E2E (s) | Replica Img/min | Cluster Img/min |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `512x512` | `20` | `1` | `117.12` | **`59.900`** | `114.816` | `1.00` | **`4.01`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `1` | `116.08` | **`61.369`** | `116.708` | `0.98` | **`3.91`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `2` | `116.08` | **`9.375`** | `12.569` | `9.51` | **`38.05`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `4` | `116.08` | **`15.990`** | `139.569` | `1.72` | **`6.88`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1024x1024` | `50` | `1` | `14.83` | **`14.616`** | `14.623` | `4.09` | **`16.37`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1328x1328` | `20` | `1` | `86.63` | **`48.276`** | `88.989` | `1.24` | **`4.96`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1328x1328` | `20` | `2` | `86.63` | **`11.785`** | `16.032` | `7.45` | **`29.80`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1328x1328` | `50` | `1` | `18.22` | **`18.257`** | `18.283` | `3.27` | **`13.09`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1664x928` | `20` | `1` | `123.67` | **`65.712`** | `124.264` | `0.91` | **`3.65`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1664x928` | `20` | `2` | `123.67` | **`11.520`** | `15.659` | `7.63` | **`30.52`** |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1664x928` | `50` | `1` | `17.40` | **`17.536`** | `17.753` | `3.41` | **`13.63`** |
| `vllm-omni-sdpa` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `1` | `10.18` | **`7.808`** | `10.330` | `7.64` | **`30.56`** |
| `vllm-omni-sdpa` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `2` | `10.18` | **`8.044`** | `10.766` | `11.10` | **`44.41`** |
| `vllm-omni-sdpa` | `BF16` | `U=4 × DP=4` | `1328x1328` | `20` | `1` | `8.29` | **`6.428`** | `7.496` | `9.22` | **`36.89`** |

#### Track B — `Qwen-Image-2.1` & `Qwen-Image-2.1-Turbo` (`Qwen3-VL` Text-Encoded Native Pipeline)

| Model Checkpoint | Engine | `TP × Cluster DP` | Resolution | Steps | Weight Load (s) | Median E2E (s) | Pixel Std | Replica Img/min | Cluster Img/min |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `1024x1024` | `8` | `13.33` | **`3.384`** | `83.17` | `17.73` | **`283.68`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `1536x1024` | `8` | `13.33` | **`5.016`** | `84.90` | `11.96` | **`191.36`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `2048x2048` | `8` | `13.33` | **`14.497`** | `94.31` | `4.14` | **`66.24`** |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `1024x1024` | `28` | `13.55` | **`10.779`** | `81.53` | `5.57` | **`89.12`** |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `1536x1024` | `28` | `13.55` | **`16.355`** | `70.32` | `3.67` | **`58.72`** |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `2048x2048` | `28` | `13.55` | **`48.005`** | `84.10` | `1.25` | **`20.00`** |

#### Track C — `TorchTPU` Multi-Stage Breakdown (`Qwen-Image` & `Qwen-Image-2.1`, `BF16` & `FP8` MXU)

| Model | Engine & Kernel | Task & Resolution | Steps | VL Encode (ms) | MMDiT Step (ms) | VAE Decode (ms) | Local PNG (ms) | Disagg PNG (ms) | E2E Disagg (s) | Cluster Img/min |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-512x512-20step` | `20` | `2,487.28` | `48.91` | `2,521.94` | `38.9` | **`41.6`** | **`6.029`** | **`39.80`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-512x512-50step` | `50` | `5.23` | `44.43` | `3.03` | `38.9` | **`41.6`** | **`2.271`** | **`105.68`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-20step` | `20` | `5.49` | `123.21` | `2,554.30` | `155.3` | **`144.8`** | **`5.169`** | **`46.44`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-50step` | `50` | `5.76` | `122.52` | `6.04` | `155.3` | **`144.8`** | **`6.283`** | **`38.20`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1328x1328-20step` | `20` | `5.49` | `228.30` | `2,683.87` | `261.1` | **`247.7`** | **`7.503`** | **`32.00`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1328x1328-50step` | `50` | `5.23` | `227.90` | `10.84` | `261.1` | **`247.7`** | **`11.659`** | **`20.60`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1664x928-20step` | `20` | `5.59` | `194.94` | `3,155.25` | `228.6` | **`212.3`** | **`7.272`** | **`33.00`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1664x928-50step` | `50` | `5.24` | `195.11` | `8.82` | `228.6` | **`212.3`** | **`9.982`** | **`24.04`** |
| `Qwen/Qwen-Image` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-20step` | `20` | `26.28` | `164.79` | `6.42` | `155.3` | **`144.8`** | **`3.473`** | **`69.08`** |
| `Qwen/Qwen-Image` | `torchtpu-eager (BF16_MXU)` | `T2I-1328x1328-20step` | `20` | `26.79` | `287.52` | `10.62` | `261.1` | **`247.7`** | **`6.036`** | **`39.76`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-512x512-20step` | `20` | `3,937.31` | `119.29` | `2,407.48` | `38.9` | **`41.6`** | **`8.772`** | **`27.36`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-512x512-50step` | `50` | `16.88` | `116.29` | `3.06` | `38.9` | **`41.6`** | **`5.876`** | **`40.84`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-20step` | `20` | `17.50` | `318.31` | `2,615.27` | `155.3` | **`144.8`** | **`9.144`** | **`26.24`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-50step` | `50` | `16.97` | `317.89` | `6.14` | `155.3` | **`144.8`** | **`16.063`** | **`14.96`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1328x1328-20step` | `20` | `17.41` | `554.99` | `2,745.46` | `261.1` | **`247.7`** | **`14.110`** | **`17.00`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1328x1328-50step` | `50` | `16.97` | `555.04` | `10.80` | `261.1` | **`247.7`** | **`28.028`** | **`8.56`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1664x928-20step` | `20` | `17.42` | `406.26` | `3,081.36` | `228.6` | **`212.3`** | **`11.436`** | **`21.00`** |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1664x928-50step` | `50` | `16.94` | `405.58` | `8.71` | `228.6` | **`212.3`** | **`20.517`** | **`11.68`** |
| `Qwen/Qwen-Image` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-20step` | `20` | `45.50` | `405.39` | `6.31` | `155.3` | **`144.8`** | **`8.305`** | **`28.88`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-8step` | `8` | `3,401.89` | `57.57` | `2,699.28` | `155.3` | **`144.8`** | **`6.707`** | **`35.80`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `T2I-1536x1024-8step` | `8` | `8.57` | `85.38` | `2,712.08` | `232.4` | **`222.3`** | **`3.626`** | **`66.20`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `T2I-2048x2048-8step` | `8` | `8.69` | `248.81` | `3,260.28` | `615.2` | **`584.0`** | **`5.843`** | **`41.08`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1024x1024-8step` | `8` | `3,478.65` | `72.02` | `6.18` | `155.3` | **`144.8`** | **`4.240`** | **`56.60`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1536x1024-8step` | `8` | `21.42` | `130.22` | `9.25` | `232.4` | **`222.3`** | **`1.329`** | **`180.56`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-8step` | `8` | `48.21` | `70.65` | `6.37` | `155.3` | **`144.8`** | **`0.765`** | **`313.88`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (BF16_MXU)` | `T2I-1536x1024-8step` | `8` | `47.41` | `109.71` | `8.77` | `232.4` | **`222.3`** | **`1.156`** | **`207.56`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-8step` | `8` | `4,855.42` | `146.58` | `2,672.86` | `155.3` | **`144.8`** | **`8.846`** | **`27.12`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1536x1024-8step` | `8` | `23.42` | `191.83` | `2,649.90` | `232.4` | **`222.3`** | **`4.430`** | **`54.16`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-2048x2048-8step` | `8` | `22.86` | `537.59` | `3,376.23` | `615.2` | **`584.0`** | **`8.284`** | **`28.96`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1024x1024-8step` | `8` | `5,140.87` | `183.71` | `6.39` | `155.3` | **`144.8`** | **`6.796`** | **`35.32`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1536x1024-8step` | `8` | `47.10` | `299.60` | `9.14` | `232.4` | **`222.3`** | **`2.710`** | **`88.56`** |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-8step` | `8` | `44.79` | `190.59` | `6.39` | `155.3` | **`144.8`** | **`1.721`** | **`139.48`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-28step` | `28` | `3,418.93` | `105.86` | `2,759.89` | `155.3` | **`144.8`** | **`9.288`** | **`25.84`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `T2I-1536x1024-28step` | `28` | `8.42` | `165.76` | `2,730.69` | `232.4` | **`222.3`** | **`7.603`** | **`31.56`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `T2I-2048x2048-28step` | `28` | `8.68` | `500.88` | `3,214.21` | `615.2` | **`584.0`** | **`17.831`** | **`13.44`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1024x1024-28step` | `28` | `3,539.28` | `137.22` | `6.14` | `155.3` | **`144.8`** | **`7.567`** | **`31.72`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `21.41` | `245.45` | `9.32` | `232.4` | **`222.3`** | **`7.160`** | **`33.52`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-28step` | `28` | `47.96` | `134.74` | `6.31` | `155.3` | **`144.8`** | **`3.972`** | **`60.44`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (BF16_MXU)` | `T2I-1536x1024-28step` | `28` | `46.84` | `206.00` | `8.72` | `232.4` | **`222.3`** | **`6.046`** | **`39.68`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-28step` | `28` | `4,843.23` | `314.41` | `2,601.35` | `155.3` | **`144.8`** | **`16.393`** | **`14.64`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1536x1024-28step` | `28` | `23.30` | `465.08` | `2,793.47` | `232.4` | **`222.3`** | **`16.061`** | **`14.96`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-2048x2048-28step` | `28` | `23.04` | `1,271.28` | `3,366.60` | `615.2` | **`584.0`** | **`39.569`** | **`6.08`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1024x1024-28step` | `28` | `5,118.60` | `402.54` | `6.25` | `155.3` | **`144.8`** | **`16.575`** | **`14.48`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `46.92` | `570.30` | `9.34` | `232.4` | **`222.3`** | **`16.281`** | **`14.76`** |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-28step` | `28` | `63.76` | `359.30` | `6.24` | `155.3` | **`144.8`** | **`10.275`** | **`23.36`** |

#### Track D — Disaggregated CPU Image Encoding (`PNG` & `WebP`) Ablation (`POST /encode_image`)

| Resolution & Format | On-Host CPU (ms) | Disagg CPU RTT (ms) | Remote AVX-512 Compute (ms) | Remote Rate (img/s) | Speedup vs On-Host |
|---|---:|---:|---:|---:|---:|
| `512x512_png` | `38.87` | **`41.62`** | `34.88` | `28.7` | **`0.93x`** |
| `512x512_webp` | `38.61` | **`41.67`** | `38.06` | `26.3` | **`0.93x`** |
| `1024x1024_png` | `155.28` | **`144.85`** | `140.96` | `7.1` | **`1.07x`** |
| `1024x1024_webp` | `155.06` | **`144.12`** | `135.23` | `7.4` | **`1.08x`** |
| `1328x1328_png` | `261.07` | **`247.68`** | `236.78` | `4.2` | **`1.05x`** |
| `1328x1328_webp` | `261.43` | **`239.10`** | `228.28` | `4.4` | **`1.09x`** |
| `1664x928_png` | `228.64` | **`212.28`** | `207.96` | `4.8` | **`1.08x`** |
| `1664x928_webp` | `228.54` | **`208.14`** | `199.02` | `5.0` | **`1.10x`** |
| `1536x1024_png` | `232.42` | **`222.32`** | `211.62` | `4.7` | **`1.05x`** |
| `1536x1024_webp` | `232.90` | **`208.74`** | `204.20` | `4.9` | **`1.12x`** |
| `2048x2048_png` | `615.21` | **`583.98`** | `564.61` | `1.8` | **`1.05x`** |
| `2048x2048_webp` | `614.81` | **`554.44`** | `540.62` | `1.8` | **`1.11x`** |

### Slice `v5p-16-wave3-keepalive` (`v5p`, `65` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v5p-16-wave3-keepalive/SUMMARY.md`](./results/v5p-16-wave3-keepalive/SUMMARY.md) · [`results/v5p-16-wave3-keepalive/summary.json`](./results/v5p-16-wave3-keepalive/summary.json) *(updated `2026-11-02T20:54:39.543913+00:00`)*

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

| Resolution & Format | On-Host CPU (ms) | Disagg CPU RTT (ms) | Remote AVX-512 Compute (ms) | Remote Rate (img/s) | Speedup vs On-Host |
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

### Slice `v5p-16-wave4-adaptive64k` (`v5p`, `5` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v5p-16-wave4-adaptive64k/SUMMARY.md`](./results/v5p-16-wave4-adaptive64k/SUMMARY.md) · [`results/v5p-16-wave4-adaptive64k/summary.json`](./results/v5p-16-wave4-adaptive64k/summary.json) *(updated `2026-11-02T21:22:16.697207+00:00`)*

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

### Slice `v7x-16` (`v7x`, `88` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v7x-16/SUMMARY.md`](./results/v7x-16/SUMMARY.md) · [`results/v7x-16/summary.json`](./results/v7x-16/summary.json) *(updated `2026-11-02T20:43:46.998504+00:00`)*

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

| Resolution & Format | On-Host CPU (ms) | Disagg CPU RTT (ms) | Remote AVX-512 Compute (ms) | Remote Rate (img/s) | Speedup vs On-Host |
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

### Slice `v7x-8-wave3-keepalive` (`v7x`, `1` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v7x-8-wave3-keepalive/SUMMARY.md`](./results/v7x-8-wave3-keepalive/SUMMARY.md) · [`results/v7x-8-wave3-keepalive/summary.json`](./results/v7x-8-wave3-keepalive/summary.json) *(updated `2026-11-02T21:22:11.533253+00:00`)*

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

### Slice `v7x-spot-4c` (`v7x`, `87` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v7x-spot-4c/SUMMARY.md`](./results/v7x-spot-4c/SUMMARY.md) · [`results/v7x-spot-4c/summary.json`](./results/v7x-spot-4c/summary.json) *(updated `2026-11-02T20:32:54.561312+00:00`)*

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

| Resolution & Format | On-Host CPU (ms) | Disagg CPU RTT (ms) | Remote AVX-512 Compute (ms) | Remote Rate (img/s) | Speedup vs On-Host |
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

---

## 6. Repository Layout & Reproduction

- [`scripts/run_qwen_image_tpu_bench.py`](./scripts/run_qwen_image_tpu_bench.py) — Main TPU `v5p` / `v7x` benchmark driver (cooperative zero-disk HF-to-GCS staging, `/dev/shm` hydration, single-host PJRT isolation, `TorchTPU` / `vLLM` execution, and A/B CPU offload ablation).
- [`scripts/multimodal_cpu_offload_server.py`](./scripts/multimodal_cpu_offload_server.py) & [`scripts/cpu_offload_client.py`](./scripts/cpu_offload_client.py) — Disaggregated CPU offload HTTP server and client.
- [`scripts/gcs_stager.py`](./scripts/gcs_stager.py) & [`scripts/tpu_single_host_setup.py`](./scripts/tpu_single_host_setup.py) — Multi-host GCS shard stager and single-host `libtpu` metadata hook.
- [`k8s/`](./k8s/) — Kubernetes Deployment & Indexed Job manifests for CPU offload pool, TPU `v5p`, and TPU `v7x`.

```bash
# 1. Deploy the disaggregated CPU offload service
kubectl apply -f k8s/multimodal-cpu-offload-service.yaml

# 2. Launch the TPU v5p / v7x Indexed Job
kubectl apply -f k8s/v5p16-gr00t-and-qwen-image.yaml
```
