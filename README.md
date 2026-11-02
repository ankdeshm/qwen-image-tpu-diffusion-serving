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
| **Disaggregated CPU Offload** | `multimodal-cpu-offload-svc` (`c3-standard-44` AVX-512): `POST /encode_image` parallel PNG/WebP encoding & TI2I `POST /preprocess_vision` |

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

## 4. Before vs. After Disaggregated Image Encoding & Cross-Generation (`v5p` vs `v7x`) Comparison

Direct A/B comparison across `v5p-16` and `v7x` evaluating **(1) on-host Pillow encode vs. disaggregated AVX-512 `POST /encode_image` (`PNG` & `WebP`)** and **(2) `v5p` vs. `v7x` `vllm-omni` serving throughput**:

| Comparison Dimension | Baseline (`v5p-16` / On-Host CPU) | Optimized (`v7x` / Disagg AVX-512 CPU) | Measured Gain | Code & Result References |
|---|---:|---:|---:|---|
| **`2048x2048` `WebP` / `PNG` Encoding (`POST /encode_image`)** | `614.81 ms` (`WebP`) / `615.21 ms` (`PNG`) on-host | **`554.44 ms` (`WebP`) / `583.98 ms` (`PNG`)** (`540.6 ms` compute) | **`1.11x` faster** (`-60.4 ms`/img + frees TPU host GIL) | [`multimodal_cpu_offload_server.py`](./scripts/multimodal_cpu_offload_server.py) · [`v5p-16/summary.json`](./results/v5p-16/summary.json) |
| **`1536x1024` & `1664x928` `WebP` Encoding** | `232.90 ms` / `228.54 ms` on-host | **`208.74 ms` / `208.14 ms`** disagg RTT | **`1.12x` / `1.10x` faster** | [`cpu_offload_client.py`](./scripts/cpu_offload_client.py) · [`v5p-16/summary.json`](./results/v5p-16/summary.json) |
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

### Slice `v7x-16` (`v7x`, `14` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v7x-16/SUMMARY.md`](./results/v7x-16/SUMMARY.md) · [`results/v7x-16/summary.json`](./results/v7x-16/summary.json) *(updated `2026-11-02T20:01:36.668802+00:00`)*

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

### Slice `v7x-spot-4c` (`v7x`, `25` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v7x-spot-4c/SUMMARY.md`](./results/v7x-spot-4c/SUMMARY.md) · [`results/v7x-spot-4c/summary.json`](./results/v7x-spot-4c/summary.json) *(updated `2026-11-02T19:59:29.819135+00:00`)*

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
