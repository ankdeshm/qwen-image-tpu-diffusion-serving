# Appendix: Multi-Track Sweep Progression, Operator Breakdowns, and Failure Analysis (Qwen-Image & Qwen-Image-2.1 on TPU v5p and v7x)

This appendix preserves the complete 396-row per-slice benchmark tables across all four evaluation tracks (`vllm-omni` HTTP serving, native `diffusers` `QwenImage21Pipeline`, `TorchTPU` multi-stage operator breakdowns, and HTTP Keep-Alive CPU PNG/WebP encoding), along with compilation diagnostics, rejected configurations, and measurement caveats. For headline results and reproduction commands, see [`README.md`](./README.md).

---

## 1. Measurement Scope Across the Four Evaluation Tracks

1. **Track A — `vllm-omni` OpenAI HTTP Serving (`Qwen/Qwen-Image`, 20B MMDiT):**
   - Full end-to-end checkpoint inference (`vllm serve /dev/shm/Qwen-Image --omni --ulysses-degree=4|8`) verified via decoded output image statistics (`pixel_std ~ 75–96`).
   - *Archived `Conc=1` `20`-step warmup note:* In the recorded `flash_attn` runs, initial warmup executed only `PROMPTS[0]`. During the `Conc=1` `20`-step pass (`2` sequential requests), `PROMPTS[1]` had a distinct `Qwen2.5-VL` token length and triggered a one-time cold XLA compilation (`~72–127 s`), skewing `Conc=1` `20`-step `Median E2E` (`37–66 s`). All subsequent `Conc=2` `20`-step (`5.11–11.79 s`), `50`-step (`8.72–18.26 s`), and `vllm-omni-sdpa` (`5.11–8.04 s`) rows ran against warm XLA caches. [`scripts/run_qwen_image_tpu_bench.py`](./scripts/run_qwen_image_tpu_bench.py) now warms up across multiple prompt token lengths upfront.
2. **Track B — Native `diffusers` `QwenImage21Pipeline` (`Qwen-Image-2.1-Turbo` 8-step & `Qwen-Image-2.1` 28-step):**
   - Full end-to-end checkpoint inference on a single TPU device (`tpu:0`, `TP=1`), enabled by the `30.21–30.84 GiB` `bfloat16` footprint fitting inside one `v5p` chip (`95 GiB`) or one `v7x` TensorCore (`96 GiB`) with zero inter-chip collectives.
   - Because each physical `v7x` chip contains `2` independent TensorCores (`tpu:0` and `tpu:1`), a single `TP=1` stream on `tpu:0` uses half a `v7x` chip (`3.60 s` vs. `3.30 s` on a full `v5p` chip at `1024×1024`), while running `2` independent replicas per physical `v7x` chip yields `~1.84×` higher throughput per physical chip (`33.38` vs. `18.19 img/min/chip`).
3. **Track C — `TorchTPU` Multi-Stage Operator Breakdown (`torchtpu-compiled` & `torchtpu-eager`):**
   - Local-shard operator microbenchmark (`run_child_torchtpu_mmdit_worker` in [`scripts/run_qwen_image_tpu_bench.py`](./scripts/run_qwen_image_tpu_bench.py)) holding real `safetensors` shards in HBM while timing local-shard `Stage1` VL encoding, `6` measured `Stage2` MMDiT blocks scaled `× 10` (`→ 60` layers), and `Stage3` VAE decoding **excluding inter-chip `AllReduce`/`AllToAll` collectives**.
   - On `torchtpu-compiled`, first-shape rows (`20`-step / `8`-step / `28`-step) include first-call `TorchDynamo` + XLA compilation inside `VL Encode` (`~1.7–5.1 s`) and `VAE Decode` (`~2.4–8.8 s`), whereas subsequent `50`-step and `TI2I` rows reflect steady-state execution (`2–24 ms` VL/VAE).
   - Slice-level `Cluster Img/min` columns in the raw tables multiply single-replica measured throughput by `cluster_dp_replicas` (`4` or `16` on `v5p-16`; `2–32` across `v7x-4`, `v7x-8`, and `v7x-16`).
4. **Track D — Disaggregated CPU Image Encoding (`POST /encode_image`, `PNG` & `WebP`):**
   - End-to-end synchronous HTTP/1.1 Keep-Alive + `TCP_NODELAY` calls from the TPU host to a `c3-standard-44` (`44` Sapphire Rapids vCPUs, AVX-512) service ([`scripts/multimodal_cpu_offload_server.py`](./scripts/multimodal_cpu_offload_server.py)) compared against local on-host Pillow encoding.

---

## 2. Summary of Failed, Rejected, or Slower Configurations

1. **`vllm-omni` `--quantization fp8` (`U=4` and `U=8` on `v7x`):** Exited with `rc=1` during server initialization (`startup_timeout` in `vllm_omni`), recorded as `DIAGNOSTIC` in `results/v7x-4/summary.json` and `results/v7x-8/summary.json`. All `vllm-omni` HTTP serving results therefore use `bfloat16`.
2. **Multi-threaded `torch.compile` on `v7x` `TP=4` (`dp_per_host=2`):** Running two concurrent `TorchDynamo` export threads in one Python process (`tpu:0` and `tpu:4` on an 8-TensorCore `v7x` host) triggered `BackendCompilerFailed: RuntimeError: set_export_mode_default_device_gen` due to thread-global export generator state in `torch_tpu 0.2.0`. Single-replica `TP=8` (`dp_per_host=1`) compiled cleanly across all shapes.
3. **`FP8` `torch._scaled_mm` on `TPU v5p` (Track C):** Because `TPU v5p` lacks native FP8 matmul hardware, `float8_e4m3fn` dynamic cast + scaled matmul ran **2.6× slower** than `bfloat16` on `Qwen-Image` `1024×1024` (`317.89 ms` vs. `122.52 ms` per compiled MMDiT step). On `TPU v7x` (native FP8 MXU), compiled `FP8` matched `bfloat16` on `Qwen-Image` (`51.85 ms` vs. `50.16 ms`) and slightly outperformed `bfloat16` on `Qwen-Image-2.1` (`37.53 ms` vs. `39.93 ms` at `1024×1024`; `195.65 ms` vs. `204.14 ms` at `2048×2048`).
4. **Synchronous Single-Image HTTP CPU Offload on `TPU v7x` Hosts (Track D):** While remote `c3-standard-44` Keep-Alive offload beat `v5p` host CPUs by `1.08×–1.14×` (`545.0 ms` vs. `613.6 ms` on `2048×2048` WebP), `tpu7x` hosts have `224` Emerald Rapids vCPUs that encode single `2048×2048` WebP images locally in `445.4 ms` vs. `544.9–553.5 ms` over HTTP (`0.80×–0.83×`). On `v7x`, CPU offload is only beneficial for asynchronous host-GIL isolation under concurrent multi-worker serving.
5. **High-Concurrency Queueing at `c=4` in `vllm-omni` (`U=4`):** Increasing per-replica concurrency from `c=2` to `c=4` degraded `1024×1024` `20`-step throughput (`9.51 → 1.72 img/min/replica` on `v5p-16`; `12.17 → 2.24 img/min/replica` on `v7x-4`) due to diffusion batch memory/recompile serialization; `c=2` was optimal across both `v5p` and `v7x`.

---

## 3. Complete Per-Slice Benchmark Tables

### Slice `v5p-16` (`v5p`, `65` Verified Benchmark Points)

- **Raw Log:** [`results/v5p-16/summary.json`](./results/v5p-16/summary.json)

#### Track A — OpenAI-Compatible `vllm-omni` HTTP Serving (`Qwen/Qwen-Image` 20B MMDiT)

| Engine | Precision | `Ulysses × Cluster DP` | Resolution | Steps | Conc | Warmup (s) | Median E2E (s) | P95 E2E (s) | Replica Img/min | Cluster Img/min |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `512x512` | `20` | `1` | `117.12` | `59.900` | `114.816` | `1.00` | `4.01` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `1` | `116.08` | `61.369` | `116.708` | `0.98` | `3.91` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `2` | `116.08` | `9.375` | `12.569` | `9.51` | `38.05` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `4` | `116.08` | `15.990` | `139.569` | `1.72` | `6.88` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1024x1024` | `50` | `1` | `14.83` | `14.616` | `14.623` | `4.09` | `16.37` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1328x1328` | `20` | `1` | `86.63` | `48.276` | `88.989` | `1.24` | `4.96` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1328x1328` | `20` | `2` | `86.63` | `11.785` | `16.032` | `7.45` | `29.80` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1328x1328` | `50` | `1` | `18.22` | `18.257` | `18.283` | `3.27` | `13.09` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1664x928` | `20` | `1` | `123.67` | `65.712` | `124.264` | `0.91` | `3.65` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1664x928` | `20` | `2` | `123.67` | `11.520` | `15.659` | `7.63` | `30.52` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1664x928` | `50` | `1` | `17.40` | `17.536` | `17.753` | `3.41` | `13.63` |
| `vllm-omni-sdpa` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `1` | `10.18` | `7.808` | `10.330` | `7.64` | `30.56` |
| `vllm-omni-sdpa` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `2` | `10.18` | `8.044` | `10.766` | `11.10` | `44.41` |
| `vllm-omni-sdpa` | `BF16` | `U=4 × DP=4` | `1328x1328` | `20` | `1` | `8.29` | `6.428` | `7.496` | `9.22` | `36.89` |

#### Track B — `Qwen-Image-2.1` & `Qwen-Image-2.1-Turbo` (`Qwen3-VL` Text-Encoded Native Pipeline)

| Model Checkpoint | Engine | `TP × Cluster DP` | Resolution | Steps | Weight Load (s) | Median E2E (s) | Pixel Std | Replica Img/min | Cluster Img/min |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `1024x1024` | `8` | `13.33` | `3.384` | `83.17` | `17.73` | `283.68` |
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `1536x1024` | `8` | `13.33` | `5.016` | `84.90` | `11.96` | `191.36` |
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `2048x2048` | `8` | `13.33` | `14.497` | `94.31` | `4.14` | `66.24` |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `1024x1024` | `28` | `13.55` | `10.779` | `81.53` | `5.57` | `89.12` |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `1536x1024` | `28` | `13.55` | `16.355` | `70.32` | `3.67` | `58.72` |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `2048x2048` | `28` | `13.55` | `48.005` | `84.10` | `1.25` | `20.00` |

#### Track C — `TorchTPU` Multi-Stage Breakdown (`Qwen-Image` & `Qwen-Image-2.1`, `BF16` & `FP8` MXU)

| Model | Engine & Kernel | Task & Resolution | Steps | VL Encode (ms) | MMDiT Step (ms) | VAE Decode (ms) | Local PNG (ms) | Disagg PNG (ms) | E2E Disagg (s) | Cluster Img/min |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-512x512-20step` | `20` | `2,487.28` | `48.91` | `2,521.94` | `38.9` | `41.6` | `6.029` | `39.80` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-512x512-50step` | `50` | `5.23` | `44.43` | `3.03` | `38.9` | `41.6` | `2.271` | `105.68` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-20step` | `20` | `5.49` | `123.21` | `2,554.30` | `155.3` | `144.8` | `5.169` | `46.44` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-50step` | `50` | `5.76` | `122.52` | `6.04` | `155.3` | `144.8` | `6.283` | `38.20` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1328x1328-20step` | `20` | `5.49` | `228.30` | `2,683.87` | `261.1` | `247.7` | `7.503` | `32.00` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1328x1328-50step` | `50` | `5.23` | `227.90` | `10.84` | `261.1` | `247.7` | `11.659` | `20.60` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1664x928-20step` | `20` | `5.59` | `194.94` | `3,155.25` | `228.6` | `212.3` | `7.272` | `33.00` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1664x928-50step` | `50` | `5.24` | `195.11` | `8.82` | `228.6` | `212.3` | `9.982` | `24.04` |
| `Qwen/Qwen-Image` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-20step` | `20` | `26.28` | `164.79` | `6.42` | `155.3` | `144.8` | `3.473` | `69.08` |
| `Qwen/Qwen-Image` | `torchtpu-eager (BF16_MXU)` | `T2I-1328x1328-20step` | `20` | `26.79` | `287.52` | `10.62` | `261.1` | `247.7` | `6.036` | `39.76` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-512x512-20step` | `20` | `3,937.31` | `119.29` | `2,407.48` | `38.9` | `41.6` | `8.772` | `27.36` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-512x512-50step` | `50` | `16.88` | `116.29` | `3.06` | `38.9` | `41.6` | `5.876` | `40.84` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-20step` | `20` | `17.50` | `318.31` | `2,615.27` | `155.3` | `144.8` | `9.144` | `26.24` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-50step` | `50` | `16.97` | `317.89` | `6.14` | `155.3` | `144.8` | `16.063` | `14.96` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1328x1328-20step` | `20` | `17.41` | `554.99` | `2,745.46` | `261.1` | `247.7` | `14.110` | `17.00` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1328x1328-50step` | `50` | `16.97` | `555.04` | `10.80` | `261.1` | `247.7` | `28.028` | `8.56` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1664x928-20step` | `20` | `17.42` | `406.26` | `3,081.36` | `228.6` | `212.3` | `11.436` | `21.00` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1664x928-50step` | `50` | `16.94` | `405.58` | `8.71` | `228.6` | `212.3` | `20.517` | `11.68` |
| `Qwen/Qwen-Image` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-20step` | `20` | `45.50` | `405.39` | `6.31` | `155.3` | `144.8` | `8.305` | `28.88` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-8step` | `8` | `3,401.89` | `57.57` | `2,699.28` | `155.3` | `144.8` | `6.707` | `35.80` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `T2I-1536x1024-8step` | `8` | `8.57` | `85.38` | `2,712.08` | `232.4` | `222.3` | `3.626` | `66.20` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `T2I-2048x2048-8step` | `8` | `8.69` | `248.81` | `3,260.28` | `615.2` | `584.0` | `5.843` | `41.08` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1024x1024-8step` | `8` | `3,478.65` | `72.02` | `6.18` | `155.3` | `144.8` | `4.240` | `56.60` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1536x1024-8step` | `8` | `21.42` | `130.22` | `9.25` | `232.4` | `222.3` | `1.329` | `180.56` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-8step` | `8` | `48.21` | `70.65` | `6.37` | `155.3` | `144.8` | `0.765` | `313.88` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (BF16_MXU)` | `T2I-1536x1024-8step` | `8` | `47.41` | `109.71` | `8.77` | `232.4` | `222.3` | `1.156` | `207.56` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-8step` | `8` | `4,855.42` | `146.58` | `2,672.86` | `155.3` | `144.8` | `8.846` | `27.12` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1536x1024-8step` | `8` | `23.42` | `191.83` | `2,649.90` | `232.4` | `222.3` | `4.430` | `54.16` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-2048x2048-8step` | `8` | `22.86` | `537.59` | `3,376.23` | `615.2` | `584.0` | `8.284` | `28.96` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1024x1024-8step` | `8` | `5,140.87` | `183.71` | `6.39` | `155.3` | `144.8` | `6.796` | `35.32` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1536x1024-8step` | `8` | `47.10` | `299.60` | `9.14` | `232.4` | `222.3` | `2.710` | `88.56` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-8step` | `8` | `44.79` | `190.59` | `6.39` | `155.3` | `144.8` | `1.721` | `139.48` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-28step` | `28` | `3,418.93` | `105.86` | `2,759.89` | `155.3` | `144.8` | `9.288` | `25.84` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `T2I-1536x1024-28step` | `28` | `8.42` | `165.76` | `2,730.69` | `232.4` | `222.3` | `7.603` | `31.56` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `T2I-2048x2048-28step` | `28` | `8.68` | `500.88` | `3,214.21` | `615.2` | `584.0` | `17.831` | `13.44` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1024x1024-28step` | `28` | `3,539.28` | `137.22` | `6.14` | `155.3` | `144.8` | `7.567` | `31.72` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `21.41` | `245.45` | `9.32` | `232.4` | `222.3` | `7.160` | `33.52` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-28step` | `28` | `47.96` | `134.74` | `6.31` | `155.3` | `144.8` | `3.972` | `60.44` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (BF16_MXU)` | `T2I-1536x1024-28step` | `28` | `46.84` | `206.00` | `8.72` | `232.4` | `222.3` | `6.046` | `39.68` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-28step` | `28` | `4,843.23` | `314.41` | `2,601.35` | `155.3` | `144.8` | `16.393` | `14.64` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1536x1024-28step` | `28` | `23.30` | `465.08` | `2,793.47` | `232.4` | `222.3` | `16.061` | `14.96` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-2048x2048-28step` | `28` | `23.04` | `1,271.28` | `3,366.60` | `615.2` | `584.0` | `39.569` | `6.08` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1024x1024-28step` | `28` | `5,118.60` | `402.54` | `6.25` | `155.3` | `144.8` | `16.575` | `14.48` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `46.92` | `570.30` | `9.34` | `232.4` | `222.3` | `16.281` | `14.76` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-28step` | `28` | `63.76` | `359.30` | `6.24` | `155.3` | `144.8` | `10.275` | `23.36` |

#### Track D — Disaggregated CPU Image Encoding (`PNG` & `WebP`) Ablation (`POST /encode_image`)

| Resolution & Format | On-Host CPU (ms) | Disagg CPU RTT (ms) | Remote AVX-512 Compute (ms) | Remote Rate (img/s) | Local / Disagg RTT Ratio |
|---|---:|---:|---:|---:|---:|
| `512x512_png` | `38.87` | `41.62` | `34.88` | `28.7` | `0.93x` |
| `512x512_webp` | `38.61` | `41.67` | `38.06` | `26.3` | `0.93x` |
| `1024x1024_png` | `155.28` | `144.85` | `140.96` | `7.1` | `1.07x` |
| `1024x1024_webp` | `155.06` | `144.12` | `135.23` | `7.4` | `1.08x` |
| `1328x1328_png` | `261.07` | `247.68` | `236.78` | `4.2` | `1.05x` |
| `1328x1328_webp` | `261.43` | `239.10` | `228.28` | `4.4` | `1.09x` |
| `1664x928_png` | `228.64` | `212.28` | `207.96` | `4.8` | `1.08x` |
| `1664x928_webp` | `228.54` | `208.14` | `199.02` | `5.0` | `1.10x` |
| `1536x1024_png` | `232.42` | `222.32` | `211.62` | `4.7` | `1.05x` |
| `1536x1024_webp` | `232.90` | `208.74` | `204.20` | `4.9` | `1.12x` |
| `2048x2048_png` | `615.21` | `583.98` | `564.61` | `1.8` | `1.05x` |
| `2048x2048_webp` | `614.81` | `554.44` | `540.62` | `1.8` | `1.11x` |

### Slice `v5p-16-keepalive` (`v5p`, `65` Verified Benchmark Points)

- **Raw Log:** [`results/v5p-16-keepalive/summary.json`](./results/v5p-16-keepalive/summary.json)

#### Track A — OpenAI-Compatible `vllm-omni` HTTP Serving (`Qwen/Qwen-Image` 20B MMDiT)

| Engine | Precision | `Ulysses × Cluster DP` | Resolution | Steps | Conc | Warmup (s) | Median E2E (s) | P95 E2E (s) | Replica Img/min | Cluster Img/min |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `512x512` | `20` | `1` | `116.41` | `60.345` | `115.844` | `0.99` | `3.98` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `1` | `117.08` | `61.504` | `116.994` | `0.97` | `3.90` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `2` | `117.08` | `9.258` | `12.496` | `9.57` | `38.27` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `4` | `117.08` | `16.073` | `144.873` | `1.66` | `6.62` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1024x1024` | `50` | `1` | `14.64` | `14.571` | `14.576` | `4.11` | `16.42` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1328x1328` | `20` | `1` | `88.12` | `46.520` | `85.525` | `1.29` | `5.15` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1328x1328` | `20` | `2` | `88.12` | `11.399` | `15.338` | `7.79` | `31.15` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1328x1328` | `50` | `1` | `17.98` | `18.172` | `18.226` | `3.29` | `13.15` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1664x928` | `20` | `1` | `127.20` | `64.846` | `122.668` | `0.92` | `3.70` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1664x928` | `20` | `2` | `127.20` | `10.331` | `13.708` | `8.71` | `34.85` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1664x928` | `50` | `1` | `16.79` | `16.992` | `17.099` | `3.52` | `14.07` |
| `vllm-omni-sdpa` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `1` | `9.21` | `6.621` | `9.148` | `9.00` | `36.01` |
| `vllm-omni-sdpa` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `2` | `9.21` | `5.830` | `7.891` | `15.12` | `60.50` |
| `vllm-omni-sdpa` | `BF16` | `U=4 × DP=4` | `1328x1328` | `20` | `1` | `7.29` | `5.623` | `7.034` | `10.52` | `42.08` |

#### Track B — `Qwen-Image-2.1` & `Qwen-Image-2.1-Turbo` (`Qwen3-VL` Text-Encoded Native Pipeline)

| Model Checkpoint | Engine | `TP × Cluster DP` | Resolution | Steps | Weight Load (s) | Median E2E (s) | Pixel Std | Replica Img/min | Cluster Img/min |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `1024x1024` | `8` | `11.94` | `3.299` | `82.45` | `18.19` | `291.04` |
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `1536x1024` | `8` | `11.94` | `4.951` | `74.66` | `12.12` | `193.92` |
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `2048x2048` | `8` | `11.94` | `14.414` | `96.06` | `4.16` | `66.56` |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `1024x1024` | `28` | `11.11` | `10.711` | `77.99` | `5.60` | `89.60` |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `1536x1024` | `28` | `11.11` | `16.180` | `76.08` | `3.71` | `59.36` |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `2048x2048` | `28` | `11.11` | `47.894` | `83.47` | `1.25` | `20.00` |

#### Track C — `TorchTPU` Multi-Stage Breakdown (`Qwen-Image` & `Qwen-Image-2.1`, `BF16` & `FP8` MXU)

| Model | Engine & Kernel | Task & Resolution | Steps | VL Encode (ms) | MMDiT Step (ms) | VAE Decode (ms) | Local PNG (ms) | Disagg PNG (ms) | E2E Disagg (s) | Cluster Img/min |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-512x512-20step` | `20` | `2,455.61` | `44.79` | `2,567.28` | `39.0` | `39.0` | `5.958` | `40.28` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-512x512-50step` | `50` | `5.22` | `44.00` | `2.95` | `39.0` | `39.0` | `2.247` | `106.80` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-20step` | `20` | `5.24` | `123.18` | `2,779.49` | `155.8` | `157.8` | `5.406` | `44.40` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-50step` | `50` | `5.24` | `121.70` | `6.08` | `155.8` | `157.8` | `6.254` | `38.36` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1328x1328-20step` | `20` | `5.26` | `226.71` | `2,801.85` | `262.5` | `241.5` | `7.583` | `31.64` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1328x1328-50step` | `50` | `5.20` | `226.38` | `10.63` | `262.5` | `241.5` | `11.576` | `20.72` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1664x928-20step` | `20` | `5.45` | `195.06` | `3,041.97` | `229.8` | `212.1` | `7.161` | `33.52` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1664x928-50step` | `50` | `5.18` | `193.12` | `8.64` | `229.8` | `212.1` | `9.882` | `24.28` |
| `Qwen/Qwen-Image` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-20step` | `20` | `15.87` | `165.14` | `6.33` | `155.8` | `157.8` | `3.483` | `68.92` |
| `Qwen/Qwen-Image` | `torchtpu-eager (BF16_MXU)` | `T2I-1328x1328-20step` | `20` | `16.39` | `287.65` | `10.70` | `262.5` | `241.5` | `6.022` | `39.84` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-512x512-20step` | `20` | `3,637.35` | `116.99` | `2,415.00` | `39.0` | `39.0` | `8.431` | `28.48` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-512x512-50step` | `50` | `16.88` | `115.88` | `2.82` | `39.0` | `39.0` | `5.853` | `41.00` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-20step` | `20` | `16.96` | `317.91` | `2,670.12` | `155.8` | `157.8` | `9.203` | `26.08` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-50step` | `50` | `16.91` | `316.76` | `6.03` | `155.8` | `157.8` | `16.019` | `15.00` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1328x1328-20step` | `20` | `16.96` | `553.83` | `2,862.03` | `262.5` | `241.5` | `14.197` | `16.92` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1328x1328-50step` | `50` | `16.84` | `554.31` | `10.81` | `262.5` | `241.5` | `27.985` | `8.56` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1664x928-20step` | `20` | `17.02` | `406.93` | `3,029.72` | `229.8` | `212.1` | `11.397` | `21.04` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1664x928-50step` | `50` | `16.92` | `405.81` | `8.76` | `229.8` | `212.1` | `20.528` | `11.68` |
| `Qwen/Qwen-Image` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-20step` | `20` | `21.71` | `403.32` | `6.22` | `155.8` | `157.8` | `8.252` | `29.08` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-8step` | `8` | `3,331.14` | `57.06` | `2,781.34` | `155.8` | `157.8` | `6.727` | `35.68` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `T2I-1536x1024-8step` | `8` | `8.37` | `86.10` | `2,716.55` | `233.7` | `215.2` | `3.629` | `66.12` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `T2I-2048x2048-8step` | `8` | `8.47` | `248.48` | `3,447.70` | `617.0` | `572.1` | `6.016` | `39.88` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1024x1024-8step` | `8` | `3,424.71` | `72.11` | `6.12` | `155.8` | `157.8` | `4.191` | `57.28` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1536x1024-8step` | `8` | `21.30` | `129.77` | `8.73` | `233.7` | `215.2` | `1.309` | `183.36` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-8step` | `8` | `24.72` | `69.72` | `6.28` | `155.8` | `157.8` | `0.747` | `321.44` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (BF16_MXU)` | `T2I-1536x1024-8step` | `8` | `18.49` | `108.63` | `8.72` | `233.7` | `215.2` | `1.111` | `215.96` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-8step` | `8` | `4,590.22` | `146.95` | `2,814.72` | `155.8` | `157.8` | `8.738` | `27.48` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1536x1024-8step` | `8` | `22.93` | `191.27` | `2,881.84` | `233.7` | `215.2` | `4.650` | `51.60` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-2048x2048-8step` | `8` | `22.79` | `536.67` | `3,372.77` | `617.0` | `572.1` | `8.261` | `29.04` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1024x1024-8step` | `8` | `4,893.07` | `185.96` | `6.37` | `155.8` | `157.8` | `6.571` | `36.52` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1536x1024-8step` | `8` | `46.95` | `299.16` | `8.87` | `233.7` | `215.2` | `2.690` | `89.24` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-8step` | `8` | `36.17` | `190.09` | `6.13` | `155.8` | `157.8` | `1.721` | `139.48` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-28step` | `28` | `3,268.52` | `105.01` | `2,653.59` | `155.8` | `157.8` | `9.020` | `26.60` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `T2I-1536x1024-28step` | `28` | `8.36` | `164.97` | `2,794.61` | `233.7` | `215.2` | `7.637` | `31.44` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `T2I-2048x2048-28step` | `28` | `8.36` | `499.12` | `3,244.19` | `617.0` | `572.1` | `17.800` | `13.48` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1024x1024-28step` | `28` | `3,400.43` | `136.51` | `6.02` | `155.8` | `157.8` | `7.412` | `32.36` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `21.23` | `244.63` | `8.74` | `233.7` | `215.2` | `7.120` | `33.72` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-28step` | `28` | `26.42` | `134.41` | `6.27` | `155.8` | `157.8` | `3.954` | `60.68` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (BF16_MXU)` | `T2I-1536x1024-28step` | `28` | `25.35` | `205.55` | `8.68` | `233.7` | `215.2` | `6.005` | `39.96` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-28step` | `28` | `4,564.53` | `314.78` | `2,781.37` | `155.8` | `157.8` | `16.318` | `14.72` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1536x1024-28step` | `28` | `23.00` | `465.72` | `2,857.56` | `233.7` | `215.2` | `16.136` | `14.88` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-2048x2048-28step` | `28` | `22.87` | `1,269.66` | `3,448.97` | `617.0` | `572.1` | `39.594` | `6.08` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1024x1024-28step` | `28` | `4,684.67` | `402.74` | `6.20` | `155.8` | `157.8` | `16.151` | `14.84` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `46.84` | `569.98` | `8.85` | `233.7` | `215.2` | `16.256` | `14.76` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-28step` | `28` | `33.23` | `358.46` | `6.30` | `155.8` | `157.8` | `10.234` | `23.44` |

#### Track D — Disaggregated CPU Image Encoding (`PNG` & `WebP`) Ablation (`POST /encode_image`)

| Resolution & Format | On-Host CPU (ms) | Disagg CPU RTT (ms) | Remote AVX-512 Compute (ms) | Remote Rate (img/s) | Local / Disagg RTT Ratio |
|---|---:|---:|---:|---:|---:|
| `512x512_png` | `39.05` | `38.97` | `36.68` | `27.3` | `1.00x` |
| `512x512_webp` | `39.06` | `36.27` | `34.13` | `29.3` | `1.08x` |
| `1024x1024_png` | `155.83` | `157.85` | `140.10` | `7.1` | `0.99x` |
| `1024x1024_webp` | `155.57` | `138.47` | `135.60` | `7.4` | `1.12x` |
| `1328x1328_png` | `262.52` | `241.47` | `236.85` | `4.2` | `1.09x` |
| `1328x1328_webp` | `262.71` | `233.05` | `228.49` | `4.4` | `1.13x` |
| `1664x928_png` | `229.78` | `212.06` | `207.64` | `4.8` | `1.08x` |
| `1664x928_webp` | `229.50` | `203.42` | `199.08` | `5.0` | `1.13x` |
| `1536x1024_png` | `233.67` | `215.15` | `210.83` | `4.7` | `1.09x` |
| `1536x1024_webp` | `233.77` | `208.15` | `203.68` | `4.9` | `1.12x` |
| `2048x2048_png` | `617.05` | `572.14` | `564.25` | `1.8` | `1.08x` |
| `2048x2048_webp` | `618.71` | `549.99` | `541.30` | `1.8` | `1.12x` |

### Slice `v5p-16-adaptive-64k` (`v5p`, `5` Verified Benchmark Points)

- **Raw Log:** [`results/v5p-16-adaptive-64k/summary.json`](./results/v5p-16-adaptive-64k/summary.json)

#### Track A — OpenAI-Compatible `vllm-omni` HTTP Serving (`Qwen/Qwen-Image` 20B MMDiT)

| Engine | Precision | `Ulysses × Cluster DP` | Resolution | Steps | Conc | Warmup (s) | Median E2E (s) | P95 E2E (s) | Replica Img/min | Cluster Img/min |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `512x512` | `20` | `1` | `117.22` | `62.261` | `119.606` | `0.96` | `3.85` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `1` | `118.32` | `63.087` | `120.096` | `0.95` | `3.80` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `2` | `118.32` | `9.367` | `12.653` | `9.45` | `37.81` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `4` | `118.32` | `75.161` | `144.301` | `1.66` | `6.65` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1024x1024` | `50` | `1` | `14.68` | `14.769` | `14.800` | `4.05` | `16.20` |

#### Track D — Disaggregated CPU Image Encoding (`PNG` & `WebP`) Ablation (`POST /encode_image`)

| Resolution & Format | On-Host CPU (ms) | Disagg CPU RTT (ms) | Remote AVX-512 Compute (ms) | Remote Rate (img/s) | Local / Disagg RTT Ratio |
|---|---:|---:|---:|---:|---:|
| `512x512_png` | `38.67` | `35.78` | `34.96` | `28.6` | `1.08x` |
| `512x512_webp` | `38.47` | `34.66` | `33.90` | `29.5` | `1.11x` |
| `1024x1024_png` | `155.83` | `140.96` | `139.76` | `7.2` | `1.11x` |
| `1024x1024_webp` | `154.94` | `136.47` | `135.17` | `7.4` | `1.14x` |
| `1328x1328_png` | `261.05` | `238.98` | `236.55` | `4.2` | `1.09x` |
| `1328x1328_webp` | `260.52` | `229.98` | `227.72` | `4.4` | `1.13x` |
| `1664x928_png` | `227.90` | `209.06` | `206.99` | `4.8` | `1.09x` |
| `1664x928_webp` | `227.88` | `201.15` | `199.02` | `5.0` | `1.13x` |
| `1536x1024_png` | `231.73` | `212.40` | `210.42` | `4.8` | `1.09x` |
| `1536x1024_webp` | `232.21` | `205.08` | `203.10` | `4.9` | `1.13x` |
| `2048x2048_png` | `625.72` | `567.54` | `563.06` | `1.8` | `1.10x` |
| `2048x2048_webp` | `613.55` | `545.01` | `540.20` | `1.9` | `1.13x` |

### Slice `v7x-16` (`v7x`, `88` Verified Benchmark Points)

- **Raw Log:** [`results/v7x-16/summary.json`](./results/v7x-16/summary.json)

#### Track A — OpenAI-Compatible `vllm-omni` HTTP Serving (`Qwen/Qwen-Image` 20B MMDiT)

| Engine | Precision | `Ulysses × Cluster DP` | Resolution | Steps | Conc | Warmup (s) | Median E2E (s) | P95 E2E (s) | Replica Img/min | Cluster Img/min |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=8` | `512x512` | `20` | `1` | `73.16` | `38.717` | `73.377` | `1.55` | `12.39` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=8` | `1024x1024` | `20` | `1` | `106.75` | `46.700` | `88.654` | `1.28` | `10.27` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=8` | `1024x1024` | `20` | `2` | `106.75` | `7.553` | `10.193` | `11.73` | `93.84` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=8` | `1024x1024` | `20` | `4` | `106.75` | `11.845` | `105.826` | `2.27` | `18.14` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=8` | `1024x1024` | `50` | `1` | `11.33` | `11.294` | `11.295` | `5.30` | `42.37` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=8` | `1328x1328` | `20` | `1` | `96.26` | `36.893` | `68.091` | `1.62` | `12.99` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=8` | `1328x1328` | `20` | `2` | `96.26` | `8.906` | `11.998` | `9.95` | `79.58` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=8` | `1328x1328` | `50` | `1` | `13.61` | `13.662` | `13.702` | `4.37` | `34.97` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=8` | `1664x928` | `20` | `1` | `120.83` | `53.985` | `102.726` | `1.11` | `8.88` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=8` | `1664x928` | `20` | `2` | `120.83` | `7.822` | `10.365` | `11.51` | `92.09` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=8` | `1664x928` | `50` | `1` | `12.64` | `12.550` | `12.623` | `4.76` | `38.07` |
| `vllm-omni-sdpa` | `BF16` | `U=4 × DP=8` | `1024x1024` | `20` | `1` | `9.02` | `6.054` | `8.737` | `9.85` | `78.81` |
| `vllm-omni-sdpa` | `BF16` | `U=4 × DP=8` | `1024x1024` | `20` | `2` | `9.02` | `5.116` | `6.843` | `17.44` | `139.50` |
| `vllm-omni-sdpa` | `BF16` | `U=4 × DP=8` | `1328x1328` | `20` | `1` | `8.01` | `5.794` | `7.716` | `10.24` | `81.91` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=4` | `512x512` | `20` | `1` | `10.11` | `6.859` | `10.055` | `8.73` | `34.93` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=4` | `1024x1024` | `20` | `1` | `46.13` | `7.025` | `10.426` | `8.49` | `33.97` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=4` | `1024x1024` | `20` | `2` | `46.13` | `5.989` | `8.092` | `14.76` | `59.05` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=4` | `1024x1024` | `20` | `4` | `46.13` | `9.655` | `25.832` | `9.28` | `37.10` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=4` | `1024x1024` | `50` | `1` | `8.67` | `8.716` | `8.830` | `6.85` | `27.41` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=4` | `1328x1328` | `20` | `1` | `62.53` | `7.970` | `12.040` | `7.47` | `29.87` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=4` | `1328x1328` | `20` | `2` | `62.53` | `6.044` | `8.219` | `14.48` | `57.93` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=4` | `1328x1328` | `50` | `1` | `9.23` | `10.034` | `10.227` | `5.94` | `23.76` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=4` | `1664x928` | `20` | `1` | `54.78` | `9.034` | `14.213` | `6.60` | `26.40` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=4` | `1664x928` | `20` | `2` | `54.78` | `5.610` | `7.414` | `16.06` | `64.24` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=4` | `1664x928` | `50` | `1` | `9.28` | `9.366` | `9.468` | `6.37` | `25.47` |

#### Track B — `Qwen-Image-2.1` & `Qwen-Image-2.1-Turbo` (`Qwen3-VL` Text-Encoded Native Pipeline)

| Model Checkpoint | Engine | `TP × Cluster DP` | Resolution | Steps | Weight Load (s) | Median E2E (s) | Pixel Std | Replica Img/min | Cluster Img/min |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=32` | `1024x1024` | `8` | `10.84` | `3.595` | `90.06` | `16.69` | `534.08` |
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=32` | `1536x1024` | `8` | `10.84` | `5.378` | `80.43` | `11.16` | `357.12` |
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=32` | `2048x2048` | `8` | `10.84` | `15.762` | `86.05` | `3.81` | `121.92` |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=32` | `1024x1024` | `28` | `11.97` | `11.919` | `76.67` | `5.03` | `160.96` |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=32` | `1536x1024` | `28` | `11.97` | `17.848` | `86.87` | `3.36` | `107.52` |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=32` | `2048x2048` | `28` | `11.97` | `52.948` | `79.03` | `1.13` | `36.16` |

#### Track C — `TorchTPU` Multi-Stage Breakdown (`Qwen-Image` & `Qwen-Image-2.1`, `BF16` & `FP8` MXU)

| Model | Engine & Kernel | Task & Resolution | Steps | VL Encode (ms) | MMDiT Step (ms) | VAE Decode (ms) | Local PNG (ms) | Disagg PNG (ms) | E2E Disagg (s) | Cluster Img/min |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-50step` | `50` | `0.95` | `52.02` | `5.05` | `117.9` | `149.9` | `2.757` | `174.16` |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1328x1328-50step` | `50` | `1.00` | `61.45` | `30.50` | `189.8` | `241.9` | `3.346` | `143.44` |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1664x928-20step` | `20` | `1.07` | `50.73` | `8,784.34` | `184.1` | `212.7` | `10.013` | `47.92` |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1664x928-50step` | `50` | `0.93` | `55.38` | `75.39` | `184.1` | `212.7` | `3.058` | `156.96` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-512x512-20step` | `20` | `1,689.32` | `20.07` | `5,576.75` | `28.1` | `41.4` | `7.709` | `31.12` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-512x512-50step` | `50` | `2.39` | `18.25` | `6.73` | `28.1` | `41.4` | `0.963` | `249.28` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-20step` | `20` | `2.35` | `52.28` | `6,303.53` | `117.9` | `149.9` | `7.501` | `32.00` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-50step` | `50` | `2.54` | `50.16` | `13.56` | `117.9` | `149.9` | `2.674` | `89.76` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1328x1328-20step` | `20` | `2.84` | `105.36` | `6,077.44` | `189.8` | `241.9` | `8.429` | `28.48` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1328x1328-50step` | `50` | `2.77` | `102.27` | `23.77` | `189.8` | `241.9` | `5.382` | `44.60` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1664x928-20step` | `20` | `2.51` | `89.02` | `8,377.05` | `184.1` | `212.7` | `10.373` | `23.12` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1664x928-50step` | `50` | `2.81` | `89.87` | `20.71` | `184.1` | `212.7` | `4.730` | `50.76` |
| `Qwen/Qwen-Image` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-20step` | `20` | `12.42` | `78.21` | `15.79` | `117.9` | `149.9` | `1.742` | `137.76` |
| `Qwen/Qwen-Image` | `torchtpu-eager (BF16_MXU)` | `T2I-1328x1328-20step` | `20` | `10.82` | `138.65` | `26.56` | `189.8` | `241.9` | `3.052` | `78.64` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-512x512-20step` | `20` | `2,182.76` | `19.68` | `5,203.71` | `28.1` | `41.4` | `7.821` | `30.68` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-512x512-50step` | `50` | `1.68` | `18.52` | `6.69` | `28.1` | `41.4` | `0.976` | `245.96` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-20step` | `20` | `1.78` | `52.10` | `5,922.07` | `117.9` | `149.9` | `7.116` | `33.72` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-50step` | `50` | `2.21` | `51.85` | `14.01` | `117.9` | `149.9` | `2.758` | `87.00` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1328x1328-20step` | `20` | `2.01` | `111.17` | `6,308.49` | `189.8` | `241.9` | `8.776` | `27.36` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1328x1328-50step` | `50` | `1.81` | `107.09` | `23.05` | `189.8` | `241.9` | `5.621` | `42.68` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1664x928-20step` | `20` | `2.41` | `88.91` | `8,178.09` | `184.1` | `212.7` | `10.171` | `23.60` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1664x928-50step` | `50` | `2.05` | `89.07` | `20.58` | `184.1` | `212.7` | `4.689` | `51.20` |
| `Qwen/Qwen-Image` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-20step` | `20` | `17.60` | `70.45` | `15.86` | `117.9` | `149.9` | `1.592` | `150.72` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-8step` | `8` | `20.06` | `25.78` | `0.82` | `117.9` | `149.9` | `0.377` | `1,273.12` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (BF16_MXU)` | `T2I-1536x1024-8step` | `8` | `15.16` | `26.77` | `1.15` | `168.6` | `218.9` | `0.449` | `1,068.00` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-8step` | `8` | `18.37` | `33.88` | `0.85` | `117.9` | `149.9` | `0.440` | `1,090.72` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-8step` | `8` | `2,303.38` | `22.02` | `6,278.58` | `117.9` | `149.9` | `8.908` | `26.96` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `T2I-1536x1024-8step` | `8` | `2.76` | `35.98` | `5,963.76` | `168.6` | `218.9` | `6.473` | `37.08` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `T2I-2048x2048-8step` | `8` | `2.75` | `103.93` | `6,931.75` | `447.2` | `571.8` | `8.338` | `28.80` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1024x1024-8step` | `8` | `2,250.84` | `28.75` | `13.89` | `117.9` | `149.9` | `2.678` | `89.64` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1536x1024-8step` | `8` | `7.02` | `54.90` | `19.77` | `168.6` | `218.9` | `0.718` | `334.32` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-8step` | `8` | `15.86` | `30.61` | `15.97` | `117.9` | `149.9` | `0.427` | `562.64` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (BF16_MXU)` | `T2I-1536x1024-8step` | `8` | `13.70` | `47.10` | `22.85` | `168.6` | `218.9` | `0.632` | `379.60` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-8step` | `8` | `2,840.03` | `18.37` | `6,222.83` | `117.9` | `149.9` | `9.360` | `25.64` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1536x1024-8step` | `8` | `2.95` | `29.95` | `6,103.55` | `168.6` | `218.9` | `6.565` | `36.56` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-2048x2048-8step` | `8` | `3.14` | `90.95` | `6,983.46` | `447.2` | `571.8` | `8.286` | `28.96` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1024x1024-8step` | `8` | `2,939.14` | `25.77` | `14.42` | `117.9` | `149.9` | `3.342` | `71.80` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1536x1024-8step` | `8` | `5.05` | `53.00` | `19.39` | `168.6` | `218.9` | `0.700` | `342.72` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-8step` | `8` | `20.23` | `36.48` | `16.19` | `117.9` | `149.9` | `0.478` | `501.96` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `1.18` | `25.76` | `127.94` | `168.6` | `218.9` | `1.102` | `435.52` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-28step` | `28` | `16.33` | `26.05` | `1.31` | `117.9` | `149.9` | `0.897` | `535.20` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (BF16_MXU)` | `T2I-1536x1024-28step` | `28` | `14.77` | `29.27` | `1.31` | `168.6` | `218.9` | `1.055` | `455.20` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `0.84` | `33.85` | `109.79` | `168.6` | `218.9` | `1.310` | `366.32` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-28step` | `28` | `21.10` | `45.33` | `0.96` | `117.9` | `149.9` | `1.441` | `333.12` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-28step` | `28` | `2,353.85` | `39.93` | `6,065.42` | `117.9` | `149.9` | `9.687` | `24.76` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `T2I-1536x1024-28step` | `28` | `3.18` | `72.99` | `5,795.39` | `168.6` | `218.9` | `8.061` | `29.76` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `T2I-2048x2048-28step` | `28` | `2.99` | `204.14` | `6,835.34` | `447.2` | `571.8` | `13.126` | `18.28` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1024x1024-28step` | `28` | `2,293.80` | `54.65` | `13.36` | `117.9` | `149.9` | `4.020` | `59.68` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `6.94` | `100.65` | `19.76` | `168.6` | `218.9` | `3.097` | `77.52` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-28step` | `28` | `16.29` | `54.99` | `15.61` | `117.9` | `149.9` | `1.722` | `139.40` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (BF16_MXU)` | `T2I-1536x1024-28step` | `28` | `14.62` | `84.19` | `23.12` | `168.6` | `218.9` | `2.614` | `91.80` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-28step` | `28` | `2,882.26` | `37.53` | `5,680.62` | `117.9` | `149.9` | `9.764` | `24.60` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1536x1024-28step` | `28` | `2.50` | `66.02` | `6,186.25` | `168.6` | `218.9` | `8.256` | `29.08` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-2048x2048-28step` | `28` | `2.82` | `196.31` | `6,689.88` | `447.2` | `571.8` | `12.761` | `18.80` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1024x1024-28step` | `28` | `3,034.54` | `50.91` | `13.72` | `117.9` | `149.9` | `4.656` | `51.56` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `5.23` | `92.24` | `19.66` | `168.6` | `218.9` | `2.859` | `83.92` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-28step` | `28` | `23.91` | `48.10` | `16.20` | `117.9` | `149.9` | `1.537` | `156.16` |

#### Track D — Disaggregated CPU Image Encoding (`PNG` & `WebP`) Ablation (`POST /encode_image`)

| Resolution & Format | On-Host CPU (ms) | Disagg CPU RTT (ms) | Remote AVX-512 Compute (ms) | Remote Rate (img/s) | Local / Disagg RTT Ratio |
|---|---:|---:|---:|---:|---:|
| `512x512_png` | `28.13` | `41.35` | `35.16` | `28.4` | `0.68x` |
| `512x512_webp` | `28.35` | `41.69` | `34.24` | `29.2` | `0.68x` |
| `1024x1024_png` | `117.87` | `149.86` | `140.70` | `7.1` | `0.79x` |
| `1024x1024_webp` | `113.33` | `140.18` | `135.48` | `7.4` | `0.81x` |
| `1328x1328_png` | `189.75` | `241.87` | `237.13` | `4.2` | `0.78x` |
| `1328x1328_webp` | `199.65` | `239.25` | `228.17` | `4.4` | `0.83x` |
| `1664x928_png` | `184.07` | `212.65` | `207.56` | `4.8` | `0.87x` |
| `1664x928_webp` | `167.63` | `203.20` | `198.84` | `5.0` | `0.82x` |
| `1536x1024_png` | `168.57` | `218.94` | `211.17` | `4.7` | `0.77x` |
| `1536x1024_webp` | `168.93` | `212.11` | `202.61` | `4.9` | `0.80x` |
| `2048x2048_png` | `447.21` | `571.82` | `563.83` | `1.8` | `0.78x` |
| `2048x2048_webp` | `445.36` | `553.49` | `539.54` | `1.9` | `0.80x` |

### Slice `v7x-8` (`v7x`, `88` Verified Benchmark Points)

- **Raw Log:** [`results/v7x-8/summary.json`](./results/v7x-8/summary.json)

#### Track A — OpenAI-Compatible `vllm-omni` HTTP Serving (`Qwen/Qwen-Image` 20B MMDiT)

| Engine | Precision | `Ulysses × Cluster DP` | Resolution | Steps | Conc | Warmup (s) | Median E2E (s) | P95 E2E (s) | Replica Img/min | Cluster Img/min |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `512x512` | `20` | `1` | `71.94` | `39.064` | `74.083` | `1.54` | `6.14` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `1` | `106.84` | `45.506` | `86.294` | `1.32` | `5.27` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `2` | `106.84` | `7.983` | `10.800` | `11.07` | `44.30` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `4` | `106.84` | `12.962` | `107.870` | `2.22` | `8.90` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1024x1024` | `50` | `1` | `11.89` | `11.318` | `11.324` | `5.28` | `21.13` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1328x1328` | `20` | `1` | `95.69` | `37.338` | `68.984` | `1.60` | `6.42` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1328x1328` | `20` | `2` | `95.69` | `8.861` | `11.947` | `9.99` | `39.95` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1328x1328` | `50` | `1` | `13.62` | `13.720` | `13.768` | `4.35` | `17.41` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1664x928` | `20` | `1` | `117.79` | `53.919` | `102.563` | `1.11` | `4.45` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1664x928` | `20` | `2` | `117.79` | `8.087` | `10.938` | `10.92` | `43.66` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `1664x928` | `50` | `1` | `12.59` | `12.602` | `12.687` | `4.74` | `18.96` |
| `vllm-omni-sdpa` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `1` | `9.45` | `6.114` | `8.826` | `9.75` | `39.01` |
| `vllm-omni-sdpa` | `BF16` | `U=4 × DP=4` | `1024x1024` | `20` | `2` | `9.45` | `5.204` | `6.860` | `17.40` | `69.60` |
| `vllm-omni-sdpa` | `BF16` | `U=4 × DP=4` | `1328x1328` | `20` | `1` | `7.99` | `5.829` | `7.837` | `10.18` | `40.74` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=2` | `512x512` | `20` | `1` | `10.40` | `7.066` | `10.324` | `8.48` | `16.96` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=2` | `1024x1024` | `20` | `1` | `48.57` | `7.398` | `10.597` | `8.07` | `16.13` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=2` | `1024x1024` | `20` | `2` | `48.57` | `6.211` | `8.473` | `14.09` | `28.18` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=2` | `1024x1024` | `20` | `4` | `48.57` | `9.781` | `25.747` | `9.30` | `18.61` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=2` | `1024x1024` | `50` | `1` | `8.93` | `9.158` | `9.174` | `6.52` | `13.05` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=2` | `1328x1328` | `20` | `1` | `59.42` | `8.480` | `12.711` | `7.02` | `14.04` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=2` | `1328x1328` | `20` | `2` | `59.42` | `5.953` | `8.086` | `14.72` | `29.44` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=2` | `1328x1328` | `50` | `1` | `9.87` | `9.595` | `9.658` | `6.21` | `12.42` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=2` | `1664x928` | `20` | `1` | `53.82` | `9.505` | `14.679` | `6.28` | `12.55` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=2` | `1664x928` | `20` | `2` | `53.82` | `6.751` | `9.157` | `13.02` | `26.04` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=2` | `1664x928` | `50` | `1` | `10.42` | `10.240` | `10.298` | `5.83` | `11.66` |

#### Track B — `Qwen-Image-2.1` & `Qwen-Image-2.1-Turbo` (`Qwen3-VL` Text-Encoded Native Pipeline)

| Model Checkpoint | Engine | `TP × Cluster DP` | Resolution | Steps | Weight Load (s) | Median E2E (s) | Pixel Std | Replica Img/min | Cluster Img/min |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `1024x1024` | `8` | `13.97` | `3.600` | `85.67` | `16.67` | `266.72` |
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `1536x1024` | `8` | `13.97` | `5.370` | `82.75` | `11.17` | `178.72` |
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `2048x2048` | `8` | `13.97` | `15.787` | `86.30` | `3.80` | `60.80` |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `1024x1024` | `28` | `11.51` | `11.893` | `78.81` | `5.05` | `80.80` |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `1536x1024` | `28` | `11.51` | `17.861` | `81.94` | `3.36` | `53.76` |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=16` | `2048x2048` | `28` | `11.51` | `52.990` | `75.48` | `1.13` | `18.08` |

#### Track C — `TorchTPU` Multi-Stage Breakdown (`Qwen-Image` & `Qwen-Image-2.1`, `BF16` & `FP8` MXU)

| Model | Engine & Kernel | Task & Resolution | Steps | VL Encode (ms) | MMDiT Step (ms) | VAE Decode (ms) | Local PNG (ms) | Disagg PNG (ms) | E2E Disagg (s) | Cluster Img/min |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-50step` | `50` | `0.98` | `63.51` | `0.90` | `114.0` | `142.2` | `3.320` | `72.28` |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1328x1328-50step` | `50` | `1.13` | `64.07` | `29.20` | `190.4` | `240.2` | `3.474` | `69.08` |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1664x928-20step` | `20` | `0.98` | `65.79` | `8,278.49` | `166.4` | `209.8` | `9.805` | `24.48` |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1664x928-50step` | `50` | `1.00` | `64.02` | `20.68` | `166.4` | `209.8` | `3.432` | `69.92` |
| `Qwen/Qwen-Image` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-20step` | `20` | `16.67` | `77.08` | `1.00` | `114.0` | `142.2` | `1.701` | `141.04` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-512x512-20step` | `20` | `1,727.05` | `19.59` | `5,629.67` | `28.3` | `36.4` | `7.785` | `15.42` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-512x512-50step` | `50` | `2.44` | `19.16` | `6.47` | `28.3` | `36.4` | `1.003` | `119.62` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-20step` | `20` | `2.88` | `51.76` | `5,763.93` | `114.0` | `142.2` | `6.944` | `17.28` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-50step` | `50` | `2.43` | `51.08` | `13.54` | `114.0` | `142.2` | `2.712` | `44.24` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1328x1328-20step` | `20` | `2.80` | `106.21` | `6,427.27` | `190.4` | `240.2` | `8.794` | `13.64` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1328x1328-50step` | `50` | `2.55` | `106.14` | `23.55` | `190.4` | `240.2` | `5.573` | `21.54` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1664x928-20step` | `20` | `2.46` | `89.72` | `8,392.99` | `166.4` | `209.8` | `10.400` | `11.54` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1664x928-50step` | `50` | `2.41` | `88.89` | `19.63` | `166.4` | `209.8` | `4.677` | `25.66` |
| `Qwen/Qwen-Image` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-20step` | `20` | `13.22` | `77.39` | `15.75` | `114.0` | `142.2` | `1.719` | `69.82` |
| `Qwen/Qwen-Image` | `torchtpu-eager (BF16_MXU)` | `T2I-1328x1328-20step` | `20` | `11.62` | `137.95` | `27.15` | `190.4` | `240.2` | `3.038` | `39.50` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-512x512-20step` | `20` | `2,250.65` | `19.78` | `5,236.68` | `28.3` | `36.4` | `7.919` | `15.16` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-512x512-50step` | `50` | `1.85` | `18.37` | `6.79` | `28.3` | `36.4` | `0.963` | `124.56` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-20step` | `20` | `2.10` | `53.00` | `5,903.98` | `114.0` | `142.2` | `7.108` | `16.88` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-50step` | `50` | `1.86` | `53.19` | `13.92` | `114.0` | `142.2` | `2.817` | `42.60` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1328x1328-20step` | `20` | `2.54` | `107.59` | `6,915.12` | `190.4` | `240.2` | `9.310` | `12.88` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1328x1328-50step` | `50` | `2.08` | `108.11` | `22.83` | `190.4` | `240.2` | `5.671` | `21.16` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1664x928-20step` | `20` | `2.07` | `90.71` | `8,383.90` | `166.4` | `209.8` | `10.410` | `11.52` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1664x928-50step` | `50` | `2.42` | `90.34` | `19.77` | `166.4` | `209.8` | `4.749` | `25.28` |
| `Qwen/Qwen-Image` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-20step` | `20` | `18.05` | `72.97` | `15.85` | `114.0` | `142.2` | `1.635` | `73.38` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-8step` | `8` | `14.37` | `28.02` | `1.08` | `114.0` | `142.2` | `0.382` | `628.60` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (BF16_MXU)` | `T2I-1536x1024-8step` | `8` | `14.00` | `24.53` | `1.11` | `168.5` | `213.4` | `0.425` | `565.08` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-8step` | `8` | `21.94` | `43.12` | `1.16` | `114.0` | `142.2` | `0.510` | `470.40` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-8step` | `8` | `2,349.32` | `21.49` | `5,760.81` | `114.0` | `142.2` | `8.424` | `14.24` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `T2I-1536x1024-8step` | `8` | `2.94` | `35.27` | `6,418.44` | `168.5` | `213.4` | `6.917` | `17.34` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `T2I-2048x2048-8step` | `8` | `3.38` | `102.11` | `6,939.72` | `449.5` | `571.7` | `8.332` | `14.40` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1024x1024-8step` | `8` | `2,305.26` | `28.54` | `13.78` | `114.0` | `142.2` | `2.713` | `44.24` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1536x1024-8step` | `8` | `6.89` | `56.18` | `19.44` | `168.5` | `213.4` | `0.712` | `168.48` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-8step` | `8` | `17.79` | `31.27` | `15.69` | `114.0` | `142.2` | `0.426` | `281.80` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (BF16_MXU)` | `T2I-1536x1024-8step` | `8` | `16.33` | `45.89` | `23.24` | `168.5` | `213.4` | `0.620` | `193.54` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-8step` | `8` | `2,917.30` | `19.76` | `6,173.59` | `114.0` | `142.2` | `9.391` | `12.78` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1536x1024-8step` | `8` | `2.72` | `30.83` | `6,148.01` | `168.5` | `213.4` | `6.611` | `18.16` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-2048x2048-8step` | `8` | `2.66` | `93.21` | `7,040.96` | `449.5` | `571.7` | `8.361` | `14.36` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1024x1024-8step` | `8` | `3,015.33` | `24.70` | `13.73` | `114.0` | `142.2` | `3.392` | `35.38` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1536x1024-8step` | `8` | `5.19` | `53.52` | `19.79` | `168.5` | `213.4` | `0.690` | `174.02` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-8step` | `8` | `20.63` | `34.56` | `16.20` | `114.0` | `142.2` | `0.455` | `263.46` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `1.08` | `21.37` | `127.53` | `168.5` | `213.4` | `0.963` | `249.08` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-28step` | `28` | `14.28` | `25.96` | `1.22` | `114.0` | `142.2` | `0.885` | `271.32` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (BF16_MXU)` | `T2I-1536x1024-28step` | `28` | `15.19` | `26.50` | `1.08` | `168.5` | `213.4` | `0.972` | `247.00` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `1.05` | `35.77` | `105.93` | `168.5` | `213.4` | `1.345` | `178.44` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-28step` | `28` | `2,393.84` | `40.10` | `6,205.53` | `114.0` | `142.2` | `9.864` | `12.16` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `T2I-1536x1024-28step` | `28` | `3.13` | `71.17` | `5,945.71` | `168.5` | `213.4` | `8.155` | `14.72` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `T2I-2048x2048-28step` | `28` | `3.11` | `204.12` | `6,980.52` | `449.5` | `571.7` | `13.271` | `9.04` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1024x1024-28step` | `28` | `2,361.86` | `55.81` | `13.92` | `114.0` | `142.2` | `4.104` | `29.24` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `6.91` | `101.13` | `20.13` | `168.5` | `213.4` | `3.095` | `38.78` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-28step` | `28` | `16.33` | `54.03` | `15.92` | `114.0` | `142.2` | `1.687` | `71.12` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (BF16_MXU)` | `T2I-1536x1024-28step` | `28` | `12.99` | `83.72` | `22.88` | `168.5` | `213.4` | `2.593` | `46.28` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-28step` | `28` | `2,830.76` | `37.86` | `6,031.46` | `114.0` | `142.2` | `10.064` | `11.92` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1536x1024-28step` | `28` | `2.46` | `66.77` | `6,002.91` | `168.5` | `213.4` | `8.088` | `14.84` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-2048x2048-28step` | `28` | `2.84` | `195.65` | `6,739.10` | `449.5` | `571.7` | `12.792` | `9.38` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1024x1024-28step` | `28` | `2,926.72` | `50.18` | `14.31` | `114.0` | `142.2` | `4.511` | `26.60` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `5.14` | `91.62` | `19.64` | `168.5` | `213.4` | `2.827` | `42.46` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-28step` | `28` | `21.05` | `47.99` | `15.87` | `114.0` | `142.2` | `1.523` | `78.80` |

#### Track D — Disaggregated CPU Image Encoding (`PNG` & `WebP`) Ablation (`POST /encode_image`)

| Resolution & Format | On-Host CPU (ms) | Disagg CPU RTT (ms) | Remote AVX-512 Compute (ms) | Remote Rate (img/s) | Local / Disagg RTT Ratio |
|---|---:|---:|---:|---:|---:|
| `512x512_png` | `28.32` | `36.39` | `34.94` | `28.6` | `0.78x` |
| `512x512_webp` | `27.87` | `35.24` | `34.05` | `29.4` | `0.79x` |
| `1024x1024_png` | `114.00` | `142.17` | `140.49` | `7.1` | `0.80x` |
| `1024x1024_webp` | `112.41` | `136.70` | `135.05` | `7.4` | `0.82x` |
| `1328x1328_png` | `190.44` | `240.24` | `237.17` | `4.2` | `0.79x` |
| `1328x1328_webp` | `196.31` | `230.67` | `227.75` | `4.4` | `0.85x` |
| `1664x928_png` | `166.42` | `209.75` | `207.22` | `4.8` | `0.79x` |
| `1664x928_webp` | `166.28` | `201.25` | `198.90` | `5.0` | `0.83x` |
| `1536x1024_png` | `168.51` | `213.37` | `210.77` | `4.7` | `0.79x` |
| `1536x1024_webp` | `178.89` | `206.30` | `203.50` | `4.9` | `0.87x` |
| `2048x2048_png` | `449.52` | `571.70` | `566.62` | `1.8` | `0.79x` |
| `2048x2048_webp` | `452.68` | `544.94` | `539.99` | `1.9` | `0.83x` |

### Slice `v7x-8-keepalive` (`v7x`, `1` Verified Benchmark Points)

- **Raw Log:** [`results/v7x-8-keepalive/summary.json`](./results/v7x-8-keepalive/summary.json)

#### Track A — OpenAI-Compatible `vllm-omni` HTTP Serving (`Qwen/Qwen-Image` 20B MMDiT)

| Engine | Precision | `Ulysses × Cluster DP` | Resolution | Steps | Conc | Warmup (s) | Median E2E (s) | P95 E2E (s) | Replica Img/min | Cluster Img/min |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=4` | `512x512` | `20` | `1` | `71.98` | `37.683` | `71.708` | `1.59` | `6.37` |

#### Track D — Disaggregated CPU Image Encoding (`PNG` & `WebP`) Ablation (`POST /encode_image`)

| Resolution & Format | On-Host CPU (ms) | Disagg CPU RTT (ms) | Remote AVX-512 Compute (ms) | Remote Rate (img/s) | Local / Disagg RTT Ratio |
|---|---:|---:|---:|---:|---:|
| `512x512_png` | `28.01` | `36.05` | `34.86` | `28.7` | `0.78x` |
| `512x512_webp` | `27.87` | `34.93` | `33.92` | `29.5` | `0.80x` |
| `1024x1024_png` | `123.63` | `142.05` | `140.39` | `7.1` | `0.87x` |
| `1024x1024_webp` | `114.16` | `136.30` | `134.95` | `7.4` | `0.84x` |
| `1328x1328_png` | `198.73` | `238.87` | `236.41` | `4.2` | `0.83x` |
| `1328x1328_webp` | `189.44` | `231.07` | `228.15` | `4.4` | `0.82x` |
| `1664x928_png` | `165.84` | `209.23` | `207.01` | `4.8` | `0.79x` |
| `1664x928_webp` | `169.34` | `201.26` | `198.97` | `5.0` | `0.84x` |
| `1536x1024_png` | `178.49` | `212.62` | `210.67` | `4.7` | `0.84x` |
| `1536x1024_webp` | `168.74` | `205.13` | `202.72` | `4.9` | `0.82x` |
| `2048x2048_png` | `452.78` | `567.80` | `562.76` | `1.8` | `0.80x` |
| `2048x2048_webp` | `450.75` | `545.52` | `540.74` | `1.8` | `0.83x` |

### Slice `v7x-4` (`v7x`, `89` Verified Benchmark Points)

- **Raw Log:** [`results/v7x-4/summary.json`](./results/v7x-4/summary.json)

#### Track A — OpenAI-Compatible `vllm-omni` HTTP Serving (`Qwen/Qwen-Image` 20B MMDiT)

| Engine | Precision | `Ulysses × Cluster DP` | Resolution | Steps | Conc | Warmup (s) | Median E2E (s) | P95 E2E (s) | Replica Img/min | Cluster Img/min |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=2` | `512x512` | `20` | `1` | `72.33` | `38.640` | `73.119` | `1.55` | `3.10` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=2` | `1024x1024` | `20` | `1` | `102.92` | `46.674` | `89.032` | `1.28` | `2.57` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=2` | `1024x1024` | `20` | `2` | `102.92` | `7.289` | `9.816` | `12.17` | `24.34` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=2` | `1024x1024` | `20` | `4` | `102.92` | `13.021` | `107.007` | `2.24` | `4.48` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=2` | `1024x1024` | `50` | `1` | `10.93` | `10.818` | `10.873` | `5.53` | `11.05` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=2` | `1328x1328` | `20` | `1` | `98.20` | `37.466` | `69.521` | `1.60` | `3.20` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=2` | `1328x1328` | `20` | `2` | `98.20` | `8.423` | `11.361` | `10.50` | `20.99` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=2` | `1328x1328` | `50` | `1` | `12.98` | `12.844` | `12.899` | `4.65` | `9.30` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=2` | `1664x928` | `20` | `1` | `118.23` | `54.283` | `103.533` | `1.10` | `2.21` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=2` | `1664x928` | `20` | `2` | `118.23` | `8.506` | `11.516` | `10.37` | `20.73` |
| `vllm-omni-flash_attn` | `BF16` | `U=4 × DP=2` | `1664x928` | `50` | `1` | `13.08` | `11.967` | `12.010` | `4.99` | `9.98` |
| `vllm-omni-sdpa` | `BF16` | `U=4 × DP=2` | `1024x1024` | `20` | `1` | `9.86` | `6.582` | `9.178` | `9.06` | `18.13` |
| `vllm-omni-sdpa` | `BF16` | `U=4 × DP=2` | `1024x1024` | `20` | `2` | `9.86` | `5.109` | `6.807` | `17.54` | `35.07` |
| `vllm-omni-sdpa` | `BF16` | `U=4 × DP=2` | `1328x1328` | `20` | `1` | `7.84` | `5.752` | `7.691` | `10.32` | `20.64` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=1` | `512x512` | `20` | `1` | `10.11` | `6.743` | `9.899` | `8.88` | `8.88` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=1` | `1024x1024` | `20` | `1` | `46.92` | `7.127` | `10.513` | `8.37` | `8.37` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=1` | `1024x1024` | `20` | `2` | `46.92` | `5.779` | `7.925` | `15.06` | `15.06` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=1` | `1024x1024` | `20` | `4` | `46.92` | `9.100` | `24.736` | `9.69` | `9.69` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=1` | `1024x1024` | `50` | `1` | `8.85` | `9.437` | `10.124` | `6.33` | `6.33` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=1` | `1328x1328` | `20` | `1` | `59.20` | `8.314` | `12.714` | `7.16` | `7.16` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=1` | `1328x1328` | `20` | `2` | `59.20` | `6.356` | `8.889` | `13.40` | `13.40` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=1` | `1328x1328` | `50` | `1` | `9.24` | `9.216` | `9.283` | `6.46` | `6.46` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=1` | `1664x928` | `20` | `1` | `54.71` | `9.175` | `14.391` | `6.50` | `6.50` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=1` | `1664x928` | `20` | `2` | `54.71` | `6.363` | `8.669` | `13.75` | `13.75` |
| `vllm-omni-flash_attn` | `BF16` | `U=8 × DP=1` | `1664x928` | `50` | `1` | `9.22` | `9.177` | `9.179` | `6.50` | `6.50` |

#### Track B — `Qwen-Image-2.1` & `Qwen-Image-2.1-Turbo` (`Qwen3-VL` Text-Encoded Native Pipeline)

| Model Checkpoint | Engine | `TP × Cluster DP` | Resolution | Steps | Weight Load (s) | Median E2E (s) | Pixel Std | Replica Img/min | Cluster Img/min |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=8` | `1024x1024` | `8` | `10.26` | `3.624` | `82.42` | `16.55` | `132.40` |
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=8` | `1536x1024` | `8` | `10.26` | `5.384` | `89.29` | `11.15` | `89.20` |
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=8` | `2048x2048` | `8` | `10.26` | `15.752` | `90.00` | `3.81` | `30.48` |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=8` | `1024x1024` | `28` | `11.77` | `11.881` | `80.59` | `5.05` | `40.40` |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=8` | `1536x1024` | `28` | `11.77` | `17.858` | `84.61` | `3.36` | `26.88` |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `TP=1 × DP=8` | `2048x2048` | `28` | `11.77` | `52.929` | `78.96` | `1.13` | `9.04` |

#### Track C — `TorchTPU` Multi-Stage Breakdown (`Qwen-Image` & `Qwen-Image-2.1`, `BF16` & `FP8` MXU)

| Model | Engine & Kernel | Task & Resolution | Steps | VL Encode (ms) | MMDiT Step (ms) | VAE Decode (ms) | Local PNG (ms) | Disagg PNG (ms) | E2E Disagg (s) | Cluster Img/min |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-50step` | `50` | `0.98` | `50.08` | `5.95` | `112.7` | `142.8` | `2.654` | `45.22` |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1328x1328-50step` | `50` | `0.79` | `50.37` | `35.59` | `199.1` | `251.6` | `2.807` | `42.76` |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1664x928-20step` | `20` | `1.02` | `49.93` | `8,363.60` | `167.6` | `209.5` | `9.573` | `12.54` |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1664x928-50step` | `50` | `0.77` | `51.69` | `25.03` | `167.6` | `209.5` | `2.820` | `42.56` |
| `Qwen/Qwen-Image` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-20step` | `20` | `18.52` | `79.49` | `1.16` | `112.7` | `142.8` | `1.752` | `68.48` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-512x512-20step` | `20` | `1,705.98` | `21.28` | `5,507.11` | `29.2` | `36.6` | `7.675` | `7.82` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-512x512-50step` | `50` | `2.23` | `19.12` | `6.48` | `29.2` | `36.6` | `1.001` | `59.92` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-20step` | `20` | `2.26` | `52.16` | `6,022.37` | `112.7` | `142.8` | `7.211` | `8.32` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-50step` | `50` | `2.25` | `50.21` | `13.42` | `112.7` | `142.8` | `2.669` | `22.48` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1328x1328-20step` | `20` | `3.03` | `107.53` | `6,239.27` | `199.1` | `251.6` | `8.644` | `6.94` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1328x1328-50step` | `50` | `2.69` | `105.31` | `23.41` | `199.1` | `251.6` | `5.543` | `10.82` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1664x928-20step` | `20` | `2.91` | `87.85` | `8,459.08` | `167.6` | `209.5` | `10.429` | `5.75` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (BF16_MXU)` | `T2I-1664x928-50step` | `50` | `2.55` | `87.25` | `20.23` | `167.6` | `209.5` | `4.595` | `13.06` |
| `Qwen/Qwen-Image` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-20step` | `20` | `13.08` | `77.72` | `15.77` | `112.7` | `142.8` | `1.726` | `34.76` |
| `Qwen/Qwen-Image` | `torchtpu-eager (BF16_MXU)` | `T2I-1328x1328-20step` | `20` | `13.81` | `141.94` | `26.91` | `199.1` | `251.6` | `3.131` | `19.16` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-512x512-20step` | `20` | `2,231.24` | `19.28` | `5,535.73` | `29.2` | `36.6` | `8.189` | `7.33` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-512x512-50step` | `50` | `1.69` | `18.80` | `6.76` | `29.2` | `36.6` | `0.985` | `60.93` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-20step` | `20` | `2.09` | `54.17` | `6,206.15` | `112.7` | `142.8` | `7.434` | `8.07` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-50step` | `50` | `1.85` | `54.90` | `14.19` | `112.7` | `142.8` | `2.904` | `20.66` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1328x1328-20step` | `20` | `2.30` | `108.35` | `6,539.69` | `199.1` | `251.6` | `8.961` | `6.70` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1328x1328-50step` | `50` | `2.02` | `108.99` | `23.37` | `199.1` | `251.6` | `5.727` | `10.48` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1664x928-20step` | `20` | `2.38` | `90.42` | `8,698.29` | `167.6` | `209.5` | `10.719` | `5.60` |
| `Qwen/Qwen-Image` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1664x928-50step` | `50` | `2.23` | `88.80` | `20.26` | `167.6` | `209.5` | `4.672` | `12.84` |
| `Qwen/Qwen-Image` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-20step` | `20` | `15.63` | `71.01` | `15.90` | `112.7` | `142.8` | `1.595` | `37.63` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-8step` | `8` | `13.92` | `27.03` | `0.84` | `112.7` | `142.8` | `0.374` | `321.00` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (BF16_MXU)` | `T2I-1536x1024-8step` | `8` | `14.68` | `25.67` | `1.31` | `169.3` | `214.4` | `0.436` | `275.42` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-8step` | `8` | `20.65` | `39.32` | `1.09` | `112.7` | `142.8` | `0.479` | `250.44` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-8step` | `8` | `2,339.94` | `22.26` | `6,260.96` | `112.7` | `142.8` | `8.922` | `6.73` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `T2I-1536x1024-8step` | `8` | `2.98` | `35.02` | `5,905.13` | `169.3` | `214.4` | `6.403` | `9.37` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `T2I-2048x2048-8step` | `8` | `2.92` | `102.14` | `7,127.43` | `450.6` | `571.4` | `8.519` | `7.04` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1024x1024-8step` | `8` | `2,279.79` | `28.80` | `13.67` | `112.7` | `142.8` | `2.691` | `22.30` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1536x1024-8step` | `8` | `7.20` | `55.87` | `19.23` | `169.3` | `214.4` | `0.712` | `84.31` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-8step` | `8` | `16.26` | `31.02` | `15.96` | `112.7` | `142.8` | `0.423` | `141.76` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (BF16_MXU)` | `T2I-1536x1024-8step` | `8` | `15.73` | `46.82` | `22.99` | `169.3` | `214.4` | `0.628` | `95.59` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-8step` | `8` | `2,845.90` | `23.59` | `5,690.17` | `112.7` | `142.8` | `8.868` | `6.77` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1536x1024-8step` | `8` | `2.39` | `30.19` | `5,741.56` | `169.3` | `214.4` | `6.200` | `9.68` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-2048x2048-8step` | `8` | `2.67` | `89.97` | `7,021.82` | `450.6` | `571.4` | `8.316` | `7.22` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1024x1024-8step` | `8` | `2,958.95` | `25.06` | `13.49` | `112.7` | `142.8` | `3.340` | `17.97` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1536x1024-8step` | `8` | `5.20` | `53.65` | `19.69` | `169.3` | `214.4` | `0.692` | `86.66` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-8step` | `8` | `20.19` | `35.46` | `15.71` | `112.7` | `142.8` | `0.462` | `129.75` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `0.99` | `21.36` | `192.90` | `169.3` | `214.4` | `1.030` | `116.46` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-28step` | `28` | `14.34` | `28.38` | `0.93` | `112.7` | `142.8` | `0.953` | `125.94` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (BF16_MXU)` | `T2I-1536x1024-28step` | `28` | `14.89` | `33.47` | `1.07` | `169.3` | `214.4` | `1.167` | `102.78` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `0.95` | `26.49` | `113.28` | `169.3` | `214.4` | `1.094` | `109.68` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-28step` | `28` | `24.22` | `46.64` | `1.04` | `112.7` | `142.8` | `1.474` | `81.40` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `T2I-1024x1024-28step` | `28` | `2,375.70` | `39.72` | `6,153.88` | `112.7` | `142.8` | `9.785` | `6.13` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `T2I-1536x1024-28step` | `28` | `3.41` | `71.26` | `6,125.04` | `169.3` | `214.4` | `8.338` | `7.20` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `T2I-2048x2048-28step` | `28` | `3.31` | `203.06` | `7,036.05` | `450.6` | `571.4` | `13.296` | `4.51` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1024x1024-28step` | `28` | `2,317.58` | `55.00` | `14.37` | `112.7` | `142.8` | `4.039` | `14.86` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (BF16_MXU)` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `6.82` | `99.59` | `19.16` | `169.3` | `214.4` | `3.053` | `19.65` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (BF16_MXU)` | `T2I-1024x1024-28step` | `28` | `16.90` | `54.72` | `15.78` | `112.7` | `142.8` | `1.708` | `35.13` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (BF16_MXU)` | `T2I-1536x1024-28step` | `28` | `15.10` | `85.62` | `23.27` | `169.3` | `214.4` | `2.650` | `22.64` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-28step` | `28` | `2,805.01` | `37.97` | `6,203.23` | `112.7` | `142.8` | `10.214` | `5.87` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1536x1024-28step` | `28` | `2.88` | `63.81` | `6,122.41` | `169.3` | `214.4` | `8.126` | `7.38` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-2048x2048-28step` | `28` | `2.68` | `194.89` | `7,022.36` | `450.6` | `571.4` | `13.053` | `4.60` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1024x1024-28step` | `28` | `2,899.46` | `50.55` | `13.84` | `112.7` | `142.8` | `4.495` | `13.35` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `5.26` | `91.71` | `19.75` | `169.3` | `214.4` | `2.831` | `21.19` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager (torch._scaled_mm(float8_e4m3fn, out_dtype=bfloat16))` | `T2I-1024x1024-28step` | `28` | `18.89` | `48.79` | `16.00` | `112.7` | `142.8` | `1.544` | `38.86` |

#### Track D — Disaggregated CPU Image Encoding (`PNG` & `WebP`) Ablation (`POST /encode_image`)

| Resolution & Format | On-Host CPU (ms) | Disagg CPU RTT (ms) | Remote AVX-512 Compute (ms) | Remote Rate (img/s) | Local / Disagg RTT Ratio |
|---|---:|---:|---:|---:|---:|
| `512x512_png` | `29.17` | `36.58` | `35.28` | `28.3` | `0.80x` |
| `512x512_webp` | `27.97` | `35.48` | `34.25` | `29.2` | `0.79x` |
| `1024x1024_png` | `112.66` | `142.84` | `140.78` | `7.1` | `0.79x` |
| `1024x1024_webp` | `112.80` | `137.68` | `135.81` | `7.4` | `0.82x` |
| `1328x1328_png` | `199.10` | `251.60` | `248.43` | `4.0` | `0.79x` |
| `1328x1328_webp` | `193.99` | `230.70` | `227.65` | `4.4` | `0.84x` |
| `1664x928_png` | `167.56` | `209.46` | `206.85` | `4.8` | `0.80x` |
| `1664x928_webp` | `165.54` | `201.80` | `198.96` | `5.0` | `0.82x` |
| `1536x1024_png` | `169.27` | `214.39` | `211.08` | `4.7` | `0.79x` |
| `1536x1024_webp` | `170.67` | `205.83` | `203.15` | `4.9` | `0.83x` |
| `2048x2048_png` | `450.61` | `571.35` | `565.46` | `1.8` | `0.79x` |
| `2048x2048_webp` | `445.77` | `546.79` | `540.95` | `1.8` | `0.82x` |

---
