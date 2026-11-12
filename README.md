# Qwen-Image (20B MMDiT) and Qwen-Image-2.1 on Google Cloud TPU v5p and v7x

End-to-end HTTP serving (`vllm-omni`), native single-device `diffusers` pipelines, TorchTPU compilation breakdowns, and disaggregated CPU image-encoding benchmarks for [`Qwen/Qwen-Image`](https://huggingface.co/Qwen/Qwen-Image), [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1), and [`Qwen/Qwen-Image-2.1-Turbo`](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) on Google Cloud TPU v5p and TPU v7x.

**Summary:** On full end-to-end `Qwen/Qwen-Image` (20B MMDiT, `1024×1024`, 20 steps), switching `vllm-omni` (`Ulysses=4`) from `flash_attn` to `TORCH_SDPA` (`sdpa`) on TPU v5p reduces warm `c=2` median latency from 9.26 s to **5.83 s (1.59× faster)**, while TPU v7x (`Ulysses=4` on 2 physical chips) reaches **5.11 s (`17.54 img/min/replica`, 1.83× higher per-replica throughput than the `v5p` `flash_attn` baseline at half the physical chip count)**. On single-device `Qwen/Qwen-Image-2.1-Turbo` (8 steps, `diffusers`), `TP=1` reaches **3.30 s on 1 TPU v5p chip (`18.19 img/min/chip`)** and **3.60 s on 1 TPU v7x TensorCore (`16.69 img/min/TensorCore`, or `33.38 img/min` across both TensorCores of one physical `v7x` chip — 1.84× higher per-chip throughput than `v5p`)**.

---

## 1. Model, Framework, and Hardware

- **Models:**
  - **[`Qwen/Qwen-Image`](https://huggingface.co/Qwen/Qwen-Image)** (`53.74 GiB` `bfloat16`): **20B double-stream MMDiT** combining `Qwen2.5-VL` (`15.45 GiB`, `28` layers, `width 3,584`), `QwenImageTransformer2DModel` (`38.05 GiB`, `60` layers, `24` heads × `128` dim = `3,072` width, `patch_size 2`), and a `0.24 GiB` VAE (`20` and `50` steps, `true_cfg_scale=4.0`).
  - **[`Qwen/Qwen-Image-2.1-Turbo`](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)** (`30.21 GiB` `bfloat16`, `8` steps, `cfg=1.0`) and **[`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1)** (`30.84 GiB` `bfloat16`, `28` steps, `cfg=4.0`): `Qwen3-VL` text encoder (`16.33 GiB`, `36` layers, `width 4,096`) + `QwenImage21Transformer2DModel` (`13.25 GiB`, `32` layers, `32` heads × `128` dim = `4,096` width, `patch_size 1`) + `AutoencoderKLQwenImage21` (`0.63–1.26 GiB`). Fits inside a single `v5p` chip or `v7x` TensorCore (`TP=1`).
- **Framework:** Python 3.12, `vllm-omni` OpenAI HTTP server (`TPU_BACKEND_TYPE=jax`, Ulysses sequence parallelism), HuggingFace `diffusers` (`QwenImage21Pipeline`), PyTorch + `torch_tpu` (`vllm-torchtpu-omni:v0.32.0`), and a `c3-standard-44` (`44` Sapphire Rapids vCPUs, AVX-512) HTTP/1.1 Keep-Alive image-encoding service ([`scripts/multimodal_cpu_offload_server.py`](./scripts/multimodal_cpu_offload_server.py)).
- **Hardware:** Google Cloud TPU v5p and TPU v7x (Ironwood) on GKE:
  - **TPU v5p** (`95 GiB` HBM2e per chip, 1 device per chip, 4 chips per host): `v5p-16` (`2x2x4`, 4 hosts = 16 chips).
  - **TPU v7x** (`192 GiB` HBM3e per chip / `96 GiB` per TensorCore, 2 TensorCores per chip, 4 chips = 8 TensorCores per host): `v7x-4` (`2x2x1`, 1 host), `v7x-8` (`2x2x2`, 2 hosts), and `v7x-16` (`2x2x4`, 4 hosts).

---

## 2. Main Results

### A. End-to-End `vllm-omni` HTTP Serving — `Qwen/Qwen-Image` (20B MMDiT, `bfloat16`, Warm XLA Cache)

| Hardware & Topology | Attention | Resolution & Steps | Concurrency | Median E2E | Replica `img/min` | Comparison |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| **`v5p-16`** (`U=4`, 4 chips/replica) | `flash_attn` | `1024×1024`, 20 steps | `c=2` | 9.258 s | 9.57 | `v5p` `flash_attn` baseline (`1.0×`) |
| **`v5p-16`** (`U=4`, 4 chips/replica) | `sdpa` | `1024×1024`, 20 steps | `c=1` / `c=2` | 6.621 s / **5.830 s** | 9.00 / **15.12** | **1.58× higher** replica rate at `c=2` than `v5p` `flash_attn` |
| **`v5p-16`** (`U=4`, 4 chips/replica) | `flash_attn` | `1024×1024`, 50 steps | `c=1` | 14.571 s | 4.11 | `v5p` 50-step baseline (`1.0×`) |
| **`v5p-16`** (`U=4`, 4 chips/replica) | `flash_attn` | `1664×928`, 20 steps | `c=2` | 10.331 s | 8.71 | `v5p` widescreen baseline (`1.0×`) |
| **`v7x-4`** (`U=4`, 2 chips = 4 TCs/replica) | `flash_attn` | `1024×1024`, 20 steps | `c=2` | 7.289 s | 12.17 | **1.27× faster** latency (**2.54×** per physical chip) vs. `v5p` `flash_attn` |
| **`v7x-4`** (`U=4`, 2 chips = 4 TCs/replica) | `sdpa` | `1024×1024`, 20 steps | `c=1` / `c=2` | 6.582 s / **5.109 s** | 9.06 / **17.54** | **1.83× higher** replica rate (**3.66×** per chip) vs. `v5p` `flash_attn` |
| **`v7x-8` / `v7x-16`** (`U=8`, 4 chips = 8 TCs) | `flash_attn` | `1024×1024`, 20 / 50 steps | `c=2` / `c=1` | **5.779 s** / **8.716 s** | 15.06 / 6.85 | **1.60× / 1.67× faster** than 4-chip `v5p` `flash_attn` (`9.26 s` / `14.57 s`) |
| **`v7x-16`** (`U=8`, 4 chips = 8 TCs) | `flash_attn` | `1664×928`, 20 / 50 steps | `c=2` / `c=1` | **5.610 s** / **9.177 s** | **16.06** / 6.50 | **1.84× / 1.85× faster** than 4-chip `v5p` `flash_attn` (`10.33 s` / `16.99 s`) |

*Notes:* Full checkpoint inference (`pixel_std 75–96`). In archived `flash_attn` runs, `c=1` 20-step included a one-time cold recompile on the second prompt token length (`37–66 s`; see [`APPENDIX.md`](./APPENDIX.md)); `c=2`, 50-step, and `sdpa` rows above reflect warm XLA execution.

### B. End-to-End Single-Device `diffusers` (`QwenImage21Pipeline`, `TP=1`, `bfloat16`)

| Model & Steps | Resolution | `1× TPU v5p` Chip (`tpu:0`) | `1× TPU v7x` TensorCore (`tpu:0` = 0.5 Chip) | Comparison |
| :--- | :--- | ---: | ---: | :--- |
| **`Qwen-Image-2.1-Turbo` (8 steps)** | `1024×1024` | **3.299 s** (`18.19 img/min/chip`) | **3.595 s** (`16.69 img/min/TC` = **`33.38/chip`**) | `v5p` is `9%` faster per single stream; `v7x` delivers **1.84× higher** throughput per 2-TC physical chip |
| **`Qwen-Image-2.1-Turbo` (8 steps)** | `1536×1024` | **4.951 s** (`12.12 img/min/chip`) | **5.370 s** (`11.17 img/min/TC` = **`22.34/chip`**) | **1.84× higher** throughput per physical `v7x` chip vs. `v5p` |
| **`Qwen-Image-2.1-Turbo` (8 steps)** | `2048×2048` | **14.414 s** (`4.16 img/min/chip`) | **15.752 s** (`3.81 img/min/TC` = **`7.62/chip`**) | **1.83× higher** throughput per physical `v7x` chip vs. `v5p` |
| **`Qwen-Image-2.1` (28 steps)** | `1024×1024` | **10.711 s** (`5.60 img/min/chip`) | **11.881 s** (`5.05 img/min/TC` = **`10.10/chip`**) | `3.25×` slower than 8-step `Turbo`; **1.80× higher** per-chip rate on `v7x` vs. `v5p` |
| **`Qwen-Image-2.1` (28 steps)** | `2048×2048` | **47.894 s** (`1.25 img/min/chip`) | **52.929 s** (`1.13 img/min/TC` = **`2.26/chip`**) | Native `2K` generation in `30.8 GiB` `bfloat16` on one core |

### C. Local-Shard `TorchTPU` MMDiT Step Compilation & Disaggregated CPU Encoding

| Benchmark Component | TPU v5p Baseline | TPU v5p Optimized | TPU v7x (`TP=8`) | Comparison |
| :--- | ---: | ---: | ---: | :--- |
| **`Qwen-Image` `1024×1024` MMDiT step** *(local-shard, no collectives)* | 164.79 ms (`eager BF16`) | 121.70 ms (`compiled BF16`) | **50.16 ms** (`compiled BF16`) | Compiled `v5p` is **1.35× faster** than eager; `v7x` is **2.43× faster** than compiled `v5p` |
| **`Qwen-Image-2.1` `2048×2048` step** *(local-shard, no collectives)* | — | 499.12 ms (`compiled BF16`) | **195.65 ms** (`compiled FP8`) | `v7x` `FP8` is **2.55× faster** than compiled `v5p` `BF16` (`FP8` is `2.6×` slower on `v5p` due to no FP8 MXU) |
| **`2048×2048` WebP CPU encode** (`batch=1`, synchronous) | 613.55 ms (on-host `v5p`) | **545.01 ms** (`c3-standard-44` Keep-Alive) | **445.36 ms** (on-host `v7x`, vs. `544.9 ms` HTTP) | **1.13× faster** via HTTP offload on `v5p`; **18% slower** via HTTP than `v7x`'s 224-vCPU host |

---

## 3. Reproducing the Main Results

1. **Deploy the disaggregated CPU offload service** ([`k8s/cpu-offload-service.yaml`](./k8s/cpu-offload-service.yaml)):
```bash
kubectl apply -f k8s/cpu-offload-service.yaml
```

2. **Run the multi-engine benchmark suite** ([`scripts/run_qwen_image_tpu_bench.py`](./scripts/run_qwen_image_tpu_bench.py), or apply [`k8s/tpu-v5p-16-job.yaml`](./k8s/tpu-v5p-16-job.yaml), [`k8s/tpu-v7x-8-job.yaml`](./k8s/tpu-v7x-8-job.yaml), or [`k8s/tpu-v7x-16-job.yaml`](./k8s/tpu-v7x-16-job.yaml)):
```bash
# TPU v5p-16 (4 hosts × 4 chips)
python3 scripts/run_qwen_image_tpu_bench.py \
  --chip-type=v5p --slice-name=v5p-16 --num-hosts=4 --cores-per-host=4 \
  --bucket=YOUR_GCS_BUCKET --use-shm --output-dir=/tmp/results/qwen-image

# TPU v7x-16 (4 hosts × 8 TensorCores)
python3 scripts/run_qwen_image_tpu_bench.py \
  --chip-type=v7x --slice-name=v7x-16 --num-hosts=4 --cores-per-host=8 \
  --bucket=YOUR_GCS_BUCKET --use-shm --output-dir=/tmp/results/qwen-image
```

---

## 4. Detailed Ablations, Engineering Notes, and Raw Logs

- **[`APPENDIX.md`](./APPENDIX.md):** Complete 396-row per-slice tables across all 7 TPU slice runs (`v5p-16`, `v5p-16-keepalive`, `v5p-16-adaptive-64k`, `v7x-4`, `v7x-8`, `v7x-8-keepalive`, `v7x-16`), `TorchTPU` VL/MMDiT/VAE breakdowns, `FP8` vs. `BF16` comparisons, and diagnostic failures (`vllm-omni` `FP8` startup, multi-threaded `TP=4` Dynamo export race).
- **[`results/`](./results/):** Raw `summary.json` outputs for all 7 slice evaluations.
