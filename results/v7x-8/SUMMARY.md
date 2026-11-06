# Qwen-Image (`20B` MMDiT) & Qwen-Image-2.1 / 2.1-Turbo (`Qwen3-VL`) — `v7x-8` (`v7x`)

- **Updated (UTC):** `2026-11-06T13:47:49.268519+00:00`
- **Hardware Topology:** `2` hosts x `8` TPU cores/host = `16` TPU cores
- **Models Covered:** `Qwen/Qwen-Image` (`Qwen2.5-VL` + 60-block MMDiT), `Qwen/Qwen-Image-2.1-Turbo` (`8`-step `Qwen3-VL`), `Qwen/Qwen-Image-2.1` (`28`-step `Qwen3-VL`)

## Disaggregated CPU PNG / WebP Image Encoding Ablation (`multimodal-cpu-offload-svc:8090`)

| Resolution | Offload Active | On-Host CPU PNG (ms) | Disagg AVX-512 PNG RTT (ms) | Disagg WebP RTT (ms) | PNG Offload Speedup |
|---|:---:|---:|---:|---:|---:|
| `512x512` | `True` | 28.32 | **36.39** | 35.24 | `0.78x` |
| `1024x1024` | `True` | 114.0 | **142.17** | 136.7 | `0.8x` |
| `1328x1328` | `True` | 190.44 | **240.24** | 230.67 | `0.79x` |
| `1664x928` | `True` | 166.42 | **209.75** | 201.25 | `0.79x` |
| `1536x1024` | `True` | 168.51 | **213.37** | 206.3 | `0.79x` |
| `2048x2048` | `True` | 449.52 | **571.7** | 544.94 | `0.79x` |

## End-to-End Diffusion Serving & TorchTPU Compilation Results

| Model | Engine | Prec | TP/Ulysses | Cluster DP | Task / Resolution | Steps | Conc | Stage-1 VL (ms) | MMDiT Step (ms) | VAE (ms) | E2E Local CPU (s) | E2E P50 (s) | Cluster img/min | Status |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `4` | `T2I-OpenAI-HTTP-512x512` | `20` | `1` | - | - | - | - | **39.064** | **6.14** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `4` | `T2I-OpenAI-HTTP-1024x1024` | `20` | `1` | - | - | - | - | **45.506** | **5.27** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `4` | `T2I-OpenAI-HTTP-1024x1024` | `20` | `2` | - | - | - | - | **7.983** | **44.3** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `4` | `T2I-OpenAI-HTTP-1024x1024` | `20` | `4` | - | - | - | - | **12.962** | **8.9** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `4` | `T2I-OpenAI-HTTP-1024x1024` | `50` | `1` | - | - | - | - | **11.318** | **21.13** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `4` | `T2I-OpenAI-HTTP-1328x1328` | `20` | `1` | - | - | - | - | **37.338** | **6.42** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `4` | `T2I-OpenAI-HTTP-1328x1328` | `20` | `2` | - | - | - | - | **8.861** | **39.95** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `4` | `T2I-OpenAI-HTTP-1328x1328` | `50` | `1` | - | - | - | - | **13.72** | **17.41** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `4` | `T2I-OpenAI-HTTP-1664x928` | `20` | `1` | - | - | - | - | **53.919** | **4.45** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `4` | `T2I-OpenAI-HTTP-1664x928` | `20` | `2` | - | - | - | - | **8.087** | **43.66** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `4` | `T2I-OpenAI-HTTP-1664x928` | `50` | `1` | - | - | - | - | **12.602** | **18.96** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-sdpa` | `BF16` | `4` | `4` | `T2I-OpenAI-HTTP-1024x1024` | `20` | `1` | - | - | - | - | **6.114** | **39.01** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-sdpa` | `BF16` | `4` | `4` | `T2I-OpenAI-HTTP-1024x1024` | `20` | `2` | - | - | - | - | **5.204** | **69.6** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-sdpa` | `BF16` | `4` | `4` | `T2I-OpenAI-HTTP-1328x1328` | `20` | `1` | - | - | - | - | **5.829** | **40.74** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `FP8` | `4` | `4` | `ALL` | `20` | `1` | - | - | - | - | **-** | **-** | `DIAGNOSTIC(Exited rc=1: startup_future, startup_timeout)
  File "/usr/local/lib/python3.12/site-packages/vllm_omni/engine/omni_engine_base.py", line 549, in _wait_for_orchestrator_init
    startup_future.result(
  File "/usr/local/)` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `8` | `2` | `T2I-OpenAI-HTTP-512x512` | `20` | `1` | - | - | - | - | **7.066** | **16.96** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `8` | `2` | `T2I-OpenAI-HTTP-1024x1024` | `20` | `1` | - | - | - | - | **7.398** | **16.13** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `8` | `2` | `T2I-OpenAI-HTTP-1024x1024` | `20` | `2` | - | - | - | - | **6.211** | **28.18** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `8` | `2` | `T2I-OpenAI-HTTP-1024x1024` | `20` | `4` | - | - | - | - | **9.781** | **18.61** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `8` | `2` | `T2I-OpenAI-HTTP-1024x1024` | `50` | `1` | - | - | - | - | **9.158** | **13.05** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `8` | `2` | `T2I-OpenAI-HTTP-1328x1328` | `20` | `1` | - | - | - | - | **8.48** | **14.04** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `8` | `2` | `T2I-OpenAI-HTTP-1328x1328` | `20` | `2` | - | - | - | - | **5.953** | **29.44** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `8` | `2` | `T2I-OpenAI-HTTP-1328x1328` | `50` | `1` | - | - | - | - | **9.595** | **12.42** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `8` | `2` | `T2I-OpenAI-HTTP-1664x928` | `20` | `1` | - | - | - | - | **9.505** | **12.55** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `8` | `2` | `T2I-OpenAI-HTTP-1664x928` | `20` | `2` | - | - | - | - | **6.751** | **26.04** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `8` | `2` | `T2I-OpenAI-HTTP-1664x928` | `50` | `1` | - | - | - | - | **10.24** | **11.66** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `FP8` | `8` | `2` | `ALL` | `20` | `1` | - | - | - | - | **-** | **-** | `DIAGNOSTIC(Exited rc=1: startup_future, startup_timeout)
  File "/usr/local/lib/python3.12/site-packages/vllm_omni/engine/omni_engine_base.py", line 549, in _wait_for_orchestrator_init
    startup_future.result(
  File "/usr/local/)` |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback` | `FP8` | `4` | `4` | `T2I-512x512-20step` | `20` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback` | `FP8` | `4` | `4` | `T2I-512x512-50step` | `50` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback` | `FP8` | `4` | `4` | `T2I-1024x1024-20step` | `20` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback` | `FP8` | `4` | `4` | `T2I-1024x1024-50step` | `50` | `1` | 0.98 | 63.51 | 0.9 | 3.292 | **3.32** | **72.28** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback` | `FP8` | `4` | `4` | `T2I-1328x1328-20step` | `20` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback` | `FP8` | `4` | `4` | `T2I-1328x1328-50step` | `50` | `1` | 1.13 | 64.07 | 29.2 | 3.424 | **3.474** | **69.08** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback` | `FP8` | `4` | `4` | `T2I-1664x928-20step` | `20` | `1` | 0.98 | 65.79 | 8278.49 | 9.762 | **9.805** | **24.48** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback` | `FP8` | `4` | `4` | `T2I-1664x928-50step` | `50` | `1` | 1.0 | 64.02 | 20.68 | 3.389 | **3.432** | **69.92** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-eager` | `FP8` | `4` | `4` | `T2I-1024x1024-20step` | `20` | `1` | 16.67 | 77.08 | 1.0 | 1.673 | **1.701** | **141.04** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `BF16` | `8` | `2` | `T2I-512x512-20step` | `20` | `1` | 1727.05 | 19.59 | 5629.67 | 7.777 | **7.785** | **15.42** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `BF16` | `8` | `2` | `T2I-512x512-50step` | `50` | `1` | 2.44 | 19.16 | 6.47 | 0.995 | **1.003** | **119.62** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `BF16` | `8` | `2` | `T2I-1024x1024-20step` | `20` | `1` | 2.88 | 51.76 | 5763.93 | 6.916 | **6.944** | **17.28** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `BF16` | `8` | `2` | `T2I-1024x1024-50step` | `50` | `1` | 2.43 | 51.08 | 13.54 | 2.684 | **2.712** | **44.24** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `BF16` | `8` | `2` | `T2I-1328x1328-20step` | `20` | `1` | 2.8 | 106.21 | 6427.27 | 8.745 | **8.794** | **13.64** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `BF16` | `8` | `2` | `T2I-1328x1328-50step` | `50` | `1` | 2.55 | 106.14 | 23.55 | 5.523 | **5.573** | **21.54** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `BF16` | `8` | `2` | `T2I-1664x928-20step` | `20` | `1` | 2.46 | 89.72 | 8392.99 | 10.356 | **10.4** | **11.54** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `BF16` | `8` | `2` | `T2I-1664x928-50step` | `50` | `1` | 2.41 | 88.89 | 19.63 | 4.633 | **4.677** | **25.66** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-eager` | `BF16` | `8` | `2` | `T2I-1024x1024-20step` | `20` | `1` | 13.22 | 77.39 | 15.75 | 1.691 | **1.719** | **69.82** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-eager` | `BF16` | `8` | `2` | `T2I-1328x1328-20step` | `20` | `1` | 11.62 | 137.95 | 27.15 | 2.988 | **3.038** | **39.5** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `FP8` | `8` | `2` | `T2I-512x512-20step` | `20` | `1` | 2250.65 | 19.78 | 5236.68 | 7.911 | **7.919** | **15.16** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `FP8` | `8` | `2` | `T2I-512x512-50step` | `50` | `1` | 1.85 | 18.37 | 6.79 | 0.955 | **0.963** | **124.56** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `FP8` | `8` | `2` | `T2I-1024x1024-20step` | `20` | `1` | 2.1 | 53.0 | 5903.98 | 7.08 | **7.108** | **16.88** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `FP8` | `8` | `2` | `T2I-1024x1024-50step` | `50` | `1` | 1.86 | 53.19 | 13.92 | 2.789 | **2.817** | **42.6** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `FP8` | `8` | `2` | `T2I-1328x1328-20step` | `20` | `1` | 2.54 | 107.59 | 6915.12 | 9.26 | **9.31** | **12.88** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `FP8` | `8` | `2` | `T2I-1328x1328-50step` | `50` | `1` | 2.08 | 108.11 | 22.83 | 5.621 | **5.671** | **21.16** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `FP8` | `8` | `2` | `T2I-1664x928-20step` | `20` | `1` | 2.07 | 90.71 | 8383.9 | 10.366 | **10.41** | **11.52** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `FP8` | `8` | `2` | `T2I-1664x928-50step` | `50` | `1` | 2.42 | 90.34 | 19.77 | 4.705 | **4.749** | **25.28** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-eager` | `FP8` | `8` | `2` | `T2I-1024x1024-20step` | `20` | `1` | 18.05 | 72.97 | 15.85 | 1.607 | **1.635** | **73.38** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `BF16` | `1` | `16` | `T2I-Diffusers-1024x1024` | `8` | `1` | - | - | - | - | **3.6** | **266.72** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `BF16` | `1` | `16` | `T2I-Diffusers-1536x1024` | `8` | `1` | - | - | - | - | **5.37** | **178.72** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `BF16` | `1` | `16` | `T2I-Diffusers-2048x2048` | `8` | `1` | - | - | - | - | **15.787** | **60.8** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `BF16` | `4` | `4` | `T2I-1024x1024-8step` | `8` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `BF16` | `4` | `4` | `T2I-1536x1024-8step` | `8` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `BF16` | `4` | `4` | `T2I-2048x2048-8step` | `8` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `BF16` | `4` | `4` | `TI2I-RefEdit-2Imgs-1024x1024-8step` | `8` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `BF16` | `4` | `4` | `TI2I-RefEdit-2Imgs-1536x1024-8step` | `8` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager` | `BF16` | `4` | `4` | `T2I-1024x1024-8step` | `8` | `1` | 14.37 | 28.02 | 1.08 | 0.354 | **0.382** | **628.6** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager` | `BF16` | `4` | `4` | `T2I-1536x1024-8step` | `8` | `1` | 14.0 | 24.53 | 1.11 | 0.38 | **0.425** | **565.08** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `FP8` | `4` | `4` | `T2I-1024x1024-8step` | `8` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `FP8` | `4` | `4` | `T2I-1536x1024-8step` | `8` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `FP8` | `4` | `4` | `T2I-2048x2048-8step` | `8` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `FP8` | `4` | `4` | `TI2I-RefEdit-2Imgs-1024x1024-8step` | `8` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `FP8` | `4` | `4` | `TI2I-RefEdit-2Imgs-1536x1024-8step` | `8` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager` | `FP8` | `4` | `4` | `T2I-1024x1024-8step` | `8` | `1` | 21.94 | 43.12 | 1.16 | 0.482 | **0.51** | **470.4** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `BF16` | `8` | `2` | `T2I-1024x1024-8step` | `8` | `1` | 2349.32 | 21.49 | 5760.81 | 8.396 | **8.424** | **14.24** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `BF16` | `8` | `2` | `T2I-1536x1024-8step` | `8` | `1` | 2.94 | 35.27 | 6418.44 | 6.872 | **6.917** | **17.34** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `BF16` | `8` | `2` | `T2I-2048x2048-8step` | `8` | `1` | 3.38 | 102.11 | 6939.72 | 8.209 | **8.332** | **14.4** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `BF16` | `8` | `2` | `TI2I-RefEdit-2Imgs-1024x1024-8step` | `8` | `1` | 2305.26 | 28.54 | 13.78 | 2.681 | **2.713** | **44.24** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `BF16` | `8` | `2` | `TI2I-RefEdit-2Imgs-1536x1024-8step` | `8` | `1` | 6.89 | 56.18 | 19.44 | 0.664 | **0.712** | **168.48** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager` | `BF16` | `8` | `2` | `T2I-1024x1024-8step` | `8` | `1` | 17.79 | 31.27 | 15.69 | 0.398 | **0.426** | **281.8** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager` | `BF16` | `8` | `2` | `T2I-1536x1024-8step` | `8` | `1` | 16.33 | 45.89 | 23.24 | 0.575 | **0.62** | **193.54** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `FP8` | `8` | `2` | `T2I-1024x1024-8step` | `8` | `1` | 2917.3 | 19.76 | 6173.59 | 9.363 | **9.391** | **12.78** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `FP8` | `8` | `2` | `T2I-1536x1024-8step` | `8` | `1` | 2.72 | 30.83 | 6148.01 | 6.566 | **6.611** | **18.16** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `FP8` | `8` | `2` | `T2I-2048x2048-8step` | `8` | `1` | 2.66 | 93.21 | 7040.96 | 8.239 | **8.361** | **14.36** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `FP8` | `8` | `2` | `TI2I-RefEdit-2Imgs-1024x1024-8step` | `8` | `1` | 3015.33 | 24.7 | 13.73 | 3.36 | **3.392** | **35.38** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `FP8` | `8` | `2` | `TI2I-RefEdit-2Imgs-1536x1024-8step` | `8` | `1` | 5.19 | 53.52 | 19.79 | 0.641 | **0.69** | **174.02** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager` | `FP8` | `8` | `2` | `T2I-1024x1024-8step` | `8` | `1` | 20.63 | 34.56 | 16.2 | 0.427 | **0.455** | **263.46** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `BF16` | `1` | `16` | `T2I-Diffusers-1024x1024` | `28` | `1` | - | - | - | - | **11.893** | **80.8** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `BF16` | `1` | `16` | `T2I-Diffusers-1536x1024` | `28` | `1` | - | - | - | - | **17.861** | **53.76** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `BF16` | `1` | `16` | `T2I-Diffusers-2048x2048` | `28` | `1` | - | - | - | - | **52.99** | **18.08** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback` | `BF16` | `4` | `4` | `T2I-1024x1024-28step` | `28` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback` | `BF16` | `4` | `4` | `T2I-1536x1024-28step` | `28` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback` | `BF16` | `4` | `4` | `T2I-2048x2048-28step` | `28` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback` | `BF16` | `4` | `4` | `TI2I-RefEdit-2Imgs-1024x1024-28step` | `28` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback` | `BF16` | `4` | `4` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `1` | 1.08 | 21.37 | 127.53 | 0.915 | **0.963** | **249.08** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager` | `BF16` | `4` | `4` | `T2I-1024x1024-28step` | `28` | `1` | 14.28 | 25.96 | 1.22 | 0.856 | **0.885** | **271.32** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager` | `BF16` | `4` | `4` | `T2I-1536x1024-28step` | `28` | `1` | 15.19 | 26.5 | 1.08 | 0.927 | **0.972** | **247.0** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback` | `FP8` | `4` | `4` | `T2I-1024x1024-28step` | `28` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback` | `FP8` | `4` | `4` | `T2I-1536x1024-28step` | `28` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback` | `FP8` | `4` | `4` | `T2I-2048x2048-28step` | `28` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback` | `FP8` | `4` | `4` | `TI2I-RefEdit-2Imgs-1024x1024-28step` | `28` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback` | `FP8` | `4` | `4` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `1` | 1.05 | 35.77 | 105.93 | 1.296 | **1.345** | **178.44** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager` | `FP8` | `4` | `4` | `T2I-1024x1024-28step` | `28` | `1` | - | - | - | - | **-** | **-** | `FAIL(RuntimeError: a required argument is in a failed state)` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled` | `BF16` | `8` | `2` | `T2I-1024x1024-28step` | `28` | `1` | 2393.84 | 40.1 | 6205.53 | 9.836 | **9.864** | **12.16** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled` | `BF16` | `8` | `2` | `T2I-1536x1024-28step` | `28` | `1` | 3.13 | 71.17 | 5945.71 | 8.11 | **8.155** | **14.72** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled` | `BF16` | `8` | `2` | `T2I-2048x2048-28step` | `28` | `1` | 3.11 | 204.12 | 6980.52 | 13.148 | **13.271** | **9.04** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled` | `BF16` | `8` | `2` | `TI2I-RefEdit-2Imgs-1024x1024-28step` | `28` | `1` | 2361.86 | 55.81 | 13.92 | 4.072 | **4.104** | **29.24** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled` | `BF16` | `8` | `2` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `1` | 6.91 | 101.13 | 20.13 | 3.046 | **3.095** | **38.78** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager` | `BF16` | `8` | `2` | `T2I-1024x1024-28step` | `28` | `1` | 16.33 | 54.03 | 15.92 | 1.659 | **1.687** | **71.12** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager` | `BF16` | `8` | `2` | `T2I-1536x1024-28step` | `28` | `1` | 12.99 | 83.72 | 22.88 | 2.549 | **2.593** | **46.28** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled` | `FP8` | `8` | `2` | `T2I-1024x1024-28step` | `28` | `1` | 2830.76 | 37.86 | 6031.46 | 10.036 | **10.064** | **11.92** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled` | `FP8` | `8` | `2` | `T2I-1536x1024-28step` | `28` | `1` | 2.46 | 66.77 | 6002.91 | 8.043 | **8.088** | **14.84** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled` | `FP8` | `8` | `2` | `T2I-2048x2048-28step` | `28` | `1` | 2.84 | 195.65 | 6739.1 | 12.67 | **12.792** | **9.38** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled` | `FP8` | `8` | `2` | `TI2I-RefEdit-2Imgs-1024x1024-28step` | `28` | `1` | 2926.72 | 50.18 | 14.31 | 4.479 | **4.511** | **26.6** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled` | `FP8` | `8` | `2` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `1` | 5.14 | 91.62 | 19.64 | 2.778 | **2.827** | **42.46** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager` | `FP8` | `8` | `2` | `T2I-1024x1024-28step` | `28` | `1` | 21.05 | 47.99 | 15.87 | 1.495 | **1.523** | **78.8** | `PASS` |
