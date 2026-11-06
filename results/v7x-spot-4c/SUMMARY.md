# Qwen-Image (`20B` MMDiT) & Qwen-Image-2.1 / 2.1-Turbo (`Qwen3-VL`) — `v7x-spot-4c` (`v7x`)

- **Updated (UTC):** `2026-11-06T14:08:16.151065+00:00`
- **Hardware Topology:** `1` hosts x `8` TPU cores/host = `8` TPU cores
- **Models Covered:** `Qwen/Qwen-Image` (`Qwen2.5-VL` + 60-block MMDiT), `Qwen/Qwen-Image-2.1-Turbo` (`8`-step `Qwen3-VL`), `Qwen/Qwen-Image-2.1` (`28`-step `Qwen3-VL`)

## Disaggregated CPU PNG / WebP Image Encoding Ablation (`multimodal-cpu-offload-svc:8090`)

| Resolution | Offload Active | On-Host CPU PNG (ms) | Disagg AVX-512 PNG RTT (ms) | Disagg WebP RTT (ms) | PNG Offload Speedup |
|---|:---:|---:|---:|---:|---:|
| `512x512` | `True` | 29.17 | **36.58** | 35.48 | `0.8x` |
| `1024x1024` | `True` | 112.66 | **142.84** | 137.68 | `0.79x` |
| `1328x1328` | `True` | 199.1 | **251.6** | 230.7 | `0.79x` |
| `1664x928` | `True` | 167.56 | **209.46** | 201.8 | `0.8x` |
| `1536x1024` | `True` | 169.27 | **214.39** | 205.83 | `0.79x` |
| `2048x2048` | `True` | 450.61 | **571.35** | 546.79 | `0.79x` |

## End-to-End Diffusion Serving & TorchTPU Compilation Results

| Model | Engine | Prec | TP/Ulysses | Cluster DP | Task / Resolution | Steps | Conc | Stage-1 VL (ms) | MMDiT Step (ms) | VAE (ms) | E2E Local CPU (s) | E2E P50 (s) | Cluster img/min | Status |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `2` | `T2I-OpenAI-HTTP-512x512` | `20` | `1` | - | - | - | - | **38.64** | **3.1** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `2` | `T2I-OpenAI-HTTP-1024x1024` | `20` | `1` | - | - | - | - | **46.674** | **2.57** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `2` | `T2I-OpenAI-HTTP-1024x1024` | `20` | `2` | - | - | - | - | **7.289** | **24.34** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `2` | `T2I-OpenAI-HTTP-1024x1024` | `20` | `4` | - | - | - | - | **13.021** | **4.48** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `2` | `T2I-OpenAI-HTTP-1024x1024` | `50` | `1` | - | - | - | - | **10.818** | **11.05** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `2` | `T2I-OpenAI-HTTP-1328x1328` | `20` | `1` | - | - | - | - | **37.466** | **3.2** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `2` | `T2I-OpenAI-HTTP-1328x1328` | `20` | `2` | - | - | - | - | **8.423** | **20.99** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `2` | `T2I-OpenAI-HTTP-1328x1328` | `50` | `1` | - | - | - | - | **12.844** | **9.3** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `2` | `T2I-OpenAI-HTTP-1664x928` | `20` | `1` | - | - | - | - | **54.283** | **2.21** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `2` | `T2I-OpenAI-HTTP-1664x928` | `20` | `2` | - | - | - | - | **8.506** | **20.73** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `4` | `2` | `T2I-OpenAI-HTTP-1664x928` | `50` | `1` | - | - | - | - | **11.967** | **9.98** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-sdpa` | `BF16` | `4` | `2` | `T2I-OpenAI-HTTP-1024x1024` | `20` | `1` | - | - | - | - | **6.582** | **18.13** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-sdpa` | `BF16` | `4` | `2` | `T2I-OpenAI-HTTP-1024x1024` | `20` | `2` | - | - | - | - | **5.109** | **35.07** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-sdpa` | `BF16` | `4` | `2` | `T2I-OpenAI-HTTP-1328x1328` | `20` | `1` | - | - | - | - | **5.752** | **20.64** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `FP8` | `4` | `2` | `ALL` | `20` | `1` | - | - | - | - | **-** | **-** | `DIAGNOSTIC(Exited rc=1: startup_future, startup_timeout)
  File "/usr/local/lib/python3.12/site-packages/vllm_omni/engine/omni_engine_base.py", line 549, in _wait_for_orchestrator_init
    startup_future.result(
  File "/usr/local/)` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `8` | `1` | `T2I-OpenAI-HTTP-512x512` | `20` | `1` | - | - | - | - | **6.743** | **8.88** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `8` | `1` | `T2I-OpenAI-HTTP-1024x1024` | `20` | `1` | - | - | - | - | **7.127** | **8.37** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `8` | `1` | `T2I-OpenAI-HTTP-1024x1024` | `20` | `2` | - | - | - | - | **5.779** | **15.06** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `8` | `1` | `T2I-OpenAI-HTTP-1024x1024` | `20` | `4` | - | - | - | - | **9.1** | **9.69** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `8` | `1` | `T2I-OpenAI-HTTP-1024x1024` | `50` | `1` | - | - | - | - | **9.437** | **6.33** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `8` | `1` | `T2I-OpenAI-HTTP-1328x1328` | `20` | `1` | - | - | - | - | **8.314** | **7.16** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `8` | `1` | `T2I-OpenAI-HTTP-1328x1328` | `20` | `2` | - | - | - | - | **6.356** | **13.4** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `8` | `1` | `T2I-OpenAI-HTTP-1328x1328` | `50` | `1` | - | - | - | - | **9.216** | **6.46** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `8` | `1` | `T2I-OpenAI-HTTP-1664x928` | `20` | `1` | - | - | - | - | **9.175** | **6.5** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `8` | `1` | `T2I-OpenAI-HTTP-1664x928` | `20` | `2` | - | - | - | - | **6.363** | **13.75** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `BF16` | `8` | `1` | `T2I-OpenAI-HTTP-1664x928` | `50` | `1` | - | - | - | - | **9.177** | **6.5** | `PASS` |
| `Qwen/Qwen-Image` | `vllm-omni-flash_attn` | `FP8` | `8` | `1` | `ALL` | `20` | `1` | - | - | - | - | **-** | **-** | `DIAGNOSTIC(Exited rc=1: startup_future, startup_timeout)
  File "/usr/local/lib/python3.12/site-packages/vllm_omni/engine/omni_engine_base.py", line 549, in _wait_for_orchestrator_init
    startup_future.result(
  File "/usr/local/)` |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback` | `FP8` | `4` | `2` | `T2I-512x512-20step` | `20` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback` | `FP8` | `4` | `2` | `T2I-512x512-50step` | `50` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback` | `FP8` | `4` | `2` | `T2I-1024x1024-20step` | `20` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback` | `FP8` | `4` | `2` | `T2I-1024x1024-50step` | `50` | `1` | 0.98 | 50.08 | 5.95 | 2.623 | **2.654** | **45.22** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback` | `FP8` | `4` | `2` | `T2I-1328x1328-20step` | `20` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback` | `FP8` | `4` | `2` | `T2I-1328x1328-50step` | `50` | `1` | 0.79 | 50.37 | 35.59 | 2.754 | **2.807** | **42.76** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback` | `FP8` | `4` | `2` | `T2I-1664x928-20step` | `20` | `1` | 1.02 | 49.93 | 8363.6 | 9.531 | **9.573** | **12.54** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-eager-fallback` | `FP8` | `4` | `2` | `T2I-1664x928-50step` | `50` | `1` | 0.77 | 51.69 | 25.03 | 2.778 | **2.82** | **42.56** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-eager` | `FP8` | `4` | `2` | `T2I-1024x1024-20step` | `20` | `1` | 18.52 | 79.49 | 1.16 | 1.722 | **1.752** | **68.48** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `BF16` | `8` | `1` | `T2I-512x512-20step` | `20` | `1` | 1705.98 | 21.28 | 5507.11 | 7.668 | **7.675** | **7.82** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `BF16` | `8` | `1` | `T2I-512x512-50step` | `50` | `1` | 2.23 | 19.12 | 6.48 | 0.994 | **1.001** | **59.92** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `BF16` | `8` | `1` | `T2I-1024x1024-20step` | `20` | `1` | 2.26 | 52.16 | 6022.37 | 7.18 | **7.211** | **8.32** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `BF16` | `8` | `1` | `T2I-1024x1024-50step` | `50` | `1` | 2.25 | 50.21 | 13.42 | 2.639 | **2.669** | **22.48** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `BF16` | `8` | `1` | `T2I-1328x1328-20step` | `20` | `1` | 3.03 | 107.53 | 6239.27 | 8.592 | **8.644** | **6.94** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `BF16` | `8` | `1` | `T2I-1328x1328-50step` | `50` | `1` | 2.69 | 105.31 | 23.41 | 5.49 | **5.543** | **10.82** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `BF16` | `8` | `1` | `T2I-1664x928-20step` | `20` | `1` | 2.91 | 87.85 | 8459.08 | 10.387 | **10.429** | **5.75** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `BF16` | `8` | `1` | `T2I-1664x928-50step` | `50` | `1` | 2.55 | 87.25 | 20.23 | 4.553 | **4.595** | **13.06** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-eager` | `BF16` | `8` | `1` | `T2I-1024x1024-20step` | `20` | `1` | 13.08 | 77.72 | 15.77 | 1.696 | **1.726** | **34.76** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-eager` | `BF16` | `8` | `1` | `T2I-1328x1328-20step` | `20` | `1` | 13.81 | 141.94 | 26.91 | 3.079 | **3.131** | **19.16** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `FP8` | `8` | `1` | `T2I-512x512-20step` | `20` | `1` | 2231.24 | 19.28 | 5535.73 | 8.182 | **8.189** | **7.33** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `FP8` | `8` | `1` | `T2I-512x512-50step` | `50` | `1` | 1.69 | 18.8 | 6.76 | 0.977 | **0.985** | **60.93** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `FP8` | `8` | `1` | `T2I-1024x1024-20step` | `20` | `1` | 2.09 | 54.17 | 6206.15 | 7.404 | **7.434** | **8.07** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `FP8` | `8` | `1` | `T2I-1024x1024-50step` | `50` | `1` | 1.85 | 54.9 | 14.19 | 2.874 | **2.904** | **20.66** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `FP8` | `8` | `1` | `T2I-1328x1328-20step` | `20` | `1` | 2.3 | 108.35 | 6539.69 | 8.908 | **8.961** | **6.7** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `FP8` | `8` | `1` | `T2I-1328x1328-50step` | `50` | `1` | 2.02 | 108.99 | 23.37 | 5.674 | **5.727** | **10.48** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `FP8` | `8` | `1` | `T2I-1664x928-20step` | `20` | `1` | 2.38 | 90.42 | 8698.29 | 10.677 | **10.719** | **5.6** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-compiled` | `FP8` | `8` | `1` | `T2I-1664x928-50step` | `50` | `1` | 2.23 | 88.8 | 20.26 | 4.63 | **4.672** | **12.84** | `PASS` |
| `Qwen/Qwen-Image` | `torchtpu-eager` | `FP8` | `8` | `1` | `T2I-1024x1024-20step` | `20` | `1` | 15.63 | 71.01 | 15.9 | 1.564 | **1.595** | **37.63** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `BF16` | `1` | `8` | `T2I-Diffusers-1024x1024` | `8` | `1` | - | - | - | - | **3.624** | **132.4** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `BF16` | `1` | `8` | `T2I-Diffusers-1536x1024` | `8` | `1` | - | - | - | - | **5.384** | **89.2** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `BF16` | `1` | `8` | `T2I-Diffusers-2048x2048` | `8` | `1` | - | - | - | - | **15.752** | **30.48** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `BF16` | `4` | `2` | `T2I-1024x1024-8step` | `8` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `BF16` | `4` | `2` | `T2I-1536x1024-8step` | `8` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `BF16` | `4` | `2` | `T2I-2048x2048-8step` | `8` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `BF16` | `4` | `2` | `TI2I-RefEdit-2Imgs-1024x1024-8step` | `8` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `BF16` | `4` | `2` | `TI2I-RefEdit-2Imgs-1536x1024-8step` | `8` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager` | `BF16` | `4` | `2` | `T2I-1024x1024-8step` | `8` | `1` | 13.92 | 27.03 | 0.84 | 0.344 | **0.374** | **321.0** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager` | `BF16` | `4` | `2` | `T2I-1536x1024-8step` | `8` | `1` | 14.68 | 25.67 | 1.31 | 0.391 | **0.436** | **275.42** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `FP8` | `4` | `2` | `T2I-1024x1024-8step` | `8` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `FP8` | `4` | `2` | `T2I-1536x1024-8step` | `8` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `FP8` | `4` | `2` | `T2I-2048x2048-8step` | `8` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `FP8` | `4` | `2` | `TI2I-RefEdit-2Imgs-1024x1024-8step` | `8` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `FP8` | `4` | `2` | `TI2I-RefEdit-2Imgs-1536x1024-8step` | `8` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager` | `FP8` | `4` | `2` | `T2I-1024x1024-8step` | `8` | `1` | 20.65 | 39.32 | 1.09 | 0.449 | **0.479** | **250.44** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `BF16` | `8` | `1` | `T2I-1024x1024-8step` | `8` | `1` | 2339.94 | 22.26 | 6260.96 | 8.892 | **8.922** | **6.73** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `BF16` | `8` | `1` | `T2I-1536x1024-8step` | `8` | `1` | 2.98 | 35.02 | 5905.13 | 6.358 | **6.403** | **9.37** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `BF16` | `8` | `1` | `T2I-2048x2048-8step` | `8` | `1` | 2.92 | 102.14 | 7127.43 | 8.398 | **8.519** | **7.04** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `BF16` | `8` | `1` | `TI2I-RefEdit-2Imgs-1024x1024-8step` | `8` | `1` | 2279.79 | 28.8 | 13.67 | 2.657 | **2.691** | **22.3** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `BF16` | `8` | `1` | `TI2I-RefEdit-2Imgs-1536x1024-8step` | `8` | `1` | 7.2 | 55.87 | 19.23 | 0.664 | **0.712** | **84.31** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager` | `BF16` | `8` | `1` | `T2I-1024x1024-8step` | `8` | `1` | 16.26 | 31.02 | 15.96 | 0.393 | **0.423** | **141.76** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager` | `BF16` | `8` | `1` | `T2I-1536x1024-8step` | `8` | `1` | 15.73 | 46.82 | 22.99 | 0.583 | **0.628** | **95.59** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `FP8` | `8` | `1` | `T2I-1024x1024-8step` | `8` | `1` | 2845.9 | 23.59 | 5690.17 | 8.837 | **8.868** | **6.77** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `FP8` | `8` | `1` | `T2I-1536x1024-8step` | `8` | `1` | 2.39 | 30.19 | 5741.56 | 6.155 | **6.2** | **9.68** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `FP8` | `8` | `1` | `T2I-2048x2048-8step` | `8` | `1` | 2.67 | 89.97 | 7021.82 | 8.195 | **8.316** | **7.22** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `FP8` | `8` | `1` | `TI2I-RefEdit-2Imgs-1024x1024-8step` | `8` | `1` | 2958.95 | 25.06 | 13.49 | 3.306 | **3.34** | **17.97** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-compiled` | `FP8` | `8` | `1` | `TI2I-RefEdit-2Imgs-1536x1024-8step` | `8` | `1` | 5.2 | 53.65 | 19.69 | 0.644 | **0.692** | **86.66** | `PASS` |
| `Qwen/Qwen-Image-2.1-Turbo` | `torchtpu-eager` | `FP8` | `8` | `1` | `T2I-1024x1024-8step` | `8` | `1` | 20.19 | 35.46 | 15.71 | 0.432 | **0.462** | **129.75** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `BF16` | `1` | `8` | `T2I-Diffusers-1024x1024` | `28` | `1` | - | - | - | - | **11.881** | **40.4** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `BF16` | `1` | `8` | `T2I-Diffusers-1536x1024` | `28` | `1` | - | - | - | - | **17.858** | **26.88** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `diffusers-QwenImage21Pipeline-single_core_tpu0` | `BF16` | `1` | `8` | `T2I-Diffusers-2048x2048` | `28` | `1` | - | - | - | - | **52.929** | **9.04** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback` | `BF16` | `4` | `2` | `T2I-1024x1024-28step` | `28` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback` | `BF16` | `4` | `2` | `T2I-1536x1024-28step` | `28` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback` | `BF16` | `4` | `2` | `T2I-2048x2048-28step` | `28` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback` | `BF16` | `4` | `2` | `TI2I-RefEdit-2Imgs-1024x1024-28step` | `28` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback` | `BF16` | `4` | `2` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `1` | 0.99 | 21.36 | 192.9 | 0.982 | **1.03** | **116.46** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager` | `BF16` | `4` | `2` | `T2I-1024x1024-28step` | `28` | `1` | 14.34 | 28.38 | 0.93 | 0.923 | **0.953** | **125.94** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager` | `BF16` | `4` | `2` | `T2I-1536x1024-28step` | `28` | `1` | 14.89 | 33.47 | 1.07 | 1.122 | **1.167** | **102.78** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback` | `FP8` | `4` | `2` | `T2I-1024x1024-28step` | `28` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback` | `FP8` | `4` | `2` | `T2I-1536x1024-28step` | `28` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback` | `FP8` | `4` | `2` | `T2I-2048x2048-28step` | `28` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback` | `FP8` | `4` | `2` | `TI2I-RefEdit-2Imgs-1024x1024-28step` | `28` | `1` | - | - | - | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (4)

While executing %submod_0 : [nu)` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager-fallback` | `FP8` | `4` | `2` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `1` | 0.95 | 26.49 | 113.28 | 1.046 | **1.094** | **109.68** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager` | `FP8` | `4` | `2` | `T2I-1024x1024-28step` | `28` | `1` | 24.22 | 46.64 | 1.04 | 1.444 | **1.474** | **81.4** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled` | `BF16` | `8` | `1` | `T2I-1024x1024-28step` | `28` | `1` | 2375.7 | 39.72 | 6153.88 | 9.754 | **9.785** | **6.13** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled` | `BF16` | `8` | `1` | `T2I-1536x1024-28step` | `28` | `1` | 3.41 | 71.26 | 6125.04 | 8.293 | **8.338** | **7.2** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled` | `BF16` | `8` | `1` | `T2I-2048x2048-28step` | `28` | `1` | 3.31 | 203.06 | 7036.05 | 13.176 | **13.296** | **4.51** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled` | `BF16` | `8` | `1` | `TI2I-RefEdit-2Imgs-1024x1024-28step` | `28` | `1` | 2317.58 | 55.0 | 14.37 | 4.006 | **4.039** | **14.86** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled` | `BF16` | `8` | `1` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `1` | 6.82 | 99.59 | 19.16 | 3.005 | **3.053** | **19.65** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager` | `BF16` | `8` | `1` | `T2I-1024x1024-28step` | `28` | `1` | 16.9 | 54.72 | 15.78 | 1.678 | **1.708** | **35.13** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager` | `BF16` | `8` | `1` | `T2I-1536x1024-28step` | `28` | `1` | 15.1 | 85.62 | 23.27 | 2.605 | **2.65** | **22.64** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled` | `FP8` | `8` | `1` | `T2I-1024x1024-28step` | `28` | `1` | 2805.01 | 37.97 | 6203.23 | 10.184 | **10.214** | **5.87** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled` | `FP8` | `8` | `1` | `T2I-1536x1024-28step` | `28` | `1` | 2.88 | 63.81 | 6122.41 | 8.081 | **8.126** | **7.38** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled` | `FP8` | `8` | `1` | `T2I-2048x2048-28step` | `28` | `1` | 2.68 | 194.89 | 7022.36 | 12.933 | **13.053** | **4.6** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled` | `FP8` | `8` | `1` | `TI2I-RefEdit-2Imgs-1024x1024-28step` | `28` | `1` | 2899.46 | 50.55 | 13.84 | 4.462 | **4.495** | **13.35** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-compiled` | `FP8` | `8` | `1` | `TI2I-RefEdit-2Imgs-1536x1024-28step` | `28` | `1` | 5.26 | 91.71 | 19.75 | 2.783 | **2.831** | **21.19** | `PASS` |
| `Qwen/Qwen-Image-2.1` | `torchtpu-eager` | `FP8` | `8` | `1` | `T2I-1024x1024-28step` | `28` | `1` | 18.89 | 48.79 | 16.0 | 1.514 | **1.544** | **38.86** | `PASS` |
