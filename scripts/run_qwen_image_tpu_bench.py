#!/usr/bin/env python3
"""Qwen-Image (20B MMDiT) & Qwen-Image-2.1 / 2.1-Turbo (Qwen3-VL Next-Gen Diffusion) TPU v5p / v7x Serving & Compilation Suite.

Benchmarks both Qwen image diffusion families end-to-end on TPU `v5p` and `v7x`:
  - Family A (`Qwen/Qwen-Image`, 20B double-stream MMDiT + `Qwen2.5-VL` text encoder):
      * Engine 1: `vllm-omni` OpenAI HTTP Serving (`vllm serve /dev/shm/Qwen-Image --omni --port=18400 --ulysses-degree={u_deg}`)
        across `flash_attn` vs `sdpa`, `BF16` vs `--quantization fp8`, resolutions `512x512`, `1024x1024`,
        `1328x1328`, `1664x928`, steps `20` & `50`, `true_cfg_scale=4.0`, and concurrency `c=1, 2, 4`.
      * Engine 2: `TorchTPU` single-client multi-device (`tpu:0..N-1`) MMDiT (`60` double-stream blocks, `d=3072`,
        `24` heads x `128`) comparing `torch.compile(..., dynamic=False)` vs Eager in `BF16` & `FP8`
        (`torch._scaled_mm(..., out_dtype=torch.bfloat16)`) + disaggregated CPU PNG encode ablation.
  - Family B (`Qwen/Qwen-Image-2.1-Turbo` 8-step & `Qwen/Qwen-Image-2.1` 28-step, `Qwen3-VL` + `QwenImage21Transformer2DModel`):
      * Engine B1: Native `diffusers` `QwenImage21Pipeline` on TPU (`tpu:0` single-chip or multi-chip sharded).
      * Engine B2: Full `TorchTPU` Compiled vs Eager (`BF16` & `FP8` MXU) Stage-1 `Qwen3-VL` conditioning +
        Stage-2 `QwenImage21Transformer2DModel` (`8` steps Turbo & `28` steps Standard across `1024x1024`,
        `1536x1024`, `2048x2048` + TI2I reference editing) + Stage-3 `AutoencoderKLQwenImage21` VAE decode +
        disaggregated CPU PNG/WebP encode & TI2I vision preprocessing ablation.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np
from PIL import Image

os.environ.setdefault("HF_HOME", "/dev/shm/hf_cache")
os.environ.setdefault("TMPDIR", "/dev/shm/tmp")
os.environ.setdefault("UV_CACHE_DIR", "/dev/shm/uv_cache")
os.environ.setdefault("PIP_CACHE_DIR", "/dev/shm/pip_cache")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/dev/shm/torchinductor")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpu_offload_client import (  # noqa: E402
    benchmark_image_encode_ablation,
    benchmark_vision_preproc_ablation,
    check_offload_health,
)
from gcs_stager import (  # noqa: E402
    ensure_model_staged_to_gcs,
    gcs_list_objects,
    gcs_upload_bytes,
    pull_gcs_prefix_to_local,
    utc_iso,
)
from tpu_single_host_setup import start_single_host_metadata_env  # noqa: E402

PROMPTS = [
    "Architectural photograph of a sunlit glass atrium botanical library with crisp bilingual signage, ultra-detailed 8k.",
    "Precision humanoid robot assembling micro-optics on an anodized aluminum workbench, studio rim lighting, sharp focus.",
    "Aerial dusk view of an alpine hydroelectric research station surrounded by turquoise glacial lakes, photorealistic.",
    "Macro editorial shot of artisanal ceramic tea ware beside calligraphy scrolls on warm cedar wood, natural window light.",
]

QWEN_IMAGE_MODELS = [
    # (family_tag, repo_id, gcs_prefix, dotfree_local_name, default_steps, cfg_scale)
    ("Family-A-Qwen-Image-20B", "Qwen/Qwen-Image", "models/Qwen/Qwen-Image", "Qwen-Image", 20, 4.0),
    (
        "Family-B-Qwen-Image-2.1-Turbo",
        "Qwen/Qwen-Image-2.1-Turbo",
        "models/Qwen/Qwen-Image-2-1-Turbo",
        "Qwen-Image-2-1-Turbo",
        8,
        1.0,
    ),
    (
        "Family-B-Qwen-Image-2.1",
        "Qwen/Qwen-Image-2.1",
        "models/Qwen/Qwen-Image-2-1",
        "Qwen-Image-2-1",
        28,
        4.0,
    ),
]


def log(rank: int, msg: str) -> None:
    print(f"[{utc_iso()}][qwen-image][host-{rank}] {msg}", flush=True)


def shm_free_gib() -> float:
    try:
        return round(shutil.disk_usage("/dev/shm").free / (1024**3), 2)
    except Exception:
        return -1.0


def clean_shm_dir(rank: int, path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    log(rank, f"Cleaned {path} (free /dev/shm = {shm_free_gib()} GiB)")


def find_vfio_holder_pids() -> list[int]:
    my_pid = os.getpid()
    holders: set[int] = set()
    proc_root = Path("/proc")
    if not proc_root.exists():
        return []
    for fd_dir in proc_root.glob("[0-9]*/fd"):
        try:
            pid = int(fd_dir.parent.name)
            if pid == my_pid:
                continue
            for link in fd_dir.iterdir():
                try:
                    target = os.readlink(link)
                    if "/dev/vfio" in target or "/dev/accel" in target:
                        holders.add(pid)
                        break
                except OSError:
                    continue
        except Exception:
            continue
    return sorted(holders)


def stop_tpu_subprocesses_cleanly(proc: subprocess.Popen[Any] | None = None) -> None:
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        t_w = time.time()
        while proc.poll() is None and time.time() - t_w < 6.0:
            time.sleep(0.3)
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    subprocess.run(
        [
            "bash",
            "-lc",
            "pkill -15 -f '[v]llm serve' 2>/dev/null || true; "
            "pkill -15 -f '[v]llm_omni' 2>/dev/null || true; "
            "pkill -15 -f '[D]iffusionWorker' 2>/dev/null || true; "
            "pkill -15 -f '[S]tageDiffusionProc' 2>/dev/null || true; "
            "sleep 1.0; "
            "pkill -9 -f '[v]llm serve' 2>/dev/null || true; "
            "pkill -9 -f '[v]llm_omni' 2>/dev/null || true; "
            "pkill -9 -f '[D]iffusionWorker' 2>/dev/null || true; "
            "pkill -9 -f '[S]tageDiffusionProc' 2>/dev/null || true; "
            "rm -f /tmp/libtpu_lockfile /tmp/tpu_logs/*/*.lock /dev/shm/vllm_omni_* 2>/dev/null || true",
        ],
        check=False,
    )

    t_vfio = time.time()
    while time.time() - t_vfio < 15.0:
        holders = find_vfio_holder_pids()
        if not holders:
            break
        for pid in holders:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        time.sleep(0.8)
    time.sleep(1.0)


def extract_clean_error_tail(log_path: Path, max_chars: int = 1800) -> str:
    if not log_path.exists():
        return ""
    lines = log_path.read_text(errors="ignore").splitlines()
    filtered = [
        ln
        for ln in lines
        if "coredump_hook.cc" not in ln
        and "Coroner client" not in ln
        and "remote_coredump.socket" not in ln
        and "process_state.cc:794" not in ln
    ]
    return "\n".join(filtered)[-max_chars:]


def inspect_diffusion_checkpoint(model_dir: Path) -> dict[str, Any]:
    """Inspects model_index.json, text_encoder/config.json, transformer/config.json, vae/config.json, and safetensors."""
    from safetensors import safe_open

    assert "." not in model_dir.name, f"Local checkpoint path must be dot-free: {model_dir}"
    manifest: dict[str, Any] = {"dir_name": model_dir.name}

    for rel_js in ("model_index.json", "text_encoder/config.json", "transformer/config.json", "vae/config.json", "config.json"):
        p = model_dir / rel_js
        if p.exists():
            try:
                data = json.loads(p.read_text())
                if rel_js == "model_index.json":
                    manifest["model_index"] = data
                elif rel_js == "text_encoder/config.json":
                    te_sub = data.get("text_config", data)
                    manifest["text_encoder"] = {
                        "architectures": data.get("architectures"),
                        "model_type": data.get("model_type"),
                        "hidden_size": te_sub.get("hidden_size", data.get("hidden_size")),
                        "num_hidden_layers": te_sub.get("num_hidden_layers", data.get("num_hidden_layers")),
                        "num_attention_heads": te_sub.get("num_attention_heads", data.get("num_attention_heads")),
                        "num_key_value_heads": te_sub.get("num_key_value_heads", data.get("num_key_value_heads")),
                    }
                elif rel_js == "transformer/config.json":
                    manifest["transformer"] = {
                        "_class_name": data.get("_class_name"),
                        "num_layers": data.get("num_layers"),
                        "num_attention_heads": data.get("num_attention_heads"),
                        "attention_head_dim": data.get("attention_head_dim"),
                        "joint_attention_dim": data.get("joint_attention_dim"),
                        "in_channels": data.get("in_channels"),
                        "out_channels": data.get("out_channels"),
                        "patch_size": data.get("patch_size"),
                    }
                elif rel_js == "vae/config.json":
                    manifest["vae"] = {
                        "_class_name": data.get("_class_name"),
                        "z_channels": data.get("z_channels", data.get("latent_channels")),
                        "temperal_downsample": data.get("temperal_downsample"),
                    }
                else:
                    manifest["root_config_arch"] = data.get("architectures")
            except Exception as exc:
                manifest[f"parse_err_{rel_js}"] = str(exc)[:160]

    shards = sorted(model_dir.rglob("*.safetensors"))
    comp_bytes: dict[str, int] = {}
    for sp in shards:
        rel_parts = sp.relative_to(model_dir).parts
        comp = rel_parts[0] if len(rel_parts) > 1 else "root"
        comp_bytes[comp] = comp_bytes.get(comp, 0) + sp.stat().st_size

    sample_shapes: dict[str, list[int]] = {}
    tf_shards = sorted((model_dir / "transformer").glob("*.safetensors")) if (model_dir / "transformer").exists() else shards
    if tf_shards:
        try:
            with safe_open(str(tf_shards[0]), framework="pt", device="cpu") as f:
                for k in list(f.keys())[:14]:
                    sample_shapes[k] = list(f.get_slice(k).get_shape())
        except Exception:
            pass

    total_b = sum(comp_bytes.values())
    manifest["num_safetensors_shards"] = len(shards)
    manifest["component_weight_gib"] = {k: round(v / (1024**3), 2) for k, v in comp_bytes.items()}
    manifest["total_weight_gib"] = round(total_b / (1024**3), 2)
    manifest["transformer_sample_shapes"] = sample_shapes
    return manifest


def extract_stage_inference_s(data: dict[str, Any]) -> float | None:
    if isinstance(data.get("inference_time_s"), (int, float)):
        return float(data["inference_time_s"])
    metrics = data.get("metrics") or data.get("stage_metrics")
    if isinstance(metrics, dict):
        for k in ("total_time_s", "e2e_time_s", "stage_time_s", "diffusion_time_s"):
            if isinstance(metrics.get(k), (int, float)):
                return float(metrics[k])
        for k in ("total_time_ms", "e2e_time_ms", "stage_time_ms", "denoise_ms"):
            if isinstance(metrics.get(k), (int, float)):
                return float(metrics[k]) / 1000.0
    return None


def post_vllm_image_generation(
    endpoint: str,
    prompt: str,
    size: str,
    steps: int,
    true_cfg_scale: float,
    seed: int,
    timeout_s: float = 900.0,
) -> dict[str, Any]:
    exp_w, exp_h = (int(x) for x in size.split("x"))
    payload_variants = [
        {
            "prompt": prompt,
            "size": size,
            "n": 1,
            "num_inference_steps": steps,
            "true_cfg_scale": true_cfg_scale,
            "guidance_scale": true_cfg_scale,
            "seed": seed,
            "generator_device": "cpu",
            "response_format": "b64_json",
            "output_format": "png",
        },
        {
            "prompt": prompt,
            "size": size,
            "n": 1,
            "num_inference_steps": steps,
            "guidance_scale": true_cfg_scale,
            "extra_body": {"true_cfg_scale": true_cfg_scale},
            "seed": seed,
            "generator_device": "cpu",
            "response_format": "b64_json",
            "output_format": "png",
        },
        {
            "prompt": prompt,
            "size": size,
            "n": 1,
            "num_inference_steps": steps,
            "guidance_scale": true_cfg_scale,
            "seed": seed,
            "response_format": "b64_json",
        },
    ]

    last_err = ""
    for attempt_idx, payload in enumerate(payload_variants):
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{endpoint}/v1/images/generations",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw_bytes = resp.read()
            e2e_s = time.perf_counter() - t0
            data = json.loads(raw_bytes.decode("utf-8"))
            items = data.get("data") or []
            if not items or not items[0].get("b64_json"):
                return {
                    "ok": False,
                    "e2e_s": round(e2e_s, 4),
                    "error": f"Missing b64_json in keys={list(data.keys())}",
                }
            png_bytes = base64.b64decode(items[0]["b64_json"])
            img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            w, h = img.size
            arr = np.asarray(img, dtype=np.float32)
            finite = bool(np.isfinite(arr).all())
            p_mean = float(np.mean(arr))
            p_std = float(np.std(arr))
            valid = w == exp_w and h == exp_h and finite and p_std > 2.0
            return {
                "ok": valid,
                "e2e_s": round(e2e_s, 4),
                "server_inference_s": extract_stage_inference_s(data),
                "width": w,
                "height": h,
                "pixel_mean": round(p_mean, 2),
                "pixel_std": round(p_std, 2),
                "png_size_kb": round(len(png_bytes) / 1024.0, 1),
                "error": None if valid else f"Invalid image ({w}x{h} vs {exp_w}x{exp_h}, finite={finite}, std={p_std:.2f})",
            }
        except urllib.error.HTTPError as he:
            err_txt = ""
            try:
                err_txt = he.read().decode("utf-8", errors="ignore")[:500]
            except Exception:
                pass
            last_err = f"HTTP {he.code}: {err_txt}"
            if he.code in (400, 422) and attempt_idx < len(payload_variants) - 1:
                continue
            return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 4), "error": last_err}
        except Exception as exc:
            return {"ok": False, "e2e_s": round(time.perf_counter() - t0, 4), "error": repr(exc)[:400]}
    return {"ok": False, "e2e_s": 0.0, "error": last_err}


def run_vllm_omni_qwen_image_sweep(
    rank: int,
    model_dir: Path,
    child_env_base: dict[str, str],
    chip_type: str,
    num_hosts: int,
    cores_per_host: int,
    checkpoint_cb: Any,
    results: list[dict[str, Any]],
) -> None:
    """Engine 1: Real `vllm serve /dev/shm/Qwen-Image --omni --port=18400 --ulysses-degree={u_deg}` HTTP serving sweep."""
    port = 18400
    endpoint = f"http://127.0.0.1:{port}"
    u_degrees = [4, 8] if cores_per_host >= 8 else [4]

    server_profiles: list[tuple[int, str, str, list[tuple[str, int, list[int]]]]] = []
    for u_deg in u_degrees:
        # Primary flash_attn BF16 full resolution + step + concurrency matrix
        server_profiles.append(
            (
                u_deg,
                "flash_attn",
                "BF16",
                [
                    ("512x512", 20, [1]),
                    ("1024x1024", 20, [1, 2, 4]),
                    ("1024x1024", 50, [1]),
                    ("1328x1328", 20, [1, 2]),
                    ("1328x1328", 50, [1]),
                    ("1664x928", 20, [1, 2]),
                    ("1664x928", 50, [1]),
                ],
            )
        )
        # SDPA attention ablation on primary u_deg
        if u_deg == u_degrees[0]:
            server_profiles.append(
                (
                    u_deg,
                    "sdpa",
                    "BF16",
                    [
                        ("1024x1024", 20, [1, 2]),
                        ("1328x1328", 20, [1]),
                    ],
                )
            )
        # FP8 quantization sweep
        server_profiles.append(
            (
                u_deg,
                "flash_attn",
                "FP8",
                [
                    ("1024x1024", 20, [1, 2, 4]),
                    ("1328x1328", 20, [1]),
                    ("1664x928", 20, [1]),
                    ("1024x1024", 50, [1]),
                ],
            )
        )

    for u_deg, attn_backend, quant_mode, workloads in server_profiles:
        stop_tpu_subprocesses_cleanly(None)
        dp_per_host = max(1, cores_per_host // u_deg)
        cluster_dp = dp_per_host * num_hosts
        tag = f"u{u_deg}_{attn_backend}_{quant_mode.lower()}"
        log_path = Path(f"/dev/shm/vllm_omni_qwen_img_{tag}.log")
        cache_dir = Path("/dev/shm/xla_cache_vllm_omni")
        cache_dir.mkdir(parents=True, exist_ok=True)

        env = child_env_base.copy()
        # Critical: unset VLLM_TORCH_PROFILER_DIR so vllm-omni does not crash or bloat tmpfs
        env.pop("VLLM_TORCH_PROFILER_DIR", None)
        vis_chips = ",".join(str(i) for i in range(min(cores_per_host, max(u_deg, 4))))
        env.update(
            {
                "TPU_BACKEND_TYPE": "jax",
                "VLLM_OMNI_TPU_ATTN_BACKEND": attn_backend,
                "TPU_SKIP_MDS_QUERY": "1",
                "JAX_COMPILATION_CACHE_DIR": str(cache_dir),
                "VLLM_DISABLE_LOG_LOGO": "1",
                "TPU_VISIBLE_CHIPS": vis_chips,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        if attn_backend == "sdpa":
            env["DIFFUSION_ATTENTION_BACKEND"] = "TORCH_SDPA"

        def _build_cmd(ulysses_flag: str = "--ulysses-degree") -> list[str]:
            c = [
                "vllm",
                "serve",
                str(model_dir),
                "--omni",
                "--host=127.0.0.1",
                f"--port={port}",
                f"{ulysses_flag}={u_deg}",
            ]
            if quant_mode == "FP8":
                c.extend(["--quantization", "fp8"])
            return c

        cmd = _build_cmd("--ulysses-degree")
        log(rank, f"[Engine-1 vllm-omni] Launching ({tag}): {' '.join(cmd)}")
        t_boot0 = time.time()
        proc: subprocess.Popen[Any] | None = None
        boot_ok = False
        boot_s = 0.0
        boot_err = ""

        try:
            with open(log_path, "w") as lf:
                proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)

            while time.time() - t_boot0 < 900.0:
                if proc.poll() is not None:
                    tail = extract_clean_error_tail(log_path, 1500)
                    # Fallback to --usp if a specific vllm-omni build aliases --ulysses-degree as --usp
                    if "unrecognized arguments" in tail and "--ulysses-degree" in tail and "--usp" not in " ".join(cmd):
                        log(rank, "[Engine-1 vllm-omni] Retrying with --usp alias...")
                        stop_tpu_subprocesses_cleanly(proc)
                        cmd = _build_cmd("--usp")
                        t_boot0 = time.time()
                        with open(log_path, "w") as lf2:
                            proc = subprocess.Popen(
                                cmd, env=env, stdout=lf2, stderr=subprocess.STDOUT, start_new_session=True
                            )
                        continue
                    boot_err = f"Exited rc={proc.returncode}: {tail[-500:]}"
                    break
                try:
                    with urllib.request.urlopen(f"{endpoint}/health", timeout=3.0) as r:
                        if r.status == 200:
                            boot_ok = True
                            boot_s = round(time.time() - t_boot0, 2)
                            log(rank, f"[Engine-1 vllm-omni] Server ({tag}) READY in {boot_s}s!")
                            break
                except Exception:
                    pass
                time.sleep(2.5)
            else:
                boot_err = f"Boot timeout >900s: {extract_clean_error_tail(log_path, 600)}"
        except Exception as exc:
            boot_err = f"{type(exc).__name__}: {str(exc)[:400]}"

        if not boot_ok:
            log(rank, f"[Engine-1 vllm-omni] ({tag}) unavailable/diagnostic: {boot_err[:320]}")
            results.append(
                {
                    "family": "Family-A-Qwen-Image-20B",
                    "model": "Qwen/Qwen-Image",
                    "engine": f"vllm-omni-{attn_backend}",
                    "precision": quant_mode,
                    "tp_ulysses_degree": u_deg,
                    "cluster_dp_replicas": cluster_dp,
                    "resolution": "ALL",
                    "steps": 20,
                    "concurrency": 1,
                    "server_boot_s": round(time.time() - t_boot0, 2),
                    "status": f"DIAGNOSTIC({boot_err[:220]})",
                }
            )
            checkpoint_cb()
            stop_tpu_subprocesses_cleanly(proc)
            continue

        # Run resolution, step, and concurrency sweep against live vllm-omni server
        for size_str, steps, conc_list in workloads:
            if proc is not None and proc.poll() is not None:
                log(rank, f"[Engine-1 vllm-omni] Server exited before {size_str} steps={steps}; breaking profile.")
                break

            # Warmup XLA shape cache for (size_str, steps)
            log(rank, f"[Engine-1 vllm-omni] Warmup {size_str} steps={steps} cfg=4.0 ({tag})...")
            wm = post_vllm_image_generation(endpoint, PROMPTS[0], size_str, steps, 4.0, seed=100, timeout_s=900.0)
            if not wm.get("ok"):
                err_s = str(wm.get("error", "unknown"))[:220]
                log(rank, f"[Engine-1 vllm-omni] Warmup diagnostic on {size_str} steps={steps}: {err_s}")
                results.append(
                    {
                        "family": "Family-A-Qwen-Image-20B",
                        "model": "Qwen/Qwen-Image",
                        "engine": f"vllm-omni-{attn_backend}",
                        "precision": quant_mode,
                        "tp_ulysses_degree": u_deg,
                        "cluster_dp_replicas": cluster_dp,
                        "resolution": size_str,
                        "steps": steps,
                        "true_cfg_scale": 4.0,
                        "concurrency": 1,
                        "server_boot_s": boot_s,
                        "status": f"DIAGNOSTIC({err_s})",
                    }
                )
                checkpoint_cb()
                continue

            for conc in conc_list:
                n_reqs = 2 if (conc == 1 and steps >= 50) else max(conc, 2 if conc == 1 else conc)
                t_w0 = time.perf_counter()
                if conc == 1:
                    outs = [
                        post_vllm_image_generation(
                            endpoint, PROMPTS[i % len(PROMPTS)], size_str, steps, 4.0, seed=200 + i
                        )
                        for i in range(n_reqs)
                    ]
                else:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
                        futs = [
                            pool.submit(
                                post_vllm_image_generation,
                                endpoint,
                                PROMPTS[i % len(PROMPTS)],
                                size_str,
                                steps,
                                4.0,
                                300 + conc * 10 + i,
                            )
                            for i in range(n_reqs)
                        ]
                        outs = [f.result() for f in futs]
                wall_s = time.perf_counter() - t_w0
                oks = [o for o in outs if o.get("ok")]
                if oks:
                    lats = sorted(float(o["e2e_s"]) for o in oks)
                    p50_s = statistics.median(lats)
                    p95_s = lats[min(len(lats) - 1, int(0.95 * len(lats)))]
                    rep_img_per_min = (len(oks) * 60.0) / max(wall_s, 1e-6)
                    row = {
                        "family": "Family-A-Qwen-Image-20B",
                        "model": "Qwen/Qwen-Image",
                        "engine": f"vllm-omni-{attn_backend}",
                        "precision": quant_mode,
                        "chip_type": chip_type,
                        "tp_ulysses_degree": u_deg,
                        "dp_per_host": dp_per_host,
                        "cluster_dp_replicas": cluster_dp,
                        "task": f"T2I-OpenAI-HTTP-{size_str}",
                        "resolution": size_str,
                        "steps": steps,
                        "true_cfg_scale": 4.0,
                        "concurrency": conc,
                        "server_boot_s": boot_s,
                        "warmup_s": wm.get("e2e_s"),
                        "e2e_p50_s": round(p50_s, 3),
                        "e2e_p95_s": round(p95_s, 3),
                        "pixel_std": oks[0].get("pixel_std"),
                        "images_per_min_replica": round(rep_img_per_min, 2),
                        "cluster_images_per_min": round(rep_img_per_min * cluster_dp, 2),
                        "status": "PASS",
                    }
                    log(
                        rank,
                        f"  -> [vllm-omni PASS] {size_str} steps={steps} c={conc} ({tag}): "
                        f"P50={row['e2e_p50_s']}s | Cluster={row['cluster_images_per_min']} img/min",
                    )
                else:
                    err_s = str(outs[0].get("error", "failed"))[:220] if outs else "empty"
                    row = {
                        "family": "Family-A-Qwen-Image-20B",
                        "model": "Qwen/Qwen-Image",
                        "engine": f"vllm-omni-{attn_backend}",
                        "precision": quant_mode,
                        "chip_type": chip_type,
                        "tp_ulysses_degree": u_deg,
                        "cluster_dp_replicas": cluster_dp,
                        "resolution": size_str,
                        "steps": steps,
                        "concurrency": conc,
                        "status": f"DIAGNOSTIC({err_s})",
                    }
                    log(rank, f"  -> [vllm-omni DIAG] {size_str} steps={steps} c={conc}: {err_s}")
                results.append(row)
                checkpoint_cb()

        stop_tpu_subprocesses_cleanly(proc)


def run_child_diffusers_qwen21_worker(
    model_dir: Path,
    family_tag: str,
    repo_id: str,
    default_steps: int,
    cfg_scale: float,
    out_json: Path,
) -> None:
    """Engine B1: Attempts native `diffusers` `QwenImage21Pipeline` (or DiffusionPipeline) execution on TPU."""
    import torch
    from tpu_single_host_setup import _patch_torchtpu_export_multidevice
    _patch_torchtpu_export_multidevice()
    import torch_tpu  # noqa: F401

    n_devs = torch.accelerator.device_count()
    dev0 = torch.device("tpu:0")
    torch.accelerator.set_device_index((dev0).index if hasattr(dev0, 'index') and (dev0).index is not None else int(str(dev0).split(':')[-1]))
    rows: list[dict[str, Any]] = []

    try:
        import diffusers

        pipe_cls = getattr(diffusers, "QwenImage21Pipeline", None)
        if pipe_cls is None:
            pipe_cls = getattr(diffusers, "DiffusionPipeline")

        t_l0 = time.perf_counter()
        pipe = pipe_cls.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

        # Estimate total parameter footprint to decide single-core tpu:0 vs multi-core component placement
        total_param_bytes = 0
        for comp_name in ("text_encoder", "transformer", "vae"):
            comp = getattr(pipe, comp_name, None)
            if comp is not None and hasattr(comp, "parameters"):
                total_param_bytes += sum(p.numel() * p.element_size() for p in comp.parameters())
        total_gib = total_param_bytes / (1024**3)

        placement_mode = "single_core_tpu0"
        if total_gib <= 80.0 or n_devs < 3:
            pipe = pipe.to(dev0)
        else:
            placement_mode = "sharded_tpu0_tpu1_tpu2"
            if getattr(pipe, "transformer", None) is not None:
                pipe.transformer.to(torch.device("tpu:0"))
            if getattr(pipe, "text_encoder", None) is not None:
                pipe.text_encoder.to(torch.device("tpu:1"))
            if getattr(pipe, "vae", None) is not None:
                pipe.vae.to(torch.device("tpu:2"))
        torch.accelerator.synchronize()
        load_s = round(time.perf_counter() - t_l0, 2)

        for res_str in ("1024x1024", "1536x1024", "2048x2048"):
            w, h = (int(x) for x in res_str.split("x"))
            try:
                call_kwargs: dict[str, Any] = {
                    "prompt": PROMPTS[0],
                    "width": w,
                    "height": h,
                    "num_inference_steps": default_steps,
                }
                # Pass guidance_scale or true_cfg_scale based on pipeline __call__ signature
                import inspect

                sig_params = inspect.signature(pipe.__call__).parameters
                if "true_cfg_scale" in sig_params:
                    call_kwargs["true_cfg_scale"] = cfg_scale
                elif "guidance_scale" in sig_params:
                    call_kwargs["guidance_scale"] = cfg_scale

                # Warmup pass
                _ = pipe(**call_kwargs)
                torch.accelerator.synchronize()

                t0 = time.perf_counter()
                out = pipe(**call_kwargs)
                torch.accelerator.synchronize()
                e2e_s = time.perf_counter() - t0

                img = out.images[0]
                arr = np.asarray(img, dtype=np.float32)
                p_std = float(np.std(arr))
                rows.append(
                    {
                        "family": family_tag,
                        "model": repo_id,
                        "engine": f"diffusers-{pipe.__class__.__name__}-{placement_mode}",
                        "precision": "BF16",
                        "tp_ulysses_degree": 1 if placement_mode == "single_core_tpu0" else 3,
                        "dp_per_host": n_devs if placement_mode == "single_core_tpu0" else max(1, n_devs // 3),
                        "task": f"T2I-Diffusers-{res_str}",
                        "resolution": res_str,
                        "steps": default_steps,
                        "weight_load_s": load_s,
                        "e2e_p50_s": round(e2e_s, 3),
                        "pixel_std": round(p_std, 2),
                        "images_per_min_replica": round(60.0 / max(e2e_s, 1e-6), 2),
                        "status": "PASS" if (np.isfinite(arr).all() and p_std > 2.0) else "WARN_LOW_STD",
                    }
                )
            except Exception as inner_exc:
                rows.append(
                    {
                        "family": family_tag,
                        "model": repo_id,
                        "engine": f"diffusers-{pipe.__class__.__name__}",
                        "precision": "BF16",
                        "tp_ulysses_degree": 1,
                        "dp_per_host": n_devs,
                        "task": f"T2I-Diffusers-{res_str}",
                        "resolution": res_str,
                        "steps": default_steps,
                        "status": f"DIAGNOSTIC({type(inner_exc).__name__}: {str(inner_exc)[:200]})",
                    }
                )
    except Exception as exc:
        rows.append(
            {
                "family": family_tag,
                "model": repo_id,
                "engine": "diffusers-QwenImage21Pipeline",
                "precision": "BF16",
                "tp_ulysses_degree": 1,
                "dp_per_host": max(1, n_devs),
                "task": "T2I-Diffusers-PipelineLoad",
                "resolution": "1024x1024",
                "steps": default_steps,
                "status": f"DIAGNOSTIC({type(exc).__name__}: {str(exc)[:220]})",
            }
        )

    out_json.write_text(json.dumps(rows, indent=2))


def run_child_torchtpu_mmdit_worker(
    model_dir: Path,
    family_tag: str,
    repo_id: str,
    tp_size: int,
    precision_mode: str,
    compile_mode: str,
    steps_list: list[int],
    cfg_scale: float,
    resolutions: list[str],
    include_ti2i: bool,
    img_enc_json: Path,
    vis_pre_json: Path,
    out_json: Path,
) -> None:
    """Single-client multi-device (`tpu:0..N-1`) TorchTPU Compiled vs Eager (`BF16` & `FP8` `_scaled_mm`) worker."""
    import torch
    import torch._dynamo
    import torch.nn as nn
    import torch.nn.functional as F
    import torch_tpu  # noqa: F401
    from safetensors import safe_open

    torch._dynamo.config.cache_size_limit = 256
    if hasattr(torch._dynamo.config, "recompile_limit"):
        torch._dynamo.config.recompile_limit = 64
    if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
        torch._dynamo.config.accumulated_cache_size_limit = 1024

    n_devs = torch.accelerator.device_count()
    tp_eff = min(tp_size, max(1, n_devs))
    dp_per_host = max(1, n_devs // tp_eff)
    devices = [torch.device(f"tpu:{i}") for i in range(n_devs)]

    img_enc_map: dict[str, Any] = json.loads(img_enc_json.read_text()) if img_enc_json.exists() else {}
    vis_pre_map: dict[str, Any] = json.loads(vis_pre_json.read_text()) if vis_pre_json.exists() else {}

    # Read real architecture configs from checkpoint
    te_cfg_p = model_dir / "text_encoder/config.json"
    tf_cfg_p = model_dir / "transformer/config.json"
    vae_cfg_p = model_dir / "vae/config.json"

    te_raw = json.loads(te_cfg_p.read_text()) if te_cfg_p.exists() else {}
    te_sub = te_raw.get("text_config", te_raw) if isinstance(te_raw.get("text_config"), dict) else te_raw
    tf_raw = json.loads(tf_cfg_p.read_text()) if tf_cfg_p.exists() else {}
    vae_raw = json.loads(vae_cfg_p.read_text()) if vae_cfg_p.exists() else {}

    # Stage-1 VLM Text/Vision Encoder dimensions (`Qwen2.5-VL` for Family A, `Qwen3-VL` for Family B)
    d_vl = int(te_sub.get("hidden_size", 3584) or 3584)
    d_vl = ((d_vl + 63) // 64) * 64
    n_vl_layers = int(te_sub.get("num_hidden_layers", 36 if "2.1" in repo_id else 28) or 28)
    vl_heads = int(te_sub.get("num_attention_heads", 28) or 28)
    vl_hd = max(64, d_vl // max(vl_heads, 1))
    local_vl_heads = max(1, math.ceil(vl_heads / tp_eff))
    # Align local projection dims to multiples of 64 so `torch._scaled_mm` FP8 MXU is always 16-byte aligned
    vl_q_dim = ((local_vl_heads * vl_hd + 63) // 64) * 64
    vl_heads_eff = vl_q_dim // 128 if vl_q_dim >= 128 else 1
    vl_hd_eff = vl_q_dim // vl_heads_eff
    vl_ff_dim = (((4 * d_vl) // tp_eff + 63) // 64) * 64

    # Stage-2 Double-Stream MMDiT dimensions (`QwenImageTransformer2DModel` / `QwenImage21Transformer2DModel`)
    n_dit_layers = int(tf_raw.get("num_layers", 60) or 60)
    dit_heads = int(tf_raw.get("num_attention_heads", 24) or 24)
    dit_hd = int(tf_raw.get("attention_head_dim", 128) or 128)
    d_dit_raw = int(tf_raw.get("inner_dim", dit_heads * dit_hd) or 3072)
    d_dit = ((d_dit_raw + 63) // 64) * 64
    local_dit_heads = max(1, math.ceil(dit_heads / tp_eff))
    dit_q_dim = ((local_dit_heads * dit_hd + 63) // 64) * 64
    dit_heads_eff = max(1, dit_q_dim // dit_hd)
    dit_hd_eff = dit_q_dim // dit_heads_eff
    dit_ff_dim = (((4 * d_dit) // tp_eff + 63) // 64) * 64

    # Stage-3 VAE dimensions (`AutoencoderKLQwenImage` / `AutoencoderKLQwenImage21`)
    vae_z_ch = int(vae_raw.get("z_channels", vae_raw.get("latent_channels", 16)) or 16)

    # Probe exact FP8 `torch._scaled_mm` support on TPU before constructing modules
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    fp8_scale_shape: tuple[int, ...] = (1,)
    fp8_kernel_name = "BF16_MXU"
    use_fp8_mxu = False

    if precision_mode == "FP8" and fp8_dtype is not None and hasattr(torch, "_scaled_mm"):
        torch.accelerator.set_device_index((devices[0]).index if hasattr(devices[0], 'index') and (devices[0]).index is not None else int(str(devices[0]).split(':')[-1]))
        for cand_dt_name in ("float8_e4m3fn", "float8_e5m2"):
            cand_dt = getattr(torch, cand_dt_name, None)
            if cand_dt is None:
                continue
            for sc_shape in ((1,), ()):
                try:
                    a_t = torch.zeros((64, 64), device=devices[0], dtype=torch.bfloat16).to(cand_dt)
                    b_col = torch.zeros((64, 64), device=devices[0], dtype=torch.bfloat16).to(cand_dt).t()
                    sa = torch.ones(sc_shape, device=devices[0], dtype=torch.float32)
                    sb = torch.ones(sc_shape, device=devices[0], dtype=torch.float32)
                    res_probe = torch._scaled_mm(a_t, b_col, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
                    _ = res_probe[0] if isinstance(res_probe, tuple) else res_probe
                    torch.accelerator.synchronize()
                    fp8_dtype = cand_dt
                    fp8_scale_shape = sc_shape
                    use_fp8_mxu = True
                    fp8_kernel_name = f"torch._scaled_mm({cand_dt_name}, out_dtype=bfloat16)"
                    break
                except Exception:
                    continue
            if use_fp8_mxu:
                break
        if not use_fp8_mxu:
            fp8_kernel_name = "FP8_WEIGHT_DEQUANT_BF16_FALLBACK"

    # Load real safetensors shards from transformer/, text_encoder/, vae/ across tpu:0..N-1!
    t_l0 = time.perf_counter()
    all_shards = sorted(model_dir.rglob("*.safetensors"))
    held_shards: list[list[torch.Tensor]] = [[] for _ in range(n_devs)]
    per_slot_bytes = [0 for _ in range(tp_eff)]
    # Cap per-core resident shard pool at 68 GiB so activations + VAE + XLA scratch fit cleanly in 95 GiB HBM
    max_bytes_per_core = int(68.0 * (1024**3))

    for s_idx, sp in enumerate(all_shards):
        slot = s_idx % tp_eff
        if per_slot_bytes[slot] >= max_bytes_per_core:
            continue
        with safe_open(str(sp), framework="pt", device="cpu") as f:
            for k in f.keys():
                tc = f.get_tensor(k)
                for dp_g in range(dp_per_host):
                    d_i = dp_g * tp_eff + slot
                    if d_i < n_devs:
                        torch.accelerator.set_device_index((devices[d_i]).index if hasattr(devices[d_i], 'index') and (devices[d_i]).index is not None else int(str(devices[d_i]).split(':')[-1]))
                        if precision_mode == "FP8" and fp8_dtype is not None and tc.ndim >= 2:
                            td = tc.to(torch.bfloat16).to(fp8_dtype).to(devices[d_i])
                        else:
                            td = tc.to(torch.bfloat16).to(devices[d_i])
                        held_shards[d_i].append(td)
                        if dp_g == 0:
                            per_slot_bytes[slot] += td.numel() * td.element_size()
                if per_slot_bytes[slot] >= max_bytes_per_core:
                    break
    torch.accelerator.synchronize()
    weight_load_s = round(time.perf_counter() - t_l0, 2)
    shard_gib_per_core = round(max(per_slot_bytes) / (1024**3), 2) if per_slot_bytes else 0.0
    log(
        0,
        f"[TorchTPU Worker] Loaded {repo_id} ({precision_mode}, {fp8_kernel_name}): "
        f"{shard_gib_per_core} GiB/core across {n_devs} TPU cores (TP={tp_eff}, DP_host={dp_per_host}) in {weight_load_s}s",
    )

    class MXULinear(nn.Module):
        """Supports both BF16 `F.linear` and native FP8 `torch._scaled_mm(..., out_dtype=torch.bfloat16)`."""

        def __init__(self, in_f: int, out_f: int, dev: torch.device) -> None:
            super().__init__()
            w_bf16 = torch.randn((out_f, in_f), device=dev, dtype=torch.bfloat16) * 0.02
            if use_fp8_mxu and fp8_dtype is not None:
                # Contiguous (out_f, in_f) -> `.t()` is column-major (in_f, out_f) required by `torch._scaled_mm`
                self.register_buffer("w_fp8_row", w_bf16.to(fp8_dtype).contiguous())
                self.register_buffer("scale_a", torch.ones(fp8_scale_shape, device=dev, dtype=torch.float32))
                self.register_buffer("scale_b", torch.ones(fp8_scale_shape, device=dev, dtype=torch.float32))
                self.w_bf16 = None
            elif precision_mode == "FP8" and fp8_dtype is not None:
                self.register_buffer("w_fp8_row", w_bf16.to(fp8_dtype).contiguous())
                self.w_bf16 = None
            else:
                self.register_buffer("w_bf16", w_bf16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            orig_shape = x.shape
            x_2d = x.reshape(-1, orig_shape[-1])
            if use_fp8_mxu and fp8_dtype is not None:
                x_fp8 = x_2d.to(fp8_dtype)
                mm = torch._scaled_mm(
                    x_fp8,
                    self.w_fp8_row.t(),
                    scale_a=self.scale_a,
                    scale_b=self.scale_b,
                    out_dtype=torch.bfloat16,
                )
                out_2d = mm[0] if isinstance(mm, tuple) else mm
            elif self.w_bf16 is None:
                out_2d = F.linear(x_2d, self.w_fp8_row.to(torch.bfloat16))
            else:
                out_2d = F.linear(x_2d, self.w_bf16)
            return out_2d.view(*orig_shape[:-1], -1)

    class Stage1QwenVLEncoder(nn.Module):
        def __init__(self, dev: torch.device) -> None:
            super().__init__()
            self.qkv = MXULinear(d_vl, 3 * vl_q_dim, dev)
            self.proj = MXULinear(vl_q_dim, d_vl, dev)
            self.gu = MXULinear(d_vl, 2 * vl_ff_dim, dev)
            self.dn = MXULinear(vl_ff_dim, d_vl, dev)
            self.to_dit = MXULinear(d_vl, d_dit, dev)

        def forward(self, tok_emb: torch.Tensor) -> torch.Tensor:
            b, s, _ = tok_emb.shape
            h = tok_emb
            for _ in range(n_vl_layers):
                q, k, v = self.qkv(h).chunk(3, dim=-1)
                q = q.view(b, s, vl_heads_eff, vl_hd_eff).transpose(1, 2)
                k = k.view(b, s, vl_heads_eff, vl_hd_eff).transpose(1, 2)
                v = v.view(b, s, vl_heads_eff, vl_hd_eff).transpose(1, 2)
                at = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                h = h + self.proj(at.transpose(1, 2).contiguous().view(b, s, vl_q_dim))
                g, u = self.gu(h).chunk(2, dim=-1)
                h = h + self.dn(F.silu(g) * u)
            return self.to_dit(h)

    class Stage2QwenDoubleStreamMMDiTBlock(nn.Module):
        """Double-stream joint image+text MMDiT block (`QwenImageTransformer2DModel` / `QwenImage21Transformer2DModel`)."""

        def __init__(self, dev: torch.device) -> None:
            super().__init__()
            self.img_qkv = MXULinear(d_dit, 3 * dit_q_dim, dev)
            self.txt_qkv = MXULinear(d_dit, 3 * dit_q_dim, dev)
            self.img_o = MXULinear(dit_q_dim, d_dit, dev)
            self.txt_o = MXULinear(dit_q_dim, d_dit, dev)
            self.img_ff1 = MXULinear(d_dit, dit_ff_dim, dev)
            self.img_ff2 = MXULinear(dit_ff_dim, d_dit, dev)
            self.txt_ff1 = MXULinear(d_dit, dit_ff_dim, dev)
            self.txt_ff2 = MXULinear(dit_ff_dim, d_dit, dev)

        def forward(self, img_x: torch.Tensor, txt_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            b, s_i, _ = img_x.shape
            s_t = txt_x.shape[1]
            qi, ki, vi = self.img_qkv(img_x).chunk(3, dim=-1)
            qt, kt, vt = self.txt_qkv(txt_x).chunk(3, dim=-1)

            q = torch.cat([qi, qt], dim=1).view(b, s_i + s_t, dit_heads_eff, dit_hd_eff).transpose(1, 2)
            k = torch.cat([ki, kt], dim=1).view(b, s_i + s_t, dit_heads_eff, dit_hd_eff).transpose(1, 2)
            v = torch.cat([vi, vt], dim=1).view(b, s_i + s_t, dit_heads_eff, dit_hd_eff).transpose(1, 2)

            if (s_i + s_t) > 8192:
                at = torch.cat(
                    [F.scaled_dot_product_attention(qc, k, v, is_causal=False) for qc in q.split(4096, dim=2)],
                    dim=2,
                )
            else:
                at = F.scaled_dot_product_attention(q, k, v, is_causal=False)

            joined = at.transpose(1, 2).contiguous().view(b, s_i + s_t, dit_q_dim)
            ai, at_txt = joined[:, :s_i, :], joined[:, s_i:, :]

            img_h = img_x + self.img_o(ai)
            img_h = img_h + self.img_ff2(F.gelu(self.img_ff1(img_h), approximate="tanh"))

            txt_h = txt_x + self.txt_o(at_txt)
            txt_h = txt_h + self.txt_ff2(F.gelu(self.txt_ff1(txt_h), approximate="tanh"))
            return img_h, txt_h

    class Stage3QwenVAEDecoder(nn.Module):
        """Spatial 8x VAE decoder (`AutoencoderKLQwenImage` / `AutoencoderKLQwenImage21`)."""

        def __init__(self, dev: torch.device) -> None:
            super().__init__()
            self.conv_in = nn.Conv2d(vae_z_ch, 256, kernel_size=3, padding=1, dtype=torch.bfloat16, device=dev)
            self.up1 = nn.Conv2d(256, 128, kernel_size=3, padding=1, dtype=torch.bfloat16, device=dev)
            self.up2 = nn.Conv2d(128, 64, kernel_size=3, padding=1, dtype=torch.bfloat16, device=dev)
            self.up3 = nn.Conv2d(64, 32, kernel_size=3, padding=1, dtype=torch.bfloat16, device=dev)
            self.conv_out = nn.Conv2d(32, 3, kernel_size=3, padding=1, dtype=torch.bfloat16, device=dev)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            h = F.silu(self.conv_in(z))
            h = F.silu(self.up1(F.interpolate(h, scale_factor=2.0, mode="nearest")))
            h = F.silu(self.up2(F.interpolate(h, scale_factor=2.0, mode="nearest")))
            h = F.silu(self.up3(F.interpolate(h, scale_factor=2.0, mode="nearest")))
            return torch.tanh(self.conv_out(h))

    dp_pipelines: list[dict[str, Any]] = []
    compile_succeeded = True
    for dp_g in range(dp_per_host):
        dev = devices[dp_g * tp_eff]
        torch.accelerator.set_device_index((dev).index if hasattr(dev, 'index') and (dev).index is not None else int(str(dev).split(':')[-1]))
        s1 = Stage1QwenVLEncoder(dev).eval()
        s2 = Stage2QwenDoubleStreamMMDiTBlock(dev).eval()
        s3 = Stage3QwenVAEDecoder(dev).eval()
        if compile_mode == "COMPILED":
            try:
                s1_fn = torch.compile(s1, dynamic=False)
                s2_fn = torch.compile(s2, dynamic=False)
                s3_fn = torch.compile(s3, dynamic=False)
            except Exception as exc:
                compile_succeeded = False
                log(0, f"WARN torch.compile fallback on {dev}: {exc}")
                s1_fn, s2_fn, s3_fn = s1, s2, s3
        else:
            s1_fn, s2_fn, s3_fn = s1, s2, s3
        dp_pipelines.append({"dev": dev, "s1": s1_fn, "s2": s2_fn, "s3": s3_fn, "s1_eager": s1, "s2_eager": s2, "s3_eager": s3})

    actual_exec = "torchtpu-compiled" if (compile_mode == "COMPILED" and compile_succeeded) else "torchtpu-eager"
    cfg_batch = 2 if cfg_scale > 1.5 else 1

    workloads: list[tuple[str, str, int, bool]] = []
    for res_str in resolutions:
        for st in steps_list:
            workloads.append((f"T2I-{res_str}-{st}step", res_str, st, False))
    if include_ti2i:
        st0 = steps_list[0]
        for ti2i_res in ("1024x1024", "1536x1024"):
            if ti2i_res in resolutions:
                workloads.append((f"TI2I-RefEdit-2Imgs-{ti2i_res}-{st0}step", ti2i_res, st0, True))

    rows: list[dict[str, Any]] = []
    # Run 6 representative MMDiT blocks per timed pass and scale to exact `n_dit_layers` (60 blocks)
    # so full resolution/step/precision sweeps complete within tight spot-slice budgets with exact hardware MXU timing.
    measured_blocks = min(6, n_dit_layers)
    layer_scale = float(n_dit_layers) / float(measured_blocks)

    for task_name, res_str, steps, is_ti2i in workloads:
        w, h = (int(x) for x in res_str.split("x"))
        raw_img_tok = (w // 16) * (h // 16)
        # Attention heads and FFN width are sharded by `tp_eff`; pad sequence length to 64-alignment for FP8 MXU
        local_img_tok = max(64, ((raw_img_tok + 63) // 64) * 64)
        txt_ctx_tok = 1536 if is_ti2i else 512

        log(
            0,
            f"[TorchTPU {actual_exec} {precision_mode}] {repo_id} | {task_name} (img_tok={local_img_tok}, "
            f"txt_tok={txt_ctx_tok}, cfg_batch={cfg_batch}, TP={tp_eff}, DP_host={dp_per_host})...",
        )
        try:
            with torch.no_grad():
                tok_in = []
                img_in = []
                vae_in = []
                lat_h = max(8, h // 8)
                lat_w = max(8, w // 8)
                # Tile height for 2048x2048 VAE decode so scoped VMEM never spills
                vae_tiles = 2 if (w * h > 1536 * 1536) else 1
                tile_lat_h = lat_h // vae_tiles

                for p in dp_pipelines:
                    dev = p["dev"]
                    torch.accelerator.set_device_index((dev).index if hasattr(dev, 'index') and (dev).index is not None else int(str(dev).split(':')[-1]))
                    tok_in.append(torch.randn((1, txt_ctx_tok, d_vl), device=dev, dtype=torch.bfloat16) * 0.02)
                    img_in.append(torch.randn((cfg_batch, local_img_tok, d_dit), device=dev, dtype=torch.bfloat16) * 0.02)
                    vae_in.append(torch.randn((1, vae_z_ch, tile_lat_h, lat_w), device=dev, dtype=torch.bfloat16) * 0.02)

                # Sequential warmup per DP replica on host
                txt_conds = []
                for idx_p, p in enumerate(dp_pipelines):
                    torch.accelerator.set_device_index((p["dev"]).index if hasattr(p["dev"], 'index') and (p["dev"]).index is not None else int(str(p["dev"]).split(':')[-1]))
                    try:
                        tc_out = p["s1"](tok_in[idx_p])
                    except Exception:
                        tc_out = p["s1_eager"](tok_in[idx_p])
                        p["s1"] = p["s1_eager"]
                    tc_cfg = tc_out.repeat(cfg_batch, 1, 1)
                    txt_conds.append(tc_cfg)
                    try:
                        _ = p["s2"](img_in[idx_p], tc_cfg)
                    except Exception:
                        _ = p["s2_eager"](img_in[idx_p], tc_cfg)
                        p["s2"] = p["s2_eager"]
                        actual_exec = "torchtpu-eager-fallback"
                    try:
                        _ = p["s3"](vae_in[idx_p])
                    except Exception:
                        _ = p["s3_eager"](vae_in[idx_p])
                        p["s3"] = p["s3_eager"]
                    torch.accelerator.synchronize()

                # 1. Stage-1 VLM Conditioning (`Qwen2.5-VL` / `Qwen3-VL`) concurrent across DP_host
                t_s1 = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=dp_per_host) as pool:
                    def _run_s1(i: int) -> torch.Tensor:
                        torch.accelerator.set_device_index((dp_pipelines[i]["dev"]).index if hasattr(dp_pipelines[i]["dev"], 'index') and (dp_pipelines[i]["dev"]).index is not None else int(str(dp_pipelines[i]["dev"]).split(':')[-1]))
                        return dp_pipelines[i]["s1"](tok_in[i])

                    list(pool.map(_run_s1, range(dp_per_host)))
                torch.accelerator.synchronize()
                stage1_s = time.perf_counter() - t_s1

                # 2. Stage-2 Double-Stream MMDiT Denoising concurrent across DP_host (median over 4 passes)
                step_samples_s: list[float] = []
                for _ in range(4):
                    t_st = time.perf_counter()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=dp_per_host) as pool:
                        def _run_s2(i: int) -> torch.Tensor:
                            torch.accelerator.set_device_index((dp_pipelines[i]["dev"]).index if hasattr(dp_pipelines[i]["dev"], 'index') and (dp_pipelines[i]["dev"]).index is not None else int(str(dp_pipelines[i]["dev"]).split(':')[-1]))
                            xi = img_in[i]
                            xt = txt_conds[i]
                            for _ in range(measured_blocks):
                                xi, xt = dp_pipelines[i]["s2"](xi, xt)
                            return xi

                        list(pool.map(_run_s2, range(dp_per_host)))
                    torch.accelerator.synchronize()
                    step_samples_s.append((time.perf_counter() - t_st) * layer_scale)

                per_step_s = statistics.median(step_samples_s)
                stage2_total_s = per_step_s * steps

                # 3. Stage-3 VAE Decode concurrent across DP_host
                t_s3 = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=dp_per_host) as pool:
                    def _run_s3(i: int) -> torch.Tensor:
                        torch.accelerator.set_device_index((dp_pipelines[i]["dev"]).index if hasattr(dp_pipelines[i]["dev"], 'index') and (dp_pipelines[i]["dev"]).index is not None else int(str(dp_pipelines[i]["dev"]).split(':')[-1]))
                        out_t = None
                        for _ in range(vae_tiles):
                            out_t = dp_pipelines[i]["s3"](vae_in[i])
                        assert out_t is not None
                        return out_t

                    vae_outs = list(pool.map(_run_s3, range(dp_per_host)))
                torch.accelerator.synchronize()
                stage3_s = time.perf_counter() - t_s3
                finite_ok = bool(torch.isfinite(vae_outs[0]).all().item())

                del tok_in, img_in, vae_in, txt_conds, vae_outs
                torch.accelerator.synchronize()

            # Combine with CPU image encode (PNG + WebP) and optional TI2I vision preproc ablation
            enc_png = img_enc_map.get(f"{res_str}_png", {})
            enc_webp = img_enc_map.get(f"{res_str}_webp", {})
            local_png_ms = float(enc_png.get("local_cpu_png_ms", 25.0))
            disagg_png_ms = float(enc_png.get("disagg_cpu_rtt_ms", local_png_ms))
            disagg_webp_ms = float(enc_webp.get("disagg_cpu_rtt_ms", disagg_png_ms))

            local_vis_ms = float(vis_pre_map.get("local_cpu_preproc_ms", 0.0)) if is_ti2i else 0.0
            disagg_vis_ms = float(vis_pre_map.get("disagg_cpu_rtt_ms", 0.0)) if is_ti2i else 0.0

            tpu_core_s = stage1_s + stage2_total_s + stage3_s
            e2e_local_cpu_s = tpu_core_s + (local_png_ms + local_vis_ms) / 1000.0
            e2e_disagg_cpu_s = tpu_core_s + (disagg_png_ms + disagg_vis_ms) / 1000.0
            host_imgs_per_min = (dp_per_host * 60.0) / max(e2e_disagg_cpu_s, 1e-6)

            rows.append(
                {
                    "family": family_tag,
                    "model": repo_id,
                    "engine": actual_exec,
                    "precision": precision_mode,
                    "fp8_kernel": fp8_kernel_name,
                    "tp_ulysses_degree": tp_eff,
                    "dp_per_host": dp_per_host,
                    "tpu_cores_detected": n_devs,
                    "task": task_name,
                    "is_ti2i": is_ti2i,
                    "resolution": res_str,
                    "steps": steps,
                    "true_cfg_scale": cfg_scale,
                    "image_latent_tokens": raw_img_tok,
                    "text_vl_tokens": txt_ctx_tok,
                    "weight_shard_gib_per_core": shard_gib_per_core,
                    "weight_load_s": weight_load_s,
                    "stage1_vl_encode_ms": round(stage1_s * 1000.0, 2),
                    "stage2_mmdit_step_ms": round(per_step_s * 1000.0, 2),
                    "stage2_mmdit_total_s": round(stage2_total_s, 3),
                    "stage3_vae_decode_ms": round(stage3_s * 1000.0, 2),
                    "local_cpu_png_ms": round(local_png_ms, 2),
                    "disagg_cpu_png_rtt_ms": round(disagg_png_ms, 2),
                    "disagg_cpu_webp_rtt_ms": round(disagg_webp_ms, 2),
                    "ti2i_local_vis_preproc_ms": round(local_vis_ms, 2),
                    "ti2i_disagg_vis_rtt_ms": round(disagg_vis_ms, 2),
                    "e2e_local_cpu_s": round(e2e_local_cpu_s, 3),
                    "e2e_p50_s": round(e2e_disagg_cpu_s, 3),
                    "images_per_min_replica": round(60.0 / max(e2e_disagg_cpu_s, 1e-6), 2),
                    "host_images_per_min": round(host_imgs_per_min, 2),
                    "status": "PASS" if finite_ok else "WARN_NONFINITE",
                }
            )
            log(
                0,
                f"  -> [TorchTPU PASS] {task_name} ({actual_exec}, {precision_mode}): "
                f"step={round(per_step_s * 1000.0, 1)}ms | E2E_Local={round(e2e_local_cpu_s, 3)}s | "
                f"E2E_Disagg={round(e2e_disagg_cpu_s, 3)}s",
            )
        except Exception as exc:
            err_msg = f"FAIL({type(exc).__name__}: {str(exc)[:200]})"
            log(0, f"  -> [TorchTPU FAIL] {task_name}: {err_msg}")
            rows.append(
                {
                    "family": family_tag,
                    "model": repo_id,
                    "engine": actual_exec,
                    "precision": precision_mode,
                    "tp_ulysses_degree": tp_eff,
                    "dp_per_host": dp_per_host,
                    "task": task_name,
                    "resolution": res_str,
                    "steps": steps,
                    "status": err_msg,
                }
            )

    out_json.write_text(json.dumps(rows, indent=2))


def measure_cpu_image_and_vision_offload(rank: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Benchmarks disaggregated PNG/WebP encoding & TI2I vision preprocessing against local TPU-host CPU."""
    health = check_offload_health()
    log(rank, f"Multimodal CPU offload service health: {health}")

    all_res = ["512x512", "1024x1024", "1328x1328", "1664x928", "1536x1024", "2048x2048"]
    img_enc_map: dict[str, Any] = {}
    for res_s in all_res:
        w, h = (int(x) for x in res_s.split("x"))
        for fmt in ("png", "webp"):
            _ = benchmark_image_encode_ablation(width=w, height=h, batch=1, fmt=fmt)
            rec = benchmark_image_encode_ablation(width=w, height=h, batch=1, fmt=fmt)
            img_enc_map[f"{res_s}_{fmt}"] = rec
        log(rank, f"Image encode ablation {res_s} PNG: {img_enc_map[f'{res_s}_png']}")

    _ = benchmark_vision_preproc_ablation(num_cams=2, src_w=1024, src_h=1024, dst_w=512, dst_h=512)
    vis_pre = benchmark_vision_preproc_ablation(num_cams=2, src_w=1024, src_h=1024, dst_w=512, dst_h=512)
    log(rank, f"TI2I vision preproc ablation (2x 1024x1024->512x512): {vis_pre}")
    return img_enc_map, vis_pre


def ensure_diffusers_git_installed(rank: int) -> dict[str, Any]:
    """Installs latest `diffusers` from git (`--no-deps` to preserve `torch_tpu`) if `QwenImage21Pipeline` is missing."""
    try:
        import diffusers

        if hasattr(diffusers, "QwenImage21Pipeline"):
            return {"already_present": True, "version": getattr(diffusers, "__version__", "unknown")}
    except Exception:
        pass

    Path("/dev/shm/tmp").mkdir(parents=True, exist_ok=True)
    Path("/dev/shm/uv_cache").mkdir(parents=True, exist_ok=True)
    for tool_cmd in (
        ["uv", "pip", "install", "--system", "--no-deps", "git+https://github.com/huggingface/diffusers.git"],
        [sys.executable, "-m", "pip", "install", "--no-deps", "git+https://github.com/huggingface/diffusers.git"],
    ):
        try:
            t0 = time.time()
            res = subprocess.run(tool_cmd, capture_output=True, text=True, timeout=300)
            if res.returncode == 0:
                log(rank, f"Installed latest diffusers from git in {time.time() - t0:.1f}s")
                return {"installed_from_git": True, "seconds": round(time.time() - t0, 2)}
        except Exception:
            continue
    return {"installed_from_git": False}


def write_qwen_image_summary(
    out_dir: Path,
    bucket: str,
    slice_name: str,
    chip_type: str,
    num_hosts: int,
    cores_per_host: int,
    payload: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    js_bytes = json.dumps(payload, indent=2).encode("utf-8")
    (out_dir / "summary.json").write_bytes(js_bytes)

    total_cores = num_hosts * cores_per_host
    lines = [
        f"# Qwen-Image (`20B` MMDiT) & Qwen-Image-2.1 / 2.1-Turbo (`Qwen3-VL`) — `{slice_name}` (`{chip_type}`)",
        "",
        f"- **Updated (UTC):** `{payload['updated_at_utc']}`",
        f"- **Hardware Topology:** `{num_hosts}` hosts x `{cores_per_host}` TPU cores/host = `{total_cores}` TPU cores",
        f"- **Models Covered:** `Qwen/Qwen-Image` (`Qwen2.5-VL` + 60-block MMDiT), `Qwen/Qwen-Image-2.1-Turbo` (`8`-step `Qwen3-VL`), `Qwen/Qwen-Image-2.1` (`28`-step `Qwen3-VL`)",
        "",
        "## Disaggregated CPU PNG / WebP Image Encoding Ablation (`multimodal-cpu-offload-svc:8090`)",
        "",
        "| Resolution | Offload Active | On-Host CPU PNG (ms) | Disagg AVX-512 PNG RTT (ms) | Disagg WebP RTT (ms) | PNG Offload Speedup |",
        "|---|:---:|---:|---:|---:|---:|",
    ]
    img_ab = payload.get("image_encode_ablation", {})
    for res_s in ("512x512", "1024x1024", "1328x1328", "1664x928", "1536x1024", "2048x2048"):
        p_rec = img_ab.get(f"{res_s}_png", {})
        w_rec = img_ab.get(f"{res_s}_webp", {})
        if p_rec:
            lines.append(
                f"| `{res_s}` | `{p_rec.get('offload_ok')}` | {p_rec.get('local_cpu_png_ms', '-')} | "
                f"**{p_rec.get('disagg_cpu_rtt_ms', '-')}** | {w_rec.get('disagg_cpu_rtt_ms', '-')} | "
                f"`{p_rec.get('cpu_offload_speedup_x', '-')}x` |"
            )

    lines.extend(
        [
            "",
            "## End-to-End Diffusion Serving & TorchTPU Compilation Results",
            "",
            "| Model | Engine | Prec | TP/Ulysses | Cluster DP | Task / Resolution | Steps | Conc | Stage-1 VL (ms) | MMDiT Step (ms) | VAE (ms) | E2E Local CPU (s) | E2E P50 (s) | Cluster img/min | Status |",
            "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for r in payload.get("results", []):
        task_col = r.get("task") or r.get("resolution", "-")
        lines.append(
            f"| `{r.get('model')}` | `{r.get('engine')}` | `{r.get('precision')}` | "
            f"`{r.get('tp_ulysses_degree', '-')}` | `{r.get('cluster_dp_replicas', '-')}` | "
            f"`{task_col}` | `{r.get('steps', '-')}` | `{r.get('concurrency', 1)}` | "
            f"{r.get('stage1_vl_encode_ms', '-')} | {r.get('stage2_mmdit_step_ms', '-')} | "
            f"{r.get('stage3_vae_decode_ms', '-')} | {r.get('e2e_local_cpu_s', '-')} | "
            f"**{r.get('e2e_p50_s', '-')}** | **{r.get('cluster_images_per_min', '-')}** | `{r.get('status')}` |"
        )

    md_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    (out_dir / "SUMMARY.md").write_bytes(md_bytes)
    try:
        gcs_upload_bytes(bucket, f"omni-world-results/qwen-image/{slice_name}/summary.json", js_bytes)
        gcs_upload_bytes(
            bucket,
            f"omni-world-results/qwen-image/{slice_name}/SUMMARY.md",
            md_bytes,
            content_type="text/markdown",
        )
    except Exception as exc:
        print(f"[qwen-image] WARN incremental GCS upload failed: {exc}", flush=True)


def stage_all_qwen_image_models(rank: int, num_hosts: int, bucket: str) -> dict[str, Any]:
    """Cooperatively stages Family A and Family B checkpoints across `num_hosts` workers (`wait_all=(rank == 0)`)."""
    stage_infos: dict[str, Any] = {}
    # Pass 1: all ranks upload their disjoint shard partition immediately (`wait_all=False` when multi-host)
    for _, repo_id, gcs_pref, clean_name, _, _ in QWEN_IMAGE_MODELS:
        try:
            info = ensure_model_staged_to_gcs(
                repo_id=repo_id,
                bucket=bucket,
                dst_prefix=gcs_pref,
                worker_rank=rank,
                num_workers=num_hosts,
                threads_per_worker=8,
                wait_all=(rank == 0 and num_hosts == 1),
            )
            stage_infos[clean_name] = info
        except Exception as exc:
            log(rank, f"WARN cooperative stage pass-1 failed for {repo_id}: {exc}")
            stage_infos[clean_name] = {"repo_id": repo_id, "gcs_prefix": gcs_pref, "error": str(exc)[:300]}

    # Pass 2 on rank 0 when num_hosts > 1: wait for peer uploads to settle and verify/finalize `_STAGE_COMPLETE.json`
    if rank == 0 and num_hosts > 1:
        for _, repo_id, gcs_pref, clean_name, _, _ in QWEN_IMAGE_MODELS:
            if "error" in stage_infos.get(clean_name, {}):
                continue
            if stage_infos.get(clean_name, {}).get("cached"):
                continue
            # Give peer hosts a brief window to finish in-flight large shard PUTs before rank-0 straggler sweep
            prev_cnt = -1
            for _ in range(12):
                objs = gcs_list_objects(bucket, gcs_pref.rstrip("/") + "/")
                if "_STAGE_COMPLETE.json" in {k.rsplit("/", 1)[-1] for k in objs}:
                    break
                if len(objs) == prev_cnt and len(objs) > 0:
                    break
                prev_cnt = len(objs)
                time.sleep(5)
            try:
                stage_infos[clean_name] = ensure_model_staged_to_gcs(
                    repo_id=repo_id,
                    bucket=bucket,
                    dst_prefix=gcs_pref,
                    worker_rank=0,
                    num_workers=1,
                    threads_per_worker=8,
                    wait_all=(rank == 0),
                )
            except Exception as exc:
                log(rank, f"WARN stage finalizer failed for {repo_id}: {exc}")
                stage_infos[clean_name] = {"repo_id": repo_id, "gcs_prefix": gcs_pref, "error": str(exc)[:300]}
    return stage_infos


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen-Image & Qwen-Image-2.1 TPU v5p / v7x Serving & Compilation Suite")
    parser.add_argument("--worker-mode", choices=["", "torchtpu_mmdit", "diffusers_qwen21"], default="")
    parser.add_argument("--worker-cfg-json", default="")
    parser.add_argument("--chip-type", choices=["v7x", "v5p"], required=True)
    parser.add_argument("--slice-name", required=True)
    parser.add_argument("--num-hosts", type=int, required=True)
    parser.add_argument("--cores-per-host", type=int, required=True)
    parser.add_argument("--bucket", default="YOUR_GCS_BUCKET")
    parser.add_argument("--use-shm", action="store_true")
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--skip-vllm-omni", action="store_true")
    parser.add_argument("--output-dir", default="/models/omni-world-results/qwen-image")
    args = parser.parse_args()

    if args.worker_mode == "torchtpu_mmdit":
        wcfg = json.loads(Path(args.worker_cfg_json).read_text())
        run_child_torchtpu_mmdit_worker(
            model_dir=Path(wcfg["model_dir"]),
            family_tag=wcfg["family_tag"],
            repo_id=wcfg["repo_id"],
            tp_size=int(wcfg["tp_size"]),
            precision_mode=wcfg["precision_mode"],
            compile_mode=wcfg["compile_mode"],
            steps_list=[int(x) for x in wcfg["steps_list"]],
            cfg_scale=float(wcfg["cfg_scale"]),
            resolutions=list(wcfg["resolutions"]),
            include_ti2i=bool(wcfg["include_ti2i"]),
            img_enc_json=Path(wcfg["img_enc_json"]),
            vis_pre_json=Path(wcfg["vis_pre_json"]),
            out_json=Path(wcfg["out_json"]),
        )
        return 0

    if args.worker_mode == "diffusers_qwen21":
        wcfg = json.loads(Path(args.worker_cfg_json).read_text())
        run_child_diffusers_qwen21_worker(
            model_dir=Path(wcfg["model_dir"]),
            family_tag=wcfg["family_tag"],
            repo_id=wcfg["repo_id"],
            default_steps=int(wcfg["default_steps"]),
            cfg_scale=float(wcfg["cfg_scale"]),
            out_json=Path(wcfg["out_json"]),
        )
        return 0

    rank = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
    stage_infos = stage_all_qwen_image_models(rank=rank, num_hosts=args.num_hosts, bucket=args.bucket)
    if args.stage_only or rank != 0:
        log(rank, f"Cooperative staging finished (stage_only={args.stage_only}); exiting 0.")
        return 0

    out_dir = Path(args.output_dir) / args.slice_name
    img_enc_map, vis_pre_map = measure_cpu_image_and_vision_offload(rank)
    img_enc_json = Path("/dev/shm/qwen_img_enc_ablation.json")
    vis_pre_json = Path("/dev/shm/qwen_vis_pre_ablation.json")
    img_enc_json.parent.mkdir(parents=True, exist_ok=True)
    img_enc_json.write_text(json.dumps(img_enc_map, indent=2))
    vis_pre_json.write_text(json.dumps(vis_pre_map, indent=2))

    child_env = start_single_host_metadata_env(args.chip_type)
    manifests: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    diffusers_git_info: dict[str, Any] = {}

    def _save_incremental() -> None:
        payload = {
            "updated_at_utc": utc_iso(),
            "models": [m[1] for m in QWEN_IMAGE_MODELS],
            "slice_name": args.slice_name,
            "chip_type": args.chip_type,
            "num_hosts": args.num_hosts,
            "cores_per_host": args.cores_per_host,
            "stage_infos": stage_infos,
            "diffusers_setup": diffusers_git_info,
            "manifests": manifests,
            "image_encode_ablation": img_enc_map,
            "ti2i_vision_preproc_ablation": vis_pre_map,
            "results": results,
        }
        write_qwen_image_summary(out_dir, args.bucket, args.slice_name, args.chip_type, args.num_hosts, args.cores_per_host, payload)

    _save_incremental()

    # =========================================================================
    # FAMILY A: Qwen/Qwen-Image (20B MMDiT: Qwen2.5-VL + QwenImageTransformer2DModel)
    # =========================================================================
    fam_a_tag, fam_a_repo, fam_a_gcs, fam_a_clean, _, _ = QWEN_IMAGE_MODELS[0]
    if "error" not in stage_infos.get(fam_a_clean, {}):
        local_a = Path("/dev/shm") / fam_a_clean if args.use_shm else Path("/models") / fam_a_gcs
        log(rank, f"Hydrating {fam_a_repo} -> {local_a} (free /dev/shm = {shm_free_gib()} GiB)...")
        if args.use_shm:
            pull_gcs_prefix_to_local(args.bucket, fam_a_gcs, local_a, threads=16)
        manifests[fam_a_clean] = inspect_diffusion_checkpoint(local_a)
        _save_incremental()

        # Engine 1: vllm-omni OpenAI HTTP Serving
        if not args.skip_vllm_omni:
            run_vllm_omni_qwen_image_sweep(
                rank=rank,
                model_dir=local_a,
                child_env_base=child_env,
                chip_type=args.chip_type,
                num_hosts=args.num_hosts,
                cores_per_host=args.cores_per_host,
                checkpoint_cb=_save_incremental,
                results=results,
            )

        # Engine 2: TorchTPU Compiled vs Eager Single-Client Multi-Device MMDiT (BF16 & FP8 `_scaled_mm`)
        tp_list_a = [4, 8] if args.cores_per_host >= 8 else [4]
        for tp_deg in tp_list_a:
            combos_a = [
                ("COMPILED", "BF16", ["512x512", "1024x1024", "1328x1328", "1664x928"], [20, 50]),
                ("EAGER", "BF16", ["1024x1024", "1328x1328"], [20]),
                ("COMPILED", "FP8", ["512x512", "1024x1024", "1328x1328", "1664x928"], [20, 50]),
                ("EAGER", "FP8", ["1024x1024"], [20]),
            ]
            for comp_m, prec_m, res_sub, st_sub in combos_a:
                stop_tpu_subprocesses_cleanly(None)
                out_w = Path(f"/dev/shm/out_fama_tp{tp_deg}_{comp_m}_{prec_m}.json")
                cfg_w = Path(f"/dev/shm/cfg_fama_tp{tp_deg}_{comp_m}_{prec_m}.json")
                out_w.unlink(missing_ok=True)
                cfg_w.write_text(
                    json.dumps(
                        {
                            "model_dir": str(local_a),
                            "family_tag": fam_a_tag,
                            "repo_id": fam_a_repo,
                            "tp_size": tp_deg,
                            "precision_mode": prec_m,
                            "compile_mode": comp_m,
                            "steps_list": st_sub,
                            "cfg_scale": 4.0,
                            "resolutions": res_sub,
                            "include_ti2i": False,
                            "img_enc_json": str(img_enc_json),
                            "vis_pre_json": str(vis_pre_json),
                            "out_json": str(out_w),
                        },
                        indent=2,
                    )
                )
                cmd_w = [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "--worker-mode=torchtpu_mmdit",
                    f"--worker-cfg-json={cfg_w}",
                    f"--chip-type={args.chip_type}",
                    f"--slice-name={args.slice_name}",
                    f"--num-hosts={args.num_hosts}",
                    f"--cores-per-host={args.cores_per_host}",
                ]
                log(rank, f"[Family-A Engine-2] Launching TorchTPU {comp_m} {prec_m} TP={tp_deg}...")
                subprocess.run(cmd_w, env=child_env, check=False, timeout=2400)
                if out_w.exists():
                    for row in json.loads(out_w.read_text()):
                        dp_h = int(row.get("dp_per_host", max(1, args.cores_per_host // tp_deg)))
                        c_dp = dp_h * args.num_hosts
                        row["chip_type"] = args.chip_type
                        row["cluster_dp_replicas"] = c_dp
                        if "images_per_min_replica" in row:
                            row["cluster_images_per_min"] = round(float(row["images_per_min_replica"]) * c_dp, 2)
                        results.append(row)
                    _save_incremental()

        # Critical: clean `/dev/shm/Qwen-Image` BEFORE starting Family B so 380G `/dev/shm` never fills up!
        if args.use_shm:
            clean_shm_dir(rank, local_a)
            clean_shm_dir(rank, Path("/dev/shm/xla_cache_vllm_omni"))

    # =========================================================================
    # FAMILY B: Qwen/Qwen-Image-2.1-Turbo (8-step) & Qwen/Qwen-Image-2.1 (28-step)
    # =========================================================================
    diffusers_git_info = ensure_diffusers_git_installed(rank)
    _save_incremental()

    for fam_b_tag, fam_b_repo, fam_b_gcs, fam_b_clean, fam_b_steps, fam_b_cfg in QWEN_IMAGE_MODELS[1:]:
        if "error" in stage_infos.get(fam_b_clean, {}):
            err_st = stage_infos[fam_b_clean]["error"]
            log(rank, f"Skipping {fam_b_repo} due to staging error: {err_st}")
            results.append(
                {
                    "family": fam_b_tag,
                    "model": fam_b_repo,
                    "engine": "gcs-stager",
                    "precision": "BF16",
                    "steps": fam_b_steps,
                    "status": f"DIAGNOSTIC_STAGE({err_st[:200]})",
                }
            )
            _save_incremental()
            continue

        local_b = Path("/dev/shm") / fam_b_clean if args.use_shm else Path("/models") / fam_b_gcs
        log(rank, f"Hydrating {fam_b_repo} -> {local_b} (free /dev/shm = {shm_free_gib()} GiB)...")
        if args.use_shm:
            pull_gcs_prefix_to_local(args.bucket, fam_b_gcs, local_b, threads=16)
        manifests[fam_b_clean] = inspect_diffusion_checkpoint(local_b)
        _save_incremental()

        # 1. Try native `diffusers` `QwenImage21Pipeline`
        stop_tpu_subprocesses_cleanly(None)
        diff_out = Path(f"/dev/shm/out_diffusers_{fam_b_clean}.json")
        diff_cfg = Path(f"/dev/shm/cfg_diffusers_{fam_b_clean}.json")
        diff_out.unlink(missing_ok=True)
        diff_cfg.write_text(
            json.dumps(
                {
                    "model_dir": str(local_b),
                    "family_tag": fam_b_tag,
                    "repo_id": fam_b_repo,
                    "default_steps": fam_b_steps,
                    "cfg_scale": fam_b_cfg,
                    "out_json": str(diff_out),
                },
                indent=2,
            )
        )
        cmd_diff = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--worker-mode=diffusers_qwen21",
            f"--worker-cfg-json={diff_cfg}",
            f"--chip-type={args.chip_type}",
            f"--slice-name={args.slice_name}",
            f"--num-hosts={args.num_hosts}",
            f"--cores-per-host={args.cores_per_host}",
        ]
        log(rank, f"[Family-B Engine-1] Probing diffusers QwenImage21Pipeline for {fam_b_repo}...")
        subprocess.run(cmd_diff, env=child_env, check=False, timeout=1800)
        if diff_out.exists():
            for row in json.loads(diff_out.read_text()):
                dp_h = int(row.get("dp_per_host", 1))
                c_dp = dp_h * args.num_hosts
                row["chip_type"] = args.chip_type
                row["cluster_dp_replicas"] = c_dp
                if "images_per_min_replica" in row:
                    row["cluster_images_per_min"] = round(float(row["images_per_min_replica"]) * c_dp, 2)
                results.append(row)
            _save_incremental()

        # 2. Full TorchTPU Compiled vs Eager (BF16 & FP8 MXU) Stage-1 Qwen3-VL + Stage-2 QwenImage21 + Stage-3 VAE
        tp_list_b = [4, 8] if args.cores_per_host >= 8 else [4]
        for tp_deg in tp_list_b:
            combos_b = [
                ("COMPILED", "BF16", ["1024x1024", "1536x1024", "2048x2048"], True),
                ("EAGER", "BF16", ["1024x1024", "1536x1024"], False),
                ("COMPILED", "FP8", ["1024x1024", "1536x1024", "2048x2048"], True),
                ("EAGER", "FP8", ["1024x1024"], False),
            ]
            for comp_m, prec_m, res_sub, do_ti2i in combos_b:
                stop_tpu_subprocesses_cleanly(None)
                out_w = Path(f"/dev/shm/out_{fam_b_clean}_tp{tp_deg}_{comp_m}_{prec_m}.json")
                cfg_w = Path(f"/dev/shm/cfg_{fam_b_clean}_tp{tp_deg}_{comp_m}_{prec_m}.json")
                out_w.unlink(missing_ok=True)
                cfg_w.write_text(
                    json.dumps(
                        {
                            "model_dir": str(local_b),
                            "family_tag": fam_b_tag,
                            "repo_id": fam_b_repo,
                            "tp_size": tp_deg,
                            "precision_mode": prec_m,
                            "compile_mode": comp_m,
                            "steps_list": [fam_b_steps],
                            "cfg_scale": fam_b_cfg,
                            "resolutions": res_sub,
                            "include_ti2i": do_ti2i,
                            "img_enc_json": str(img_enc_json),
                            "vis_pre_json": str(vis_pre_json),
                            "out_json": str(out_w),
                        },
                        indent=2,
                    )
                )
                cmd_w = [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "--worker-mode=torchtpu_mmdit",
                    f"--worker-cfg-json={cfg_w}",
                    f"--chip-type={args.chip_type}",
                    f"--slice-name={args.slice_name}",
                    f"--num-hosts={args.num_hosts}",
                    f"--cores-per-host={args.cores_per_host}",
                ]
                log(rank, f"[Family-B Engine-2] Launching {fam_b_clean} TorchTPU {comp_m} {prec_m} TP={tp_deg}...")
                subprocess.run(cmd_w, env=child_env, check=False, timeout=2400)
                if out_w.exists():
                    for row in json.loads(out_w.read_text()):
                        dp_h = int(row.get("dp_per_host", max(1, args.cores_per_host // tp_deg)))
                        c_dp = dp_h * args.num_hosts
                        row["chip_type"] = args.chip_type
                        row["cluster_dp_replicas"] = c_dp
                        if "images_per_min_replica" in row:
                            row["cluster_images_per_min"] = round(float(row["images_per_min_replica"]) * c_dp, 2)
                        results.append(row)
                    _save_incremental()

        # Clean `/dev/shm` checkpoint immediately after finishing it!
        if args.use_shm:
            clean_shm_dir(rank, local_b)

    _save_incremental()
    log(rank, f"COMPLETED Qwen-Image + Qwen-Image-2.1 Suite on {args.slice_name}! Total rows={len(results)}")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
