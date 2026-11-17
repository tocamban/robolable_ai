import gc
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from project_manager import atomic_write_json

try:
    import psutil
except Exception:
    psutil = None

VALID_DEVICE_PREFERENCES = {"auto", "cpu", "cuda"}
DEVICE_CONFIG_FILENAME = "device_config.json"


def resolve_weight_path(
    weights_dir: str,
    base_dir: str,
    filename: str,
    fallback_to_weights: bool = False,
) -> str | None:
    """Resolve an existing weight file path in weights/ or base_dir."""
    candidates = (os.path.join(weights_dir, filename), os.path.join(base_dir, filename))
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0] if fallback_to_weights else None


def _file_size_mb(path: str | None) -> float | None:
    if not path or not os.path.exists(path):
        return None
    return round(os.path.getsize(path) / (1024 * 1024), 1)


def get_cpu_display_name() -> str:
    """Return a readable CPU label with core count."""
    cores = os.cpu_count() or 1
    proc = (platform.processor() or "").strip()
    if proc and len(proc) <= 48:
        return f"CPU ({cores} luồng)"
    return f"CPU ({cores} luồng - Intel/AMD)"


def load_device_preference(base_dir: str) -> str:
    """Load persisted device preference ('auto' | 'cpu' | 'cuda')."""
    cfg_path = os.path.join(base_dir, DEVICE_CONFIG_FILENAME)
    if not os.path.exists(cfg_path):
        return "auto"
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pref = str((data or {}).get("preferred_device", "auto")).strip().lower()
        return pref if pref in VALID_DEVICE_PREFERENCES else "auto"
    except Exception:
        return "auto"


def save_device_preference(base_dir: str, preference: str) -> None:
    """Atomically persist device preference to device_config.json."""
    pref = str(preference or "auto").strip().lower()
    if pref not in VALID_DEVICE_PREFERENCES:
        raise ValueError(f"Thiết bị không hợp lệ: {preference}")
    atomic_write_json(os.path.join(base_dir, DEVICE_CONFIG_FILENAME), {"preferred_device": pref})


def resolve_runtime_device(preference: str) -> tuple[str, str, str | None]:
    """
    Resolve active PyTorch device ('cuda' or 'cpu'), display name, and optional fallback warning.
    """
    pref = str(preference or "auto").strip().lower()
    cuda_ok = bool(torch.cuda.is_available())
    if pref == "cpu":
        return "cpu", get_cpu_display_name(), None
    if pref == "cuda":
        if cuda_ok:
            return "cuda", torch.cuda.get_device_name(0), None
        return (
            "cpu",
            get_cpu_display_name(),
            "PyTorch hiện tại là bản CPU nên chưa thể kích hoạt CUDA; hệ thống tạm chạy CPU.",
        )
    # "auto"
    if cuda_ok:
        return "cuda", torch.cuda.get_device_name(0), None
    return "cpu", get_cpu_display_name(), None


def _query_nvidia_via_pynvml() -> dict | None:
    """Query physical NVIDIA GPU details using nvidia-ml-py (pynvml)."""
    try:
        import pynvml
    except Exception:
        return None

    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        count = pynvml.nvmlDeviceGetCount()
        if count <= 0:
            return None
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        raw_name = pynvml.nvmlDeviceGetName(handle)
        gpu_name = raw_name.decode("utf-8", errors="ignore") if isinstance(raw_name, bytes) else str(raw_name)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

        raw_driver = pynvml.nvmlSystemGetDriverVersion()
        driver_ver = raw_driver.decode("utf-8", errors="ignore") if isinstance(raw_driver, bytes) else str(raw_driver)

        cuda_drv_ver = None
        try:
            ver_int = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            major, minor = ver_int // 1000, (ver_int % 1000) // 10
            cuda_drv_ver = f"{major}.{minor}"
        except Exception:
            pass

        compute_cap = None
        try:
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            compute_cap = f"{major}.{minor}"
        except Exception:
            pass

        temp_c = None
        try:
            temp_c = int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
        except Exception:
            pass

        util_pct = None
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            util_pct = int(util.gpu)
        except Exception:
            pass

        return {
            "gpu_detected": True,
            "gpu_count": int(count),
            "gpu_name": gpu_name.strip(),
            "vram_total_mb": round(mem.total / (1024 * 1024), 1),
            "vram_free_mb": round(mem.free / (1024 * 1024), 1),
            "vram_used_mb": round(mem.used / (1024 * 1024), 1),
            "driver_version": driver_ver.strip(),
            "max_cuda_version": cuda_drv_ver,
            "compute_capability": compute_cap,
            "temperature_c": temp_c,
            "gpu_util_pct": util_pct,
        }
    except Exception:
        return None
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def _query_nvidia_via_smi() -> dict | None:
    """Fallback physical GPU query via nvidia-smi CLI."""
    smi_path = shutil.which("nvidia-smi")
    if not smi_path:
        return None
    try:
        res = subprocess.run(
            [
                smi_path,
                "--query-gpu=name,memory.total,memory.free,memory.used,driver_version,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode != 0 or not res.stdout.strip():
            return None
        first_line = res.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in first_line.split(",")]
        if len(parts) < 5:
            return None

        def _to_float(v: str) -> float | None:
            try:
                return round(float(v), 1)
            except Exception:
                return None

        return {
            "gpu_detected": True,
            "gpu_count": len(res.stdout.strip().splitlines()),
            "gpu_name": parts[0],
            "vram_total_mb": _to_float(parts[1]),
            "vram_free_mb": _to_float(parts[2]),
            "vram_used_mb": _to_float(parts[3]),
            "driver_version": parts[4],
            "max_cuda_version": None,
            "compute_capability": None,
            "temperature_c": int(float(parts[5])) if len(parts) > 5 and _to_float(parts[5]) is not None else None,
            "gpu_util_pct": int(float(parts[6])) if len(parts) > 6 and _to_float(parts[6]) is not None else None,
        }
    except Exception:
        return None


def detect_nvidia_gpu() -> dict:
    """Detect physical NVIDIA GPU telemetry regardless of whether PyTorch has CUDA."""
    info = _query_nvidia_via_pynvml() or _query_nvidia_via_smi()
    if info:
        return info
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_mb = round(props.total_memory / (1024 * 1024), 1)
        return {
            "gpu_detected": True,
            "gpu_count": int(torch.cuda.device_count()),
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_total_mb": total_mb,
            "vram_free_mb": None,
            "vram_used_mb": None,
            "driver_version": None,
            "max_cuda_version": torch.version.cuda,
            "compute_capability": f"{props.major}.{props.minor}",
            "temperature_c": None,
            "gpu_util_pct": None,
        }
    return {
        "gpu_detected": False,
        "gpu_count": 0,
        "gpu_name": None,
        "vram_total_mb": None,
        "vram_free_mb": None,
        "vram_used_mb": None,
        "driver_version": None,
        "max_cuda_version": None,
        "compute_capability": None,
        "temperature_c": None,
        "gpu_util_pct": None,
    }


def _select_cuda_wheel_tag(max_cuda_version: str | None) -> str:
    """Choose the best matching PyTorch CUDA wheel tag given Python & NVIDIA driver CUDA ceiling."""
    if sys.version_info >= (3, 14):
        return "cu126"
    if max_cuda_version:
        try:
            major, minor = [int(x) for x in max_cuda_version.split(".")[:2]]
            if (major, minor) < (12, 4):
                return "cu121"
            if (major, minor) < (12, 6):
                return "cu124"
        except Exception:
            pass
    return "cu126"


def get_hardware_info(
    active_device: str,
    active_device_name: str,
    preferred_device: str,
    fallback_note: str | None = None,
) -> dict:
    """Build complete runtime hardware & PyTorch telemetry payload for UI/API."""
    cuda_available = bool(torch.cuda.is_available())
    gpu_info = detect_nvidia_gpu()
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    try:
        import torchvision
        tv_ver = torchvision.__version__
    except Exception:
        tv_ver = None

    ram_total_gb = None
    ram_avail_gb = None
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            ram_total_gb = round(vm.total / (1024 ** 3), 1)
            ram_avail_gb = round(vm.available / (1024 ** 3), 1)
        except Exception:
            pass

    cuda_tag = _select_cuda_wheel_tag(gpu_info.get("max_cuda_version"))
    install_cmd = (
        f"venv\\Scripts\\pip.exe uninstall -y torch torchvision && "
        f"venv\\Scripts\\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/{cuda_tag}"
    )

    if cuda_available:
        if active_device == "cuda":
            switch_hint = f"Đang dùng GPU ({active_device_name}). Có thể chuyển qua lại với CPU bất kỳ lúc nào."
        else:
            switch_hint = f"Đang chạy CPU theo lựa chọn người dùng (GPU {gpu_info.get('gpu_name') or 'CUDA'} đã sẵn sàng để bật)."
    elif gpu_info["gpu_detected"]:
        switch_hint = (
            f"Phát hiện phần cứng {gpu_info['gpu_name']} ({int(round(gpu_info['vram_total_mb'] or 0))} MB VRAM), "
            f"nhưng PyTorch hiện tại ({torch.__version__}) là bản CPU (+cpu) vì Python {py_ver} chưa được cài wheel CUDA ({cuda_tag})."
        )
    else:
        switch_hint = "Không phát hiện GPU NVIDIA rời trên máy; hệ thống chạy trên CPU."

    return {
        "device": active_device,
        "active_device": active_device,
        "device_name": active_device_name,
        "preferred_device": preferred_device,
        "cuda_available": cuda_available,
        "can_switch_to_gpu": cuda_available,
        "fallback_note": fallback_note,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torchvision_version": tv_ver,
        "python_version": py_ver,
        "cpu_cores": os.cpu_count() or 1,
        "cpu_name": get_cpu_display_name(),
        "ram_total_gb": ram_total_gb,
        "ram_available_gb": ram_avail_gb,
        "recommended_model": "sam2_tiny",
        "recommended_cuda_tag": cuda_tag,
        "cuda_install_command": install_cmd,
        "switch_hint": switch_hint,
        **gpu_info,
    }


def _sync_if_cuda(device_str: str) -> None:
    if device_str == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _benchmark_tensor_device(device_str: str, warmup: int = 1, runs: int = 5) -> dict:
    """Measure PyTorch CNN + projection throughput on the target device."""
    model = None
    x = None
    proj = None
    try:
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((32, 32)),
        ).to(device_str).eval()
        proj = torch.randn((64 * 32 * 32, 128), device=device_str)
        x = torch.randn((1, 3, 512, 512), device=device_str)

        if device_str == "cuda":
            torch.cuda.reset_peak_memory_stats(0)

        with torch.no_grad():
            for _ in range(warmup):
                feat = model(x).flatten(1)
                _ = feat @ proj
            _sync_if_cuda(device_str)

            latencies_ms = []
            for _ in range(runs):
                _sync_if_cuda(device_str)
                t0 = time.perf_counter()
                feat = model(x).flatten(1)
                _ = feat @ proj
                _sync_if_cuda(device_str)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        peak_vram_mb = None
        if device_str == "cuda":
            peak_vram_mb = round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 1)

        return {
            "executed": True,
            "runs": runs,
            "avg_ms": round(sum(latencies_ms) / len(latencies_ms), 2),
            "min_ms": round(min(latencies_ms), 2),
            "peak_vram_mb": peak_vram_mb,
        }
    except Exception as e:
        return {
            "executed": False,
            "reason": str(e),
            "oom": "out of memory" in str(e).lower(),
        }
    finally:
        del model, x, proj
        gc.collect()
        if device_str == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _pick_benchmark_image(sample_image_path: str | None) -> tuple[str, bool]:
    """Return a valid RGB image path for benchmarking; synthesize one safely if needed."""
    if sample_image_path and os.path.isfile(sample_image_path):
        try:
            with Image.open(sample_image_path) as im:
                im.verify()
            return sample_image_path, False
        except Exception:
            pass

    arr = np.zeros((480, 640, 3), dtype=np.uint8)
    arr[:, :] = (34, 90, 42)
    yy, xx = np.ogrid[:480, :640]
    fruit_mask = ((xx - 320) ** 2) / (110 ** 2) + ((yy - 250) ** 2) / (130 ** 2) <= 1.0
    arr[fruit_mask] = (220, 48, 42)
    calyx_mask = ((xx - 320) ** 2) / (90 ** 2) + ((yy - 135) ** 2) / (30 ** 2) <= 1.0
    arr[calyx_mask] = (58, 165, 62)

    tmp = tempfile.NamedTemporaryFile(prefix="robolabel_bench_", suffix=".jpg", delete=False)
    tmp_path = tmp.name
    tmp.close()
    Image.fromarray(arr, mode="RGB").save(tmp_path, quality=92)
    return tmp_path, True


def _benchmark_sam_model(
    weight_path: str | None,
    img_path: str,
    device_str: str,
    runs: int = 2,
) -> dict:
    """Benchmark actual Ultralytics SAM inference latency and peak VRAM."""
    if not weight_path or not os.path.exists(weight_path):
        return {"executed": False, "reason": "Không tìm thấy file trọng số SAM."}

    from ultralytics import SAM

    model = None
    try:
        with Image.open(img_path) as im:
            w, h = im.size
        box = [w * 0.2, h * 0.2, w * 0.8, h * 0.8]

        load_t0 = time.perf_counter()
        model = SAM(weight_path)
        load_ms = (time.perf_counter() - load_t0) * 1000.0

        if device_str == "cuda":
            torch.cuda.reset_peak_memory_stats(0)

        # Warmup
        _ = model(img_path, bboxes=[box], device=device_str, verbose=False)
        _sync_if_cuda(device_str)

        latencies_ms = []
        for _ in range(runs):
            _sync_if_cuda(device_str)
            t0 = time.perf_counter()
            _ = model(img_path, bboxes=[box], device=device_str, verbose=False)
            _sync_if_cuda(device_str)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        peak_vram_mb = None
        if device_str == "cuda":
            peak_vram_mb = round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 1)

        return {
            "executed": True,
            "runs": runs,
            "load_ms": round(load_ms, 1),
            "avg_ms": round(sum(latencies_ms) / len(latencies_ms), 1),
            "min_ms": round(min(latencies_ms), 1),
            "peak_vram_mb": peak_vram_mb,
        }
    except Exception as e:
        return {
            "executed": False,
            "reason": str(e),
            "oom": "out of memory" in str(e).lower(),
        }
    finally:
        del model
        gc.collect()
        if device_str == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _benchmark_clip_model(
    weight_path: str | None,
    img_path: str,
    device_str: str,
    runs: int = 3,
) -> dict:
    """Benchmark actual CLIP ViT-B/32 zero-shot classification latency."""
    if not weight_path or not os.path.exists(weight_path):
        return {"executed": False, "reason": "Không tìm thấy weights/clip/ViT-B-32.pt."}

    model = None
    try:
        import clip

        model, preprocess = clip.load(weight_path, device=device_str)
        prompts = [
            "an orange yellowish almost ripe strawberry fruit",
            "green strawberry calyx leaves sepals",
            "a strawberry stem stalk",
            "a pale green unripe strawberry fruit",
            "a bright red ripe strawberry fruit",
        ]
        tokens = clip.tokenize(prompts).to(device_str)

        with Image.open(img_path) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            crop = rgb.crop((int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)))
        img_in = preprocess(crop).unsqueeze(0).to(device_str)

        if device_str == "cuda":
            torch.cuda.reset_peak_memory_stats(0)

        with torch.no_grad():
            text_feat = model.encode_text(tokens)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)
            img_feat = model.encode_image(img_in)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            _ = (100.0 * img_feat @ text_feat.T).softmax(dim=-1)
            _sync_if_cuda(device_str)

            latencies_ms = []
            for _ in range(runs):
                _sync_if_cuda(device_str)
                t0 = time.perf_counter()
                img_feat = model.encode_image(img_in)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                _ = (100.0 * img_feat @ text_feat.T).softmax(dim=-1)
                _sync_if_cuda(device_str)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        peak_vram_mb = None
        if device_str == "cuda":
            peak_vram_mb = round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 1)

        return {
            "executed": True,
            "runs": runs,
            "avg_ms": round(sum(latencies_ms) / len(latencies_ms), 2),
            "min_ms": round(min(latencies_ms), 2),
            "peak_vram_mb": peak_vram_mb,
        }
    except Exception as e:
        return {
            "executed": False,
            "reason": str(e),
            "oom": "out of memory" in str(e).lower(),
        }
    finally:
        del model
        gc.collect()
        if device_str == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate_gpu_suitability(
    hw_info: dict,
    cpu_bench: dict,
    gpu_bench: dict,
    weights_dir: str,
    base_dir: str,
) -> dict:
    """Assess physical GPU hardware and measured benchmark timings against RoboLabel models."""
    vram_total = hw_info.get("vram_total_mb") or 0.0
    vram_free = hw_info.get("vram_free_mb") or vram_total
    gpu_detected = bool(hw_info.get("gpu_detected"))
    cuda_available = bool(hw_info.get("cuda_available"))
    gpu_name = hw_info.get("gpu_name") or "Chưa phát hiện GPU NVIDIA"

    sam_tiny_path = resolve_weight_path(weights_dir, base_dir, "sam2.1_t.pt")
    sam_large_path = resolve_weight_path(weights_dir, base_dir, "sam2.1_l.pt")
    clip_path = resolve_weight_path(weights_dir, base_dir, os.path.join("clip", "ViT-B-32.pt"))

    def _model_vram_status(est_need_mb: float, measured_peak_mb: float | None = None, oom: bool = False):
        if oom:
            return "no_fit", "Báo lỗi thiếu VRAM (CUDA Out of Memory) khi chạy thử"
        if not gpu_detected:
            return "cpu_only", "Không có GPU NVIDIA; chạy trên CPU"
        effective_mb = max(est_need_mb, measured_peak_mb or 0.0)
        if vram_free >= effective_mb + 600:
            return "fit", "Chạy mượt, dư dả VRAM"
        if vram_total >= effective_mb:
            return "tight", "Vừa sát ngưỡng VRAM (nên đóng app đồ họa khác hoặc ưu tiên Tiny)"
        return "no_fit", "Thiếu VRAM, dễ tràn bộ nhớ GPU"

    specs = (
        ("sam_tiny", "sam2_tiny", "SAM 2.1 (Tiny) - Khuyên dùng", sam_tiny_path, 74.5, 1300.0),
        ("clip", "clip_vit_b32", "CLIP (ViT-B/32) - Tự nhận diện nhãn", clip_path, 337.6, 950.0),
        ("sam_large", "sam2_large", "SAM 2.1 (Large) - Chi tiết cao", sam_large_path, 428.4, 3600.0),
    )
    model_compatibility = []
    for bench_key, model_id, display_name, w_path, default_mb, est_vram_mb in specs:
        gpu_m = gpu_bench.get(bench_key) or {}
        cpu_m = cpu_bench.get(bench_key) or {}
        peak_vram = gpu_m.get("peak_vram_mb")
        status, note = _model_vram_status(est_vram_mb, peak_vram, bool(gpu_m.get("oom")))
        model_compatibility.append({
            "id": model_id,
            "name": display_name,
            "weight_mb": _file_size_mb(w_path) or default_mb,
            "est_vram_mb": int(est_vram_mb),
            "measured_peak_vram_mb": peak_vram,
            "cpu_ms": cpu_m.get("avg_ms"),
            "gpu_ms": gpu_m.get("avg_ms"),
            "status": status,
            "note": note,
        })

    tiny_status = model_compatibility[0]["status"]
    gpu_sam_tiny = gpu_bench.get("sam_tiny") or {}

    cpu_sam_ms = (cpu_bench.get("sam_tiny") or {}).get("avg_ms")
    gpu_sam_ms = gpu_sam_tiny.get("avg_ms")
    speedup_sam = (
        round(cpu_sam_ms / gpu_sam_ms, 2)
        if cpu_sam_ms and gpu_sam_ms and gpu_sam_ms > 0
        else None
    )

    cpu_tensor_ms = (cpu_bench.get("tensor") or {}).get("avg_ms")
    gpu_tensor_ms = (gpu_bench.get("tensor") or {}).get("avg_ms")
    speedup_tensor = (
        round(cpu_tensor_ms / gpu_tensor_ms, 2)
        if cpu_tensor_ms and gpu_tensor_ms and gpu_tensor_ms > 0
        else None
    )

    if cuda_available and gpu_bench.get("executed"):
        if tiny_status == "fit":
            overall_status = "excellent"
            badge_label = f"GPU Phù Hợp ({gpu_name})"
            speed_txt = f"nhanh hơn CPU {speedup_sam}x" if speedup_sam else "ổn định"
            summary = (
                f"{gpu_name} ({int(round(vram_total))} MB VRAM) hoạt động tốt với PyTorch CUDA: "
                f"SAM 2.1 Tiny mất {gpu_sam_ms} ms/ảnh ({speed_txt}, CPU {cpu_sam_ms} ms)."
            )
            recommendation = (
                "Nên bật chế độ GPU với SAM 2.1 (Tiny) để đánh nhãn tức thì. "
                + (
                    "Với 4GB VRAM, hạn chế dùng SAM 2.1 Large liên tục để tránh tràn VRAM."
                    if vram_total <= 4500
                    else "Dung lượng VRAM đủ cho cả SAM 2.1 Large."
                )
            )
        else:
            overall_status = "limited"
            badge_label = f"GPU Hạn Chế VRAM ({gpu_name})"
            summary = f"{gpu_name} có {int(round(vram_total))} MB VRAM, khá sát ngưỡng khi chạy SAM 2.1."
            recommendation = "Ưu tiên SAM 2.1 (Tiny); nếu gặp lỗi OOM hãy chuyển về CPU."
    elif gpu_detected and not cuda_available:
        overall_status = "needs_cuda_pytorch"
        badge_label = f"{gpu_name} Phù Hợp Phần Cứng (Chưa bật CUDA)"
        cpu_txt = f"{cpu_sam_ms} ms/ảnh" if cpu_sam_ms else "chậm hơn GPU"
        cc = hw_info.get("compute_capability") or "7.x"
        summary = (
            f"Phần cứng {gpu_name} ({int(round(vram_total))} MB VRAM, trống {int(round(vram_free))} MB, "
            f"Compute {cc}, Driver {hw_info.get('driver_version') or 'N/A'}) RẤT PHÙ HỢP cho SAM 2.1 Tiny và CLIP (tự nhận diện nhãn polygon). "
            f"Hiện app đang chạy trên CPU ({cpu_txt} cho SAM 2.1 Tiny) vì venv mới cài PyTorch {hw_info.get('torch_version')}."
        )
        recommendation = (
            "Cài lại torch/torchvision bản CUDA (ví dụ "
            f"{hw_info.get('recommended_cuda_tag')}) bằng lệnh bên dưới rồi khởi động lại server để bật nút chuyển GPU."
        )
    else:
        overall_status = "cpu_only"
        badge_label = "Đang dùng CPU (Không có GPU NVIDIA)"
        summary = f"Không phát hiện GPU NVIDIA; benchmark CPU cho SAM 2.1 Tiny đạt {cpu_sam_ms or 'N/A'} ms/ảnh."
        recommendation = "Tiếp tục dùng CPU với SAM 2.1 (Tiny) để giữ tốc độ phản hồi ổn định."

    return {
        "overall_status": overall_status,
        "badge_label": badge_label,
        "summary": summary,
        "recommendation": recommendation,
        "speedup_sam_tiny": speedup_sam,
        "speedup_tensor": speedup_tensor,
        "model_compatibility": model_compatibility,
    }


def run_device_benchmark(
    weights_dir: str,
    base_dir: str,
    hw_info: dict,
    sample_image_path: str | None = None,
    include_sam_large: bool = False,
) -> dict:
    """Run real CPU and (when CUDA is available) GPU benchmarks + hardware suitability report."""
    started = time.perf_counter()
    img_path, should_cleanup_img = _pick_benchmark_image(sample_image_path)

    sam_tiny_path = resolve_weight_path(weights_dir, base_dir, "sam2.1_t.pt")
    sam_large_path = resolve_weight_path(weights_dir, base_dir, "sam2.1_l.pt")
    clip_path = resolve_weight_path(weights_dir, base_dir, os.path.join("clip", "ViT-B-32.pt"))

    def _bench_suite(dev: str, tensor_runs: int, sam_tiny_runs: int, clip_runs: int, sam_large_runs: int) -> dict:
        return {
            "executed": True,
            "tensor": _benchmark_tensor_device(dev, warmup=(2 if dev == "cuda" else 1), runs=tensor_runs),
            "sam_tiny": _benchmark_sam_model(sam_tiny_path, img_path, dev, runs=sam_tiny_runs),
            "clip": _benchmark_clip_model(clip_path, img_path, dev, runs=clip_runs),
            "sam_large": (
                _benchmark_sam_model(sam_large_path, img_path, dev, runs=sam_large_runs)
                if include_sam_large
                else {"executed": False, "skipped": True}
            ),
        }

    try:
        cpu_bench = _bench_suite("cpu", 5, 2, 3, 1)
        gpu_bench = (
            _bench_suite("cuda", 8, 3, 5, 2)
            if hw_info.get("cuda_available")
            else {
                "executed": False,
                "reason": (
                    f"PyTorch hiện tại ({hw_info.get('torch_version')}) chưa có CUDA backend "
                    f"(torch.cuda.is_available() == False)."
                ),
                "install_command": hw_info.get("cuda_install_command"),
            }
        )

        suitability = evaluate_gpu_suitability(hw_info, cpu_bench, gpu_bench, weights_dir, base_dir)
        elapsed_sec = round(time.perf_counter() - started, 2)

        return {
            "success": True,
            "elapsed_sec": elapsed_sec,
            "sample_image": os.path.basename(img_path) if not should_cleanup_img else "synthetic_640x480.jpg",
            "hardware": hw_info,
            "benchmarks": {
                "cpu": cpu_bench,
                "gpu": gpu_bench,
            },
            "suitability": suitability,
        }
    finally:
        if should_cleanup_img and os.path.exists(img_path):
            try:
                os.remove(img_path)
            except Exception:
                pass
