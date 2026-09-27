"""Detect the compute device once and pick a GPU-first path for each runtime.

Policy: always use a GPU when one is usable, otherwise fall back to CPU. Every
fallback records a reason so /health and the popup can show it.

Run `python -m atx.device` to print what was detected.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from functools import cache

# Order of preference for onnxruntime execution providers.
_ONNX_PREFERENCE = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]


@dataclass
class Gpu:
    name: str
    vendor: str  # "nvidia" | "amd" | "intel" | "other"
    vram_mb: int | None = None


@dataclass
class DeviceInfo:
    gpus: list[Gpu]
    onnx_providers: list[str]   # pass to onnxruntime.InferenceSession(providers=...)
    ct2_device: str             # "cuda" | "cpu"
    llama_build: str            # "cuda" | "vulkan" | "cpu"
    fallbacks: dict[str, str] = field(default_factory=dict)  # component -> reason

    @property
    def primary_gpu(self) -> Gpu | None:
        return self.gpus[0] if self.gpus else None

    def summary(self) -> str:
        """One line for the popup's status, e.g. 'GPU: NVIDIA GeForce RTX 3070 Ti'."""
        gpu = self.primary_gpu
        if gpu is None:
            return "CPU (no GPU found)"
        if self.fallbacks:
            return f"GPU: {gpu.name} (CPU fallback: {', '.join(self.fallbacks)})"
        return f"GPU: {gpu.name}"


def _run(cmd: list[str], timeout: float = 10) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
        return out.stdout
    except (OSError, subprocess.SubprocessError):
        return None


def _vendor(name: str) -> str:
    n = name.lower()
    if "nvidia" in n or "geforce" in n or "quadro" in n or "rtx" in n:
        return "nvidia"
    if "amd" in n or "radeon" in n:
        return "amd"
    if "intel" in n:
        return "intel"
    return "other"


def _nvidia_gpus() -> list[Gpu]:
    if not shutil.which("nvidia-smi"):
        return []
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    gpus = []
    for line in (out or "").splitlines():
        name, _, vram = line.partition(",")
        if name.strip():
            gpus.append(Gpu(name.strip(), "nvidia", int(vram) if vram.strip().isdigit() else None))
    return gpus


def _windows_adapters() -> list[Gpu]:
    if sys.platform != "win32":
        return []
    out = _run([
        "powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }",
    ])
    names = [n.strip() for n in (out or "").splitlines() if n.strip()]
    # Skip software/remote adapters that can't run compute.
    names = [n for n in names if "basic" not in n.lower() and "remote" not in n.lower()]
    return [Gpu(n, _vendor(n)) for n in names]


def detect_gpus() -> list[Gpu]:
    """All usable GPUs, best first: NVIDIA (with VRAM from nvidia-smi), then AMD, then Intel."""
    gpus = _nvidia_gpus()
    seen = {g.name for g in gpus}
    gpus += [g for g in _windows_adapters() if g.name not in seen and g.vendor != "nvidia"]
    rank = {"nvidia": 0, "amd": 1, "intel": 2, "other": 3}
    return sorted(gpus, key=lambda g: rank[g.vendor])


def _onnx_providers(fallbacks: dict[str, str]) -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError:
        fallbacks["ocr"] = "onnxruntime not installed"
        return ["CPUExecutionProvider"]
    available = set(ort.get_available_providers())
    providers = [p for p in _ONNX_PREFERENCE if p in available]
    if providers == ["CPUExecutionProvider"]:
        fallbacks["ocr"] = f"no GPU execution provider (available: {sorted(available)})"
    return providers or ["CPUExecutionProvider"]


def _ct2_device(fallbacks: dict[str, str]) -> str:
    try:
        import ctranslate2
    except ImportError:
        fallbacks["very_quick"] = "ctranslate2 not installed"
        return "cpu"
    try:
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
        fallbacks["very_quick"] = "no CUDA device visible to CTranslate2"
    except Exception as e:  # missing CUDA DLLs surface here on Windows
        fallbacks["very_quick"] = f"CUDA unavailable: {e}"
    return "cpu"


@cache
def detect() -> DeviceInfo:
    gpus = detect_gpus()
    fallbacks: dict[str, str] = {}

    if any(g.vendor == "nvidia" for g in gpus):
        llama_build = "cuda"
    elif gpus:
        llama_build = "vulkan"
    else:
        llama_build = "cpu"
        fallbacks["llm"] = "no GPU found"

    info = DeviceInfo(
        gpus=gpus,
        onnx_providers=_onnx_providers(fallbacks),
        ct2_device=_ct2_device(fallbacks) if gpus else "cpu",
        llama_build=llama_build,
        fallbacks=fallbacks,
    )
    if not gpus:
        # Don't report per-component reasons when the answer is simply "no GPU".
        info.fallbacks = {"all": "no GPU found"}
    return info


def record_fallback(component: str, reason: str) -> None:
    """Called by runtimes when a GPU path fails at load time and they drop to CPU."""
    detect().fallbacks[component] = reason


if __name__ == "__main__":
    info = detect()
    print(info.summary())
    print(json.dumps(asdict(info), indent=2))
