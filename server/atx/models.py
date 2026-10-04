"""Download translation models and llama.cpp builds into server/models/.

    python -m atx.models                 # what this machine needs (no benchmark-only models)
    python -m atx.models qwen3.5-4b      # specific entries
    python -m atx.models --list

Files are streamed to <name>.part and renamed when complete; finished files are
skipped, so re-running resumes cheaply. llama.cpp builds are unzipped into
models/llama.cpp/<build>/.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from atx import device

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

LLAMA_TAG = "b11206"  # 2026-09-27
LLAMA_RELEASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}"
LLAMA_BUILDS = {
    "cuda": [f"llama-{LLAMA_TAG}-bin-win-cuda-13.4-x64.zip", "cudart-llama-bin-win-cuda-13.4-x64.zip"],
    "vulkan": [f"llama-{LLAMA_TAG}-bin-win-vulkan-x64.zip"],
    "sycl": [f"llama-{LLAMA_TAG}-bin-win-sycl-x64.zip"],
    "openvino": [f"llama-{LLAMA_TAG}-bin-win-openvino-2026.4-x64.zip"],
    "cpu": [f"llama-{LLAMA_TAG}-bin-win-cpu-x64.zip"],
}


@dataclass
class HfModel:
    repo: str
    files: list[str]
    # llama-server --override-kv fixes for broken metadata, e.g. "tokenizer.ggml.eos_token_id=int:120020"
    override_kv: list[str] = field(default_factory=list)

    def urls(self) -> list[tuple[str, str]]:
        return [(f"https://huggingface.co/{self.repo}/resolve/main/{f}", f) for f in self.files]


MODELS: dict[str, HfModel] = {
    # Very quick tier: MarianMT already converted to CTranslate2.
    "opus-mt-zh-en": HfModel("gaudi/opus-mt-zh-en-ctranslate2",
                             ["model.bin", "config.json", "shared_vocabulary.json", "source.spm", "target.spm"]),
    # Quick tier.
    "hy-mt2-1.8b": HfModel("tencent/Hy-MT2-1.8B-GGUF", ["Hy-MT2-1.8B-Q4_K_M.gguf"]),
    # Accurate tier: 2B on the target, 4B on a discrete GPU.
    "qwen3.5-2b": HfModel("unsloth/Qwen3.5-2B-GGUF", ["Qwen3.5-2B-Q4_K_M.gguf"]),
    "qwen3.5-4b": HfModel("unsloth/Qwen3.5-4B-GGUF", ["Qwen3.5-4B-Q4_K_M.gguf"]),
}

# Other quantizations of Hy-MT2-1.8B, for bench.bench_quant only (never downloaded
# by default). Q8_0 is the near-lossless reference; the rest are importance-matrix
# quants in standard formats, which stock llama.cpp runs on every backend.
# Tencent's own 2-bit and 1.25-bit files need unmerged, CPU-only (ARM-optimized)
# kernels, so they're left out.
# mradermacher's files set eos_token_id to 3 ("$") instead of Tencent's 120020, so
# generation never stops (the answer, then rambling until max_tokens).
_I1 = "mradermacher/Hy-MT2-1.8B-i1-GGUF"
_I1_FIX = ["tokenizer.ggml.eos_token_id=int:120020"]
HY_MT_QUANTS: dict[str, HfModel] = {
    "hy-mt2-1.8b-q8_0": HfModel("tencent/Hy-MT2-1.8B-GGUF", ["Hy-MT2-1.8B-Q8_0.gguf"]),
    **{f"hy-mt2-1.8b-{q.lower()}": HfModel(_I1, [f"Hy-MT2-1.8B.i1-{q}.gguf"], _I1_FIX)
       for q in ["IQ4_XS", "Q3_K_M", "IQ3_XXS", "Q2_K", "IQ2_M"]},
}
MODELS.update(HY_MT_QUANTS)

# Speech-bubble and text detector (RT-DETR, Apache-2.0), small int8 variant: see atx/detector.py.
MODELS["comic-detector"] = HfModel("ogkalu/comic-text-and-bubble-detector", ["detector-v4-s_int8.onnx"])
DISCRETE_ONLY = {"qwen3.5-4b"}   # the accurate tier uses it only on a discrete GPU
BENCH_ONLY = {"qwen3.5-2b", *HY_MT_QUANTS}


def model_path(name: str) -> Path:
    """Directory of a downloaded model (the GGUF file for LLMs: see gguf_path)."""
    return MODELS_DIR / name


def gguf_path(name: str) -> Path:
    return model_path(name) / MODELS[name].files[0]


def llama_server_exe(build: str) -> Path:
    return MODELS_DIR / "llama.cpp" / build / "llama-server.exe"


def _download(client: httpx.Client, url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  have {dest.relative_to(MODELS_DIR)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with client.stream("GET", url) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        done, next_report = 0, 0.0
        with part.open("wb") as f:
            for chunk in resp.iter_bytes(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total and done / total >= next_report:
                    print(f"  {dest.name}: {done / 2**20:,.0f} / {total / 2**20:,.0f} MB", flush=True)
                    next_report += 0.25
    part.rename(dest)


def download_model(client: httpx.Client, name: str) -> None:
    print(f"{name}:")
    for url, filename in MODELS[name].urls():
        _download(client, url, model_path(name) / filename)


def download_llama(client: httpx.Client, build: str) -> None:
    print(f"llama.cpp {build} ({LLAMA_TAG}):")
    target = MODELS_DIR / "llama.cpp" / build
    for asset in LLAMA_BUILDS[build]:
        zip_path = MODELS_DIR / "llama.cpp" / "_zips" / asset
        _download(client, f"{LLAMA_RELEASE}/{asset}", zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(target)
    if not llama_server_exe(build).exists():
        raise SystemExit(f"llama-server.exe missing after unzipping {build}")


def default_targets() -> tuple[list[str], list[str]]:
    builds = [device.detect().llama_build]
    if builds[0] != "cpu":
        builds.append("cpu")  # always have a CPU fallback build
    gpu = device.detect().primary_gpu
    discrete = gpu is not None and gpu.vendor in ("nvidia", "amd")
    skip = BENCH_ONLY | (set() if discrete else DISCRETE_ONLY)
    return [m for m in MODELS if m not in skip], builds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help=f"models {list(MODELS)} or llama builds {list(LLAMA_BUILDS)}")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.list:
        for name, m in MODELS.items():
            print(f"{name:15s} {m.repo}  {', '.join(m.files)}")
        for build, assets in LLAMA_BUILDS.items():
            print(f"llama {build:9s} {', '.join(assets)}")
        return

    if args.names:
        models = [n for n in args.names if n in MODELS]
        builds = [n for n in args.names if n in LLAMA_BUILDS]
        unknown = set(args.names) - set(models) - set(builds)
        if unknown:
            raise SystemExit(f"unknown: {sorted(unknown)}")
    else:
        models, builds = default_targets()

    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60, read=300)) as client:
        for b in builds:
            download_llama(client, b)
        for m in models:
            download_model(client, m)


if __name__ == "__main__":
    main()
