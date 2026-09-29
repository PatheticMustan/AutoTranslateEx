"""Run llama.cpp's llama-server as a subprocess, GPU first with CPU fallback.

Only one LLM runs at a time; switching tiers means stopping one server and
starting another. The server log goes to models/llama-server.log.
"""

from __future__ import annotations

import atexit
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx

from atx import device as devmod
from atx.models import MODELS_DIR, gguf_path, llama_server_exe

LOG = MODELS_DIR / "llama-server.log"
START_TIMEOUT_S = 180

_job = None  # Windows job object holding every llama-server we start


def _kill_with_parent(proc: subprocess.Popen) -> None:
    """Put the process in a job that Windows kills when this Python process exits.

    atexit doesn't run if the console window is closed or Python crashes, and an
    orphaned llama-server keeps its VRAM and port (the next start would then
    talk to the old one).
    """
    global _job
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if _job is None:
        class LimitInfo(ctypes.Structure):  # JOBOBJECT_EXTENDED_LIMIT_INFORMATION
            _fields_ = [("basic", ctypes.c_byte * (64 if ctypes.sizeof(ctypes.c_void_p) == 8 else 48)),
                        ("io", ctypes.c_byte * 48), ("sizes", ctypes.c_size_t * 4)]

        k32.CreateJobObjectW.restype = wintypes.HANDLE
        job = k32.CreateJobObjectW(None, None)
        info = LimitInfo()
        # BasicLimitInformation.LimitFlags sits at offset 16 on both 32- and 64-bit.
        ctypes.c_uint32.from_buffer(info.basic, 16).value = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info)):
            return  # 9 = JobObjectExtendedLimitInformation
        _job = job
    k32.AssignProcessToJobObject(wintypes.HANDLE(_job), wintypes.HANDLE(int(proc._handle)))


class LlamaServer:
    def __init__(self, model: str, *, parallel: int = 1, ctx_per_slot: int = 2048,
                 port: int = 8766, threads: int = devmod.CPU_THREADS):
        self.model = model
        self.parallel = parallel
        self.ctx = ctx_per_slot * parallel
        self.port = port
        self.threads = threads
        self.url = f"http://127.0.0.1:{port}"
        self.device = "not started"
        self.offload = ""  # e.g. "25/25 layers on GPU", from the server log
        self.load_s = 0.0
        self._proc: subprocess.Popen | None = None
        atexit.register(self.stop)

    def start(self) -> None:
        build = devmod.detect().llama_build
        if build != "cpu":
            try:
                self._launch(build, gpu_layers=99)
                self.device = f"gpu ({build} build)"
                return
            except RuntimeError as e:
                devmod.record_fallback("llm", f"{build} build failed: {e}")
                self.stop()
        self._launch("cpu", gpu_layers=0)
        self.device = "cpu"

    def _launch(self, build: str, gpu_layers: int) -> None:
        exe = llama_server_exe(build)
        if not exe.exists():
            raise RuntimeError(f"{exe} missing; run `python -m atx.models {build}`")
        cmd = [
            str(exe), "-m", str(gguf_path(self.model)),
            "--host", "127.0.0.1", "--port", str(self.port),
            "-ngl", str(gpu_layers), "-c", str(self.ctx), "-np", str(self.parallel),
            "-t", str(self.threads), "--no-webui", "--jinja",
            "-lv", "4",  # the default level doesn't log which device holds the model
        ]
        t0 = time.perf_counter()
        log = LOG.open("w", encoding="utf-8")
        self._proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                      creationflags=subprocess.CREATE_NO_WINDOW)
        _kill_with_parent(self._proc)
        while time.perf_counter() - t0 < START_TIMEOUT_S:
            if self._proc.poll() is not None:
                raise RuntimeError(f"exited with code {self._proc.returncode} (see {LOG})")
            try:
                if httpx.get(f"{self.url}/health", timeout=2).status_code == 200:
                    self.load_s = time.perf_counter() - t0
                    self.offload = self._read_offload()
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        raise RuntimeError(f"not ready after {START_TIMEOUT_S}s (see {LOG})")

    @staticmethod
    def _read_offload() -> str:
        """Which device holds the model, from the server log, e.g. 'CUDA0 (NVIDIA ...), 1156 MiB'."""
        log = LOG.read_text("utf-8", errors="replace")
        dev = re.search(r"using device (\S+) \(([^)]+)\)", log)
        if not dev:
            return "CPU"
        mem = re.search(r"projected to use (\d+) MiB of device memory", log)
        return f"{dev[1]} ({dev[2]})" + (f", {mem[1]} MiB" if mem else "")

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def __enter__(self) -> LlamaServer:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
