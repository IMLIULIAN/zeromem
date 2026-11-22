"""Zero-dependency hardware & GPU/VRAM probe for automatic V2 quantization selection."""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HardwareProfile:
    os_name: str
    arch: str
    ram_gb: float
    gpu_vendor: str  # "nvidia" | "apple" | "amd" | "cpu"
    gpu_name: str
    vram_gb: float
    recommend_quant: str  # "q8_0" | "q4_k_m"
    recommend_num_ctx: int
    recommend_timeout: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_line(self) -> str:
        if self.gpu_vendor == "cpu":
            return f"{self.os_name}/{self.arch} · CPU 模式 (RAM {self.ram_gb:.1f} GB) → 推荐 {self.recommend_quant.upper()} / ctx={self.recommend_num_ctx}"
        return (
            f"{self.os_name}/{self.arch} · {self.gpu_name} ({self.vram_gb:.1f} GB) · "
            f"RAM {self.ram_gb:.1f} GB → 推荐 {self.recommend_quant.upper()} / ctx={self.recommend_num_ctx}"
        )


def _detect_ram_gb() -> float:
    sys_name = platform.system()
    try:
        if sys_name == "Windows":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return round(st.ullTotalPhys / (1024 ** 3), 1)
        elif sys_name == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=3, text=True).strip()
            return round(int(out) / (1024 ** 3), 1)
        elif Path("/proc/meminfo").is_file():
            for line in Path("/proc/meminfo").read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 ** 2), 1)
    except Exception:
        pass
    return 0.0


def _detect_nvidia_gpu() -> tuple[str, float] | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        kwargs: dict[str, Any] = {"timeout": 4, "text": True, "capture_output": True, "check": False}
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        proc = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            **kwargs,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        best_name = ""
        best_mb = 0.0
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    mb = float(parts[1])
                except ValueError:
                    continue
                if mb > best_mb:
                    best_mb = mb
                    best_name = parts[0]
        if best_mb > 0:
            return best_name, round(best_mb / 1024.0, 1)
    except Exception:
        return None
    return None


def _detect_apple_silicon(ram_gb: float) -> tuple[str, float] | None:
    if platform.system() != "Darwin":
        return None
    arch = platform.machine().lower()
    if arch not in ("arm64", "aarch64"):
        return None
    chip = "Apple Silicon (Metal)"
    try:
        brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=3, text=True).strip()
        if brand:
            chip = f"{brand} (Metal)"
    except Exception:
        pass
    # macOS Metal can allocate ~70% of unified memory for GPU workloads
    usable_gb = round(ram_gb * 0.7, 1) if ram_gb > 0 else 8.0
    return chip, usable_gb


def _detect_amd_rocm() -> tuple[str, float] | None:
    exe = shutil.which("rocm-smi")
    if not exe:
        return None
    try:
        proc = subprocess.run([exe, "--showmeminfo", "vram"], timeout=4, text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            return None
        for line in proc.stdout.splitlines():
            if "Total Memory" in line:
                nums = [tok for tok in line.replace(":", " ").split() if tok.isdigit()]
                if nums:
                    gb = round(int(nums[-1]) / (1024 ** 3), 1)
                    if gb > 0:
                        return "AMD Radeon (ROCm)", gb
    except Exception:
        return None
    return None


def detect_hardware() -> HardwareProfile:
    """Detect local OS, RAM and GPU VRAM, returning safe V2 defaults."""
    os_name = platform.system()
    arch = platform.machine() or "unknown"
    ram_gb = _detect_ram_gb()

    nv = _detect_nvidia_gpu()
    if nv is not None:
        gpu_name, vram_gb = nv
        if vram_gb >= 7.5:
            return HardwareProfile(
                os_name=os_name,
                arch=arch,
                ram_gb=ram_gb,
                gpu_vendor="nvidia",
                gpu_name=gpu_name,
                vram_gb=vram_gb,
                recommend_quant="q8_0",
                recommend_num_ctx=8192,
                recommend_timeout=90.0,
                reason="显存 >= 7.5 GB，推荐精度最高的 Q8_0 (4.48 GB) + 8K 上下文",
            )
        if vram_gb >= 4.0:
            ctx = 8192 if vram_gb >= 5.5 else 4096
            return HardwareProfile(
                os_name=os_name,
                arch=arch,
                ram_gb=ram_gb,
                gpu_vendor="nvidia",
                gpu_name=gpu_name,
                vram_gb=vram_gb,
                recommend_quant="q4_k_m",
                recommend_num_ctx=ctx,
                recommend_timeout=120.0,
                reason="显存 4–7 GB，推荐轻量 Q4_K_M (2.71 GB) 防止爆显存回落 CPU",
            )

    apple = _detect_apple_silicon(ram_gb)
    if apple is not None:
        gpu_name, usable_gb = apple
        if ram_gb >= 15.5:
            return HardwareProfile(
                os_name=os_name,
                arch=arch,
                ram_gb=ram_gb,
                gpu_vendor="apple",
                gpu_name=gpu_name,
                vram_gb=usable_gb,
                recommend_quant="q8_0",
                recommend_num_ctx=8192,
                recommend_timeout=90.0,
                reason="Apple Silicon >= 16 GB 统一内存，推荐 Q8_0 + 8K 上下文",
            )
        mem_desc = f"{ram_gb:.1f} GB (<16 GB)" if ram_gb > 0 else "不足 16 GB"
        return HardwareProfile(
            os_name=os_name,
            arch=arch,
            ram_gb=ram_gb,
            gpu_vendor="apple",
            gpu_name=gpu_name,
            vram_gb=usable_gb,
            recommend_quant="q4_k_m",
            recommend_num_ctx=4096,
            recommend_timeout=150.0,
            reason=f"Apple Silicon 统一内存 {mem_desc}，推荐轻量 Q4_K_M + 4K 上下文",
        )

    amd = _detect_amd_rocm()
    if amd is not None:
        gpu_name, vram_gb = amd
        quant = "q8_0" if vram_gb >= 7.5 else "q4_k_m"
        ctx = 8192 if vram_gb >= 5.5 else 4096
        return HardwareProfile(
            os_name=os_name,
            arch=arch,
            ram_gb=ram_gb,
            gpu_vendor="amd",
            gpu_name=gpu_name,
            vram_gb=vram_gb,
            recommend_quant=quant,
            recommend_num_ctx=ctx,
            recommend_timeout=120.0,
            reason=f"AMD ROCm 显存 {vram_gb:.1f} GB，推荐 {quant.upper()} + {ctx} 上下文",
        )

    return HardwareProfile(
        os_name=os_name,
        arch=arch,
        ram_gb=ram_gb,
        gpu_vendor="cpu",
        gpu_name="CPU / 核显",
        vram_gb=0.0,
        recommend_quant="q4_k_m",
        recommend_num_ctx=4096,
        recommend_timeout=240.0,
        reason="未检测到独立大显存 GPU，推荐 Q4_K_M + 4K 上下文并放宽超时至 240s（必要时写入可加 --no-verify）",
    )
