"""What this machine can train on: OS, CPU, memory, disk and every GPU.

Standard library only, and never imports torch: it runs before any ML stack
is installed, to decide which one to install. Every probe is best-effort with
a short timeout — a missing tool or an odd driver leaves a field empty rather
than failing the command.

    >>> from systemone.hardware import detect
    >>> machine = detect()
    >>> machine.accelerator            # "cuda", "rocm", "xpu", "mlx", "directml" or "cpu"
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

TIMEOUT = 6


@dataclass
class Gpu:
    vendor: str  # nvidia | amd | intel | apple | other
    name: str
    memory_bytes: int | None = None
    driver: str | None = None
    # The software stack that can drive it: "cuda 12.4", "rocm 6.2", "metal", "xpu", "directml".
    runtime: str | None = None


@dataclass
class Machine:
    os: str
    os_version: str
    arch: str
    cpu: str
    cores: int
    memory_bytes: int | None
    disk_free_bytes: int | None
    gpus: list[Gpu] = field(default_factory=list)
    # Apple silicon: the GPU shares system memory.
    unified_memory: bool = False

    @property
    def accelerator(self) -> str:
        """The backend training should use here, best first."""
        vendors = {g.vendor for g in self.gpus}
        if "nvidia" in vendors:
            return "cuda"
        if "apple" in vendors:
            return "mlx"
        if "amd" in vendors:
            return "rocm" if self.os == "Linux" else "directml"
        if "intel" in vendors and any((g.memory_bytes or 0) >= 4 * 1024**3 for g in self.gpus):
            return "xpu"
        return "cpu"

    @property
    def training_memory_bytes(self) -> int | None:
        """Memory a model can be trained in.

        The largest GPU's, or system memory on unified-memory and CPU-only machines.
        """
        if self.unified_memory or not self.gpus or self.accelerator == "cpu":
            return self.memory_bytes
        sizes = [g.memory_bytes for g in self.gpus if g.memory_bytes]
        return max(sizes) if sizes else None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["accelerator"] = self.accelerator
        data["training_memory_bytes"] = self.training_memory_bytes
        return data


def _run(cmd: list[str]) -> str | None:
    exe = shutil.which(cmd[0])
    if exe is None:
        return None
    try:
        done = subprocess.run(  # noqa: S603 - fixed argument lists of system tools
            [exe, *cmd[1:]], capture_output=True, text=True, timeout=TIMEOUT, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


# --- memory, CPU, disk -------------------------------------------------------


def _memory() -> int | None:
    system = platform.system()
    if system == "Darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        return int(out.strip()) if out and out.strip().isdigit() else None
    if system == "Linux":
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        except OSError:
            return None
    if system == "Windows":
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [  # noqa: RUF012 - ctypes layout
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = Status()
        status.dwLength = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return int(status.ullTotalPhys)
    return None


def _cpu() -> str:
    system = platform.system()
    if system == "Darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out and out.strip():
            return out.strip()
    if system == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith(("model name", "hardware", "cpu model")):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if system == "Windows":
        out = _run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]
        )
        if out and out.strip():
            return out.strip().splitlines()[0]
    return platform.processor() or platform.machine()


def _os_version() -> str:
    system = platform.system()
    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0]}"
    if system == "Windows":
        return f"Windows {platform.release()} ({platform.version()})"
    try:
        info = dict(
            line.split("=", 1)
            for line in Path("/etc/os-release").read_text().splitlines()
            if "=" in line
        )
        return info.get("PRETTY_NAME", "").strip('"') or f"Linux {platform.release()}"
    except OSError:
        return f"{system} {platform.release()}"


# --- GPUs --------------------------------------------------------------------


def _nvidia() -> list[Gpu]:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return []
    header = _run(["nvidia-smi"]) or ""
    cuda = re.search(r"CUDA Version:\s*([\d.]+)", header)
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        name, mib, driver = parts[0], parts[1], parts[2]
        gpus.append(
            Gpu(
                "nvidia",
                name,
                int(float(mib)) * 1024**2 if re.fullmatch(r"[\d.]+", mib) else None,
                driver,
                f"cuda {cuda.group(1)}" if cuda else "cuda",
            )
        )
    return gpus


def _amd_linux() -> list[Gpu]:
    out = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    if not out:
        return []
    try:
        data = json.loads(out)
    except ValueError:
        return []
    version = _run(["rocm-smi", "--showdriverversion"]) or ""
    gpus = []
    for key, card in data.items():
        if not key.startswith("card") or not isinstance(card, dict):
            continue
        name = (
            card.get("Card Series") or card.get("Card series") or card.get("Card SKU") or "AMD GPU"
        )
        total = str(card.get("VRAM Total Memory (B)") or "")
        gpus.append(Gpu("amd", str(name), int(total) if total.isdigit() else None, None, "rocm"))
    if gpus and (match := re.search(r"Driver version:\s*(\S+)", version)):
        for gpu in gpus:
            gpu.driver = match.group(1)
    return gpus


def _lspci() -> list[Gpu]:
    out = _run(["lspci"])
    if not out:
        return []
    gpus = []
    for line in out.splitlines():
        if not re.search(r"VGA compatible|3D controller|Display controller", line):
            continue
        description = line.split(": ", 1)[-1]
        lower = description.lower()
        vendor = (
            "nvidia"
            if "nvidia" in lower
            else "amd"
            if "amd" in lower or "ati " in lower or "radeon" in lower
            else "intel"
            if "intel" in lower
            else "other"
        )
        gpus.append(Gpu(vendor, description))
    return gpus


_WINDOWS_GPUS = r"""
$key = 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}'
Get-ChildItem $key -ErrorAction SilentlyContinue | ForEach-Object {
  $p = Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue
  if ($p.DriverDesc) {
    [pscustomobject]@{
      Name = $p.DriverDesc
      Memory = $p.'HardwareInformation.qwMemorySize'
      Driver = $p.DriverVersion
    }
  }
} | ConvertTo-Json -Compress
"""


def _windows() -> list[Gpu]:
    # The registry has the real VRAM size; Win32_VideoController caps it at 4 GB.
    out = _run(["powershell", "-NoProfile", "-Command", _WINDOWS_GPUS])
    if not out or not out.strip():
        return []
    try:
        data = json.loads(out)
    except ValueError:
        return []
    rows = data if isinstance(data, list) else [data]
    gpus = []
    for row in rows:
        name = str(row.get("Name") or "")
        if not name or "basic display" in name.lower() or "remote" in name.lower():
            continue
        lower = name.lower()
        vendor = (
            "nvidia"
            if "nvidia" in lower
            else "amd"
            if "amd" in lower or "radeon" in lower
            else "intel"
            if "intel" in lower
            else "other"
        )
        memory = row.get("Memory")
        gpus.append(
            Gpu(
                vendor,
                name,
                int(memory) if isinstance(memory, int) else None,
                row.get("Driver"),
                "directml",
            )
        )
    return gpus


def _apple() -> list[Gpu]:
    if platform.machine().lower() not in ("arm64", "aarch64"):
        return []
    name = "Apple GPU"
    out = _run(["system_profiler", "SPDisplaysDataType", "-json"])
    if out:
        try:
            display = json.loads(out).get("SPDisplaysDataType", [{}])[0]
            chip = display.get("sppci_model") or display.get("_name")
            cores = display.get("sppci_cores")
            if chip:
                name = f"{chip} ({cores}-core GPU)" if cores else str(chip)
        except (ValueError, IndexError, AttributeError):
            pass
    return [Gpu("apple", name, _memory(), None, "metal")]


def _gpus() -> list[Gpu]:
    system = platform.system()
    if system == "Darwin":
        return _apple()
    nvidia = _nvidia()
    if system == "Linux":
        return (nvidia + _amd_linux()) or _lspci()
    if system == "Windows":
        # nvidia-smi describes NVIDIA cards better (CUDA version); the registry covers the rest.
        others = [g for g in _windows() if not (nvidia and g.vendor == "nvidia")]
        return nvidia + others
    return nvidia


def detect(workdir: Path | None = None) -> Machine:
    """Describe this machine. `workdir` is where free disk space is measured."""
    try:
        free: int | None = shutil.disk_usage(workdir or Path.home()).free
    except OSError:
        free = None
    gpus = _gpus()
    return Machine(
        os=platform.system(),
        os_version=_os_version(),
        arch=platform.machine(),
        cpu=_cpu(),
        cores=os.cpu_count() or 1,
        memory_bytes=_memory(),
        disk_free_bytes=free,
        gpus=gpus,
        unified_memory=any(g.vendor == "apple" for g in gpus),
    )


ACCELERATOR_LABEL = {
    "cuda": "NVIDIA CUDA",
    "rocm": "AMD ROCm",
    "directml": "DirectML (AMD or Intel on Windows)",
    "xpu": "Intel XPU",
    "mlx": "Apple MLX (Metal)",
    "cpu": "CPU only",
}
