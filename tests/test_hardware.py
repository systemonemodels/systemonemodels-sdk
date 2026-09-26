"""Machine detection: parse what the system tools print, pick the right backend."""

from __future__ import annotations

import json

import pytest

from systemone import hardware

NVIDIA_CSV = "NVIDIA GeForce RTX 4090, 24564, 560.94\nNVIDIA GeForce RTX 3060, 12288, 560.94\n"
NVIDIA_HEADER = "| NVIDIA-SMI 560.94   Driver Version: 560.94   CUDA Version: 12.6 |"


def fake_tools(outputs: dict[str, str | None]):  # type: ignore[no-untyped-def]
    def run(cmd: list[str]) -> str | None:
        key = " ".join(cmd)
        for prefix, out in outputs.items():
            if key.startswith(prefix):
                return out
        return None

    return run


def test_nvidia_cards_and_cuda_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        hardware,
        "_run",
        fake_tools({"nvidia-smi --query-gpu": NVIDIA_CSV, "nvidia-smi": NVIDIA_HEADER}),
    )
    gpus = hardware._nvidia()
    assert [g.name for g in gpus] == ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3060"]
    assert gpus[0].memory_bytes == 24564 * 1024**2
    assert gpus[0].runtime == "cuda 12.6" and gpus[0].driver == "560.94"


def test_amd_on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    data = {"card0": {"Card Series": "Radeon RX 7900 XTX", "VRAM Total Memory (B)": "25753026560"}}
    monkeypatch.setattr(
        hardware,
        "_run",
        fake_tools(
            {
                "rocm-smi --showproductname": json.dumps(data),
                "rocm-smi --showdriverversion": "Driver version: 6.8.5",
            }
        ),
    )
    [gpu] = hardware._amd_linux()
    assert gpu.vendor == "amd" and gpu.memory_bytes == 25753026560 and gpu.driver == "6.8.5"


def test_windows_registry_sizes_beyond_4_gb(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"Name": "AMD Radeon RX 7800 XT", "Memory": 17163091968, "Driver": "32.0.11021"},
        {"Name": "Microsoft Basic Display Adapter", "Memory": None, "Driver": "10.0"},
    ]
    monkeypatch.setattr(hardware, "_run", fake_tools({"powershell": json.dumps(rows)}))
    [gpu] = hardware._windows()
    assert gpu.vendor == "amd" and gpu.memory_bytes == 17163091968 and gpu.runtime == "directml"


def test_lspci_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    out = (
        "00:02.0 VGA compatible controller: Intel Corporation Arc A770 Graphics\n"
        "00:1f.3 Audio device: Intel\n"
    )
    monkeypatch.setattr(hardware, "_run", fake_tools({"lspci": out}))
    [gpu] = hardware._lspci()
    assert gpu.vendor == "intel" and "Arc A770" in gpu.name


def machine(gpus: list[hardware.Gpu], os_name: str = "Linux") -> hardware.Machine:
    shared = any(g.vendor == "apple" for g in gpus)
    return hardware.Machine(os_name, "x", "x86_64", "cpu", 8, 64 * 1024**3, None, gpus, shared)


@pytest.mark.parametrize(
    ("gpus", "os_name", "expected"),
    [
        ([hardware.Gpu("nvidia", "RTX", 8 * 1024**3)], "Windows", "cuda"),
        ([hardware.Gpu("amd", "RX", 16 * 1024**3)], "Linux", "rocm"),
        ([hardware.Gpu("amd", "RX", 16 * 1024**3)], "Windows", "directml"),
        ([hardware.Gpu("intel", "Arc A770", 16 * 1024**3)], "Linux", "xpu"),
        ([hardware.Gpu("intel", "UHD 620", 1024**3)], "Linux", "cpu"),
        ([hardware.Gpu("apple", "M4", 16 * 1024**3)], "Darwin", "mlx"),
        ([], "Linux", "cpu"),
    ],
)
def test_accelerator(gpus: list[hardware.Gpu], os_name: str, expected: str) -> None:
    assert machine(gpus, os_name).accelerator == expected


def test_training_memory_is_the_gpu_unless_memory_is_shared() -> None:
    assert (
        machine([hardware.Gpu("nvidia", "RTX", 8 * 1024**3)]).training_memory_bytes == 8 * 1024**3
    )
    assert (
        machine([hardware.Gpu("apple", "M4", 16 * 1024**3)], "Darwin").training_memory_bytes
        == 64 * 1024**3
    )
    assert machine([]).training_memory_bytes == 64 * 1024**3


def test_detect_runs_here() -> None:
    found = hardware.detect()
    assert found.cores >= 1 and found.os
    assert found.to_dict()["accelerator"] in hardware.ACCELERATOR_LABEL
