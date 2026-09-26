"""Machine detection: parse what the system tools print, pick the right backend."""

from __future__ import annotations

import json

import pytest

from systemone import hardware

NVIDIA_CSV = (
    "NVIDIA GeForce RTX 4090, 24564, 560.94, 8.9\nNVIDIA GeForce RTX 3060, 12288, 560.94, 8.6\n"
)
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
        fake_tools({"nvidia-smi --format": NVIDIA_CSV, "nvidia-smi": NVIDIA_HEADER}),
    )
    gpus = hardware._nvidia()
    assert [g.name for g in gpus] == ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3060"]
    assert gpus[0].memory_bytes == 24564 * 1024**2
    assert gpus[0].runtime == "cuda 12.6" and gpus[0].driver == "560.94" and gpus[0].arch == "8.9"


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
    assert gpu.vendor == "amd" and gpu.memory_bytes == 17163091968


def test_lspci_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    out = (
        "00:02.0 VGA compatible controller: Intel Corporation Arc A770 Graphics\n"
        "00:1f.3 Audio device: Intel\n"
    )
    monkeypatch.setattr(hardware, "_run", fake_tools({"lspci": out}))
    [gpu] = hardware._lspci()
    assert gpu.vendor == "intel" and "Arc A770" in gpu.name


def machine(gpus: list[hardware.Gpu], os_name: str = "Linux") -> hardware.Machine:  # noqa: D103
    shared = any(g.vendor == "apple" for g in gpus)
    return hardware.Machine(os_name, "x", "x86_64", "cpu", 8, 64 * 1024**3, None, gpus, shared)


G = 1024**3


@pytest.mark.parametrize(
    ("gpus", "os_name", "expected", "build"),
    [
        (
            [hardware.Gpu("nvidia", "RTX 4090", 24 * G, "580.1", arch="8.9")],
            "Windows",
            "cuda",
            "cu130",
        ),
        (
            [hardware.Gpu("nvidia", "RTX 3060", 12 * G, "560.9", arch="8.6")],
            "Linux",
            "cuda",
            "cu126",
        ),
        (
            [hardware.Gpu("nvidia", "GTX 1080", 8 * G, "580.1", arch="6.1")],
            "Linux",
            "cuda",
            "cu126",
        ),
        ([hardware.Gpu("nvidia", "GTX 970", 4 * G, "470.2", arch="5.2")], "Linux", "cpu", "cpu"),
        ([hardware.Gpu("amd", "Radeon RX 7900 XTX", 24 * G)], "Linux", "rocm", "rocm7.2"),
        ([hardware.Gpu("amd", "AMD Radeon RX 7900 XTX", 24 * G)], "Windows", "rocm", "amd-windows"),
        ([hardware.Gpu("amd", "AMD Radeon RX 6800", 16 * G)], "Windows", "cpu", "cpu"),
        (
            [hardware.Gpu("intel", "Intel(R) Arc(TM) A770 Graphics", 16 * G)],
            "Windows",
            "xpu",
            "xpu",
        ),
        ([hardware.Gpu("intel", "Intel(R) UHD Graphics 620", G)], "Windows", "cpu", "cpu"),
        ([hardware.Gpu("apple", "M4", 16 * G)], "Darwin", "mlx", "pypi"),
        ([], "Linux", "cpu", "cpu"),
    ],
)
def test_accelerator_and_torch_build(
    gpus: list[hardware.Gpu], os_name: str, expected: str, build: str
) -> None:
    arch = "arm64" if os_name == "Darwin" else "x86_64"
    found = hardware.Machine(os_name, "x", arch, "cpu", 8, 64 * G, None, gpus, os_name == "Darwin")
    assert found.accelerator == expected
    assert found.torch_build.backend == build


def test_amd_windows_uses_amds_index_for_the_right_target() -> None:
    gpus = [hardware.Gpu("amd", "AMD Radeon RX 9070 XT", 16 * G)]
    build = hardware.Machine("Windows", "x", "AMD64", "cpu", 8, 32 * G, None, gpus).torch_build
    assert build.index_url == hardware.AMD_WINDOWS_INDEX
    assert build.requirement == "torch[device-gfx1201]==2.13.0+rocm10.0.0"


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
