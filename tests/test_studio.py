"""`systemone run studio`: fetch System One Studio safely, run it in its own environment."""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from systemone import cli, hardware, studio

runner = CliRunner()

# The studio's pyproject.toml after the rename (October 2026), and before it.
NEW_PYPROJECT = """[project]
name = "systemone-studio"
keywords = ["system-one", "systemone-studio", "laya"]

[project.scripts]
systemone-studio = "systemone_studio.server:main"
# The command's name before the rename, kept as an alias: the same program.
layastudio = "systemone_studio.server:main"

[project.urls]
Repository = "https://github.com/biplovgautam/LayaStudio"
"""
OLD_PYPROJECT = """[project]
name = "layastudio"

[project.scripts]
layastudio = "layastudio.server:main"

[project.urls]
Repository = "https://github.com/biplovgautam/LayaStudio"
"""


def plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def make_archive(
    path: Path, members: dict[str, bytes], links: dict[str, str] | None = None
) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tar.addfile(info)
    return path


def test_extract_drops_the_top_folder_and_keeps_local_files(tmp_path: Path) -> None:
    archive = make_archive(
        tmp_path / "a.tar.gz",
        {
            "LayaStudio-main/pyproject.toml": b"[project]\nname='layastudio'\n",
            "LayaStudio-main/layastudio/__init__.py": b"",
        },
    )
    dest = tmp_path / "app"
    (dest / ".venv").mkdir(parents=True)
    (dest / ".venv" / "keep").write_text("mine")
    assert studio.extract(archive, dest) == 2
    assert (dest / "pyproject.toml").exists()
    assert (dest / "layastudio" / "__init__.py").exists()
    assert (dest / ".venv" / "keep").read_text() == "mine"


def test_extract_refuses_paths_that_climb_out(tmp_path: Path) -> None:
    archive = make_archive(tmp_path / "a.tar.gz", {"LayaStudio-main/../../evil.txt": b"x"})
    with pytest.raises(RuntimeError, match="Unsafe path"):
        studio.extract(archive, tmp_path / "app")
    assert not (tmp_path / "evil.txt").exists()


def test_extract_skips_links(tmp_path: Path) -> None:
    archive = make_archive(
        tmp_path / "a.tar.gz",
        {"LayaStudio-main/README.md": b"hi"},
        links={"LayaStudio-main/passwd": "/etc/passwd"},
    )
    dest = tmp_path / "app"
    studio.extract(archive, dest)
    assert (dest / "README.md").exists()
    assert not (dest / "passwd").exists()


def test_studio_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LAYASTUDIO_HOME", raising=False)
    monkeypatch.delenv("SYSTEMONE_STUDIO_HOME", raising=False)
    args = studio.studio_args(port=9000, browser=False, workspace=None, extra=["--model", "x/y"])
    assert args[:3] == ["--port", "9000", "--no-browser"]
    assert args[3] == "--workspace" and args[4].endswith(".layastudio/workspace")
    assert args[-2:] == ["--model", "x/y"]
    # A workspace passed through to the studio wins over the default one.
    passed = studio.studio_args(port=None, browser=True, workspace=None, extra=["--workspace", "w"])
    assert passed == ["--workspace", "w"]
    monkeypatch.setenv("LAYASTUDIO_HOME", "/somewhere")
    assert "--workspace" not in studio.studio_args(
        port=None, browser=True, workspace=None, extra=[]
    )


def test_the_workspace_variable_under_its_new_name_and_its_old_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert studio.workspace_from_environment({}) is None
    assert studio.workspace_from_environment({"LAYASTUDIO_HOME": "/old"}) == "/old"
    both = {"SYSTEMONE_STUDIO_HOME": "/new", "LAYASTUDIO_HOME": "/old"}
    assert studio.workspace_from_environment(both) == "/new"
    # As in the studio, the new name wins even when it is set to nothing.
    emptied = {"SYSTEMONE_STUDIO_HOME": "", "LAYASTUDIO_HOME": "/old"}
    assert studio.workspace_from_environment(emptied) == ""

    monkeypatch.delenv("LAYASTUDIO_HOME", raising=False)
    monkeypatch.setenv("SYSTEMONE_STUDIO_HOME", "/somewhere")
    assert "--workspace" not in studio.studio_args(
        port=None, browser=True, workspace=None, extra=[]
    )


def test_the_studio_gets_its_workspace_under_both_names_and_not_this_clis_venv() -> None:
    env = studio.child_environment(
        {
            "PATH": "/bin",
            "VIRTUAL_ENV": "/cli/.venv",
            "SYSTEMONE_STUDIO_HOME": "/new",
            "LAYASTUDIO_HOME": "/old",
        }
    )
    # A copy from before the rename reads only LAYASTUDIO_HOME: it gets the new value.
    assert env == {"PATH": "/bin", "SYSTEMONE_STUDIO_HOME": "/new", "LAYASTUDIO_HOME": "/new"}
    assert studio.child_environment({"LAYASTUDIO_HOME": "/old"}) == {"LAYASTUDIO_HOME": "/old"}


def test_the_command_comes_from_the_checkout(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    assert studio.command_in(tmp_path) == "systemone-studio"
    pyproject.write_text(NEW_PYPROJECT)
    assert studio.command_in(tmp_path) == "systemone-studio"
    # A copy from before the rename, kept by --no-update or an update that failed.
    pyproject.write_text(OLD_PYPROJECT)
    assert studio.command_in(tmp_path) == "layastudio"
    # The name outside [project.scripts] is not a command.
    pyproject.write_text(
        OLD_PYPROJECT.replace('name = "layastudio"', 'keywords = ["systemone-studio"]')
        + '\n[tool.other]\nsystemone-studio = "x"\n'
    )
    assert studio.command_in(tmp_path) == "layastudio"


def test_prepare_prefers_uv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(NEW_PYPROJECT)
    monkeypatch.setattr(shutil, "which", lambda name: "/opt/uv" if name == "uv" else None)
    launch = studio.prepare(tmp_path, apple_silicon())
    assert launch.how == "uv"
    assert launch.command == [
        "/opt/uv",
        "run",
        "--project",
        str(tmp_path),
        "--extra",
        "systemone",
        "systemone-studio",
    ]


@pytest.mark.parametrize(
    ("pyproject", "name"),
    [(NEW_PYPROJECT, "systemone-studio"), (OLD_PYPROJECT, "layastudio")],
    ids=["current", "before-the-rename"],
)
def test_every_way_of_running_runs_the_checkouts_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pyproject: str, name: str
) -> None:
    (tmp_path / "pyproject.toml").write_text(pyproject)
    venv_command = [str(studio._venv_bin(tmp_path / ".venv", name))]

    def fake_run(cmd, cwd=None):  # type: ignore[no-untyped-def]
        venv_python = studio._venv_bin(tmp_path / ".venv", "python")
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.touch()
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def fake_install(
        cmd: list[str], cwd: Path | None = None, check: bool = False
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(studio, "_run", fake_run)
    monkeypatch.setattr(subprocess, "run", fake_install)

    monkeypatch.setattr(shutil, "which", lambda tool: "/opt/uv" if tool == "uv" else None)
    assert studio.prepare(tmp_path, apple_silicon()).command[-1] == name
    assert studio.prepare(tmp_path, fake_machine(), log=lambda _: None).command == venv_command
    # No uv: a virtual environment of its own.
    monkeypatch.setattr(shutil, "which", lambda tool: None)
    monkeypatch.setattr(studio, "_python_for_venv", lambda: sys.executable)
    assert studio.prepare(tmp_path, apple_silicon(), log=lambda _: None).command == venv_command


def test_prepare_without_a_new_enough_python_says_what_to_do(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(studio, "_python_for_venv", lambda: None)
    with pytest.raises(RuntimeError, match="Python 3.11"):
        studio.prepare(tmp_path, apple_silicon())


def apple_silicon() -> hardware.Machine:
    return hardware.Machine(
        "Darwin",
        "macOS 26",
        "arm64",
        "Apple M4",
        10,
        16 * 1024**3,
        None,
        [hardware.Gpu("apple", "Apple M4", 16 * 1024**3)],
        True,
    )


def test_only_intel_macs_are_left_out() -> None:
    intel_mac = hardware.Machine("Darwin", "macOS 14", "x86_64", "Intel", 8, 16 * 1024**3, None, [])
    ok, why = studio.supported_here(intel_mac)
    assert not ok and "Intel Macs" in why
    assert studio.supported_here(apple_silicon()) == (True, "")
    windows_cpu = hardware.Machine("Windows", "11", "AMD64", "x", 8, 16 * 1024**3, None, [])
    assert studio.supported_here(windows_cpu) == (True, "")


def test_torch_machines_get_their_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='layastudio'\n")
    calls: list[list[str]] = []
    monkeypatch.setattr(shutil, "which", lambda name: "/opt/uv" if name == "uv" else None)

    def fake_run(cmd, cwd=None):  # type: ignore[no-untyped-def]
        calls.append(list(cmd))
        venv_python = studio._venv_bin(tmp_path / ".venv", "python")
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.touch()
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(studio, "_run", fake_run)

    def fake_install(
        cmd: list[str], cwd: Path | None = None, check: bool = False
    ) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_install)
    rtx = hardware.Machine(
        "Linux",
        "Ubuntu",
        "x86_64",
        "x",
        16,
        64 * 1024**3,
        None,
        [hardware.Gpu("nvidia", "RTX 4090", 24 * 1024**3, "580.1", arch="8.9")],
    )
    launch = studio.prepare(tmp_path, rtx, log=lambda _: None)
    assert launch.how == "PyTorch cu130"
    assert ["/opt/uv", "venv", "--python", "3.12", str(tmp_path / ".venv")] in calls
    install = next(c for c in calls if "pip" in c)
    assert install[install.index("--torch-backend") + 1] == "cu130"
    assert install[-1].endswith("[torch,systemone]")
    calls.clear()
    studio.prepare(tmp_path, rtx, log=lambda _: None)
    assert not any("pip" in c for c in calls)  # nothing changed: no reinstall


def fake_machine() -> hardware.Machine:
    return hardware.Machine(
        os="Linux",
        os_version="Ubuntu 24.04",
        arch="x86_64",
        cpu="Test CPU",
        cores=8,
        memory_bytes=32 * 1024**3,
        disk_free_bytes=100 * 1024**3,
        gpus=[hardware.Gpu("nvidia", "RTX 4090", 24 * 1024**3, "560.1", "cuda 12.6")],
    )


def test_run_studio_refuses_unsupported_machines_without_force(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(studio, "supported_here", lambda machine=None: (False, "Not here yet."))
    monkeypatch.setattr(hardware, "detect", lambda *_: fake_machine())
    result = runner.invoke(cli.app, ["run", "studio", "--no-sign-in"])
    assert result.exit_code == 1
    assert "Not here yet." in plain(result.output) and "--force" in plain(result.output)


def test_run_studio_fetches_prepares_and_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ran: list[list[str]] = []
    monkeypatch.setattr(studio, "supported_here", lambda machine=None: (True, ""))
    monkeypatch.setattr(studio, "app_dir", lambda: tmp_path / "app")
    monkeypatch.setattr(studio, "fetch", lambda dest, update, log: "current")
    monkeypatch.setattr(
        studio, "prepare", lambda app, machine, log: studio.Launch(["systemone-studio"], "uv")
    )
    monkeypatch.setattr(hardware, "detect", lambda *_: fake_machine())

    def fake_run(command: list[str], cwd: Path | None = None) -> int:
        ran.append(list(command))
        return 0

    monkeypatch.setattr(studio, "run", fake_run)
    monkeypatch.setenv("SYSTEMONE_STUDIO_HOME", str(tmp_path / "ws"))
    result = runner.invoke(
        cli.app,
        ["run", "studio", "--no-sign-in", "--no-browser", "--", "--model", "aac6fef/laya-mlx"],
    )
    assert result.exit_code == 0, result.output
    assert "System One Studio up to date" in plain(result.output)
    assert "RTX 4090" in plain(result.output) and "NVIDIA CUDA" in plain(result.output)
    assert ran == [["systemone-studio", "--no-browser", "--model", "aac6fef/laya-mlx"]]


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_fetch_clones_then_updates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "LayaStudio"
    source.mkdir()

    def git(*args: str, cwd: Path = source) -> None:
        subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)  # noqa: S603, S607 - test fixture

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (source / "pyproject.toml").write_text("[project]\nname='layastudio'\n")
    git("add", ".")
    git("commit", "-q", "-m", "one")
    monkeypatch.setenv(studio.ENV_REPO, str(source))

    dest = tmp_path / "app"
    assert studio.fetch(dest, log=lambda _: None) == "installed"
    assert studio.fetch(dest, log=lambda _: None) == "current"
    (source / "NEW").write_text("x")
    git("add", ".")
    git("commit", "-q", "-m", "two")
    assert studio.fetch(dest, log=lambda _: None) == "updated"
    assert (dest / "NEW").exists()
    assert studio.fetch(dest, update=False, log=lambda _: None) == "kept"
