"""`systemone run studio`: fetch Laya Studio safely, run it in its own environment."""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from systemone import cli, studio

runner = CliRunner()


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


def test_prepare_prefers_uv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(studio.shutil, "which", lambda name: "/opt/uv" if name == "uv" else None)
    launch = studio.prepare(tmp_path)
    assert launch.how == "uv"
    assert launch.command == [
        "/opt/uv",
        "run",
        "--project",
        str(tmp_path),
        "--extra",
        "systemone",
        "layastudio",
    ]


def test_prepare_without_a_new_enough_python_says_what_to_do(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(studio.shutil, "which", lambda name: None)
    monkeypatch.setattr(studio, "_python_for_venv", lambda: None)
    with pytest.raises(RuntimeError, match="Python 3.11"):
        studio.prepare(tmp_path)


def test_platform_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")
    monkeypatch.setattr(studio.platform, "machine", lambda: "x86_64")
    ok, why = studio.supported_here()
    assert not ok and "Linux" in why
    monkeypatch.setattr(studio.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(studio.platform, "machine", lambda: "arm64")
    assert studio.supported_here() == (True, "")


def test_run_studio_refuses_unsupported_machines_without_force(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(studio, "supported_here", lambda: (False, "Not here yet."))
    result = runner.invoke(cli.app, ["run", "studio", "--no-sign-in"])
    assert result.exit_code == 1
    assert "Not here yet." in plain(result.output) and "--force" in plain(result.output)


def test_run_studio_fetches_prepares_and_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ran: list[list[str]] = []
    monkeypatch.setattr(studio, "supported_here", lambda: (True, ""))
    monkeypatch.setattr(studio, "app_dir", lambda: tmp_path / "app")
    monkeypatch.setattr(studio, "fetch", lambda dest, update, log: "current")
    monkeypatch.setattr(studio, "prepare", lambda app, log: studio.Launch(["layastudio"], "uv"))
    monkeypatch.setattr(studio, "run", lambda command: ran.append(list(command)) or 0)
    monkeypatch.setenv("LAYASTUDIO_HOME", str(tmp_path / "ws"))
    result = runner.invoke(
        cli.app,
        ["run", "studio", "--no-sign-in", "--no-browser", "--", "--model", "aac6fef/laya-mlx"],
    )
    assert result.exit_code == 0, result.output
    assert "Laya Studio up to date" in plain(result.output)
    assert ran == [["layastudio", "--no-browser", "--model", "aac6fef/laya-mlx"]]


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
