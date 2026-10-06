"""`systemone run studio`: fetch System One Studio, keep it current, and start it.

System One Studio (formerly LayaStudio) is the local fine-tuning app that
publishes to System One Models.
It is a separate open-source project, so the CLI does not depend on it: this
module clones it once (or downloads its source when git is missing), updates
it on each run, gives it its own Python environment, and runs it. The studio
opens in the browser and publishes through this CLI's login.

Its package and command are `systemone-studio`. Until October 2026 they were
`layastudio`, and a copy from before then (kept by --no-update or an update
that failed, or an old --source checkout) has only that command, so the
command comes from the checkout's own pyproject.toml (command_in()).

Layout, under the platform's data directory:

    <data>/systemone/studio/app        the System One Studio checkout (replaceable)

Datasets, runs and checkpoints stay in the studio's own workspace,
~/.layastudio/workspace (the folder keeps its old name), so updating or deleting
the code never touches them; $SYSTEMONE_STUDIO_HOME (formerly $LAYASTUDIO_HOME)
moves it.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from systemone import hardware
from systemone.config import APP_NAME

# The one place the studio's repository is named.
STUDIO_REPO = "https://github.com/biplovgautam/LayaStudio"
# This CLI's own variables. They share the studio's prefix; the studio reads no REPO or DIR.
ENV_REPO = "SYSTEMONE_STUDIO_REPO"
ENV_DIR = "SYSTEMONE_STUDIO_DIR"
COMMAND = "systemone-studio"
# The command's name before the rename; current copies keep it as an alias.
OLD_COMMAND = "layastudio"
# The studio's workspace variable, and its name before the rename. As in the studio, the
# new name wins whenever it is set, even to nothing.
ENV_HOME = "SYSTEMONE_STUDIO_HOME"
OLD_ENV_HOME = "LAYASTUDIO_HOME"
DEFAULT_WORKSPACE = Path("~/.layastudio/workspace")
# System One Studio's own floor.
MIN_PYTHON = (3, 11)

Log = Callable[[str], None]


def repo_url() -> str:
    return os.environ.get(ENV_REPO, STUDIO_REPO).rstrip("/")


def data_dir() -> Path:
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", "~")).expanduser() / APP_NAME
    return Path(os.environ.get("XDG_DATA_HOME", "~/.local/share")).expanduser() / APP_NAME


def app_dir() -> Path:
    if override := os.environ.get(ENV_DIR):
        return Path(override).expanduser()
    return data_dir() / "studio" / "app"


def supported_here(machine: hardware.Machine | None = None) -> tuple[bool, str]:
    """Whether System One Studio can train on this machine, and why not.

    Apple silicon trains on MLX; Windows and Linux on PyTorch — NVIDIA, AMD, Intel Arc or
    the CPU. Only Intel Macs are left out: neither current PyTorch nor MLX builds for them.
    """
    build = (machine or hardware.detect()).torch_build
    if build.backend == "unsupported":
        return False, build.reason
    return True, ""


# --- fetching ---------------------------------------------------------------


def _run(cmd: Sequence[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    # Argument lists of known executables (git, python, uv), never a shell string.
    return subprocess.run(list(cmd), cwd=cwd, capture_output=True, text=True, check=False)  # noqa: S603


def fetch(dest: Path, *, update: bool = True, log: Log = print) -> str:
    """Make `dest` a current System One Studio checkout. Returns what happened.

    With git: clone once, fast-forward after. Without git: download the source
    archive. An update that fails — offline, local edits — keeps the copy that
    is there, so a studio that worked yesterday still starts today.
    """
    git = shutil.which("git")
    exists = (dest / "pyproject.toml").exists()
    if git:
        if not exists:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and any(dest.iterdir()):
                raise RuntimeError(
                    f"{dest} exists and is not a System One Studio checkout; move it away."
                )
            log(f"Cloning System One Studio from {repo_url()}")
            done = _run([git, "clone", "--depth", "1", repo_url(), str(dest)])
            if done.returncode != 0:
                raise RuntimeError(
                    f"git clone failed: {done.stderr.strip() or done.stdout.strip()}"
                )
            return "installed"
        if not update:
            return "kept"
        if not (dest / ".git").exists():
            return _download(dest, log) if update else "kept"
        before = _run([git, "rev-parse", "HEAD"], cwd=dest).stdout.strip()
        pulled = _run([git, "pull", "--ff-only", "--quiet"], cwd=dest)
        if pulled.returncode != 0:
            log(
                "Could not update System One Studio (offline, or local changes); "
                "using the copy you have."
            )
            return "kept"
        after = _run([git, "rev-parse", "HEAD"], cwd=dest).stdout.strip()
        return "updated" if after != before else "current"
    if exists and not update:
        return "kept"
    try:
        return _download(dest, log)
    except OSError as exc:
        if exists:
            log(f"Could not download System One Studio ({exc}); using the copy you have.")
            return "kept"
        raise


def _download(dest: Path, log: Log) -> str:
    url = f"{repo_url()}/archive/refs/heads/main.tar.gz"
    if not url.startswith("https://"):
        raise RuntimeError("The System One Studio source must be fetched over https.")
    log(f"Downloading System One Studio from {url}")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "studio.tar.gz"
        with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as out:  # noqa: S310 - https checked above
            shutil.copyfileobj(response, out)
        extract(archive, dest)
    return "installed"


def extract(archive: Path, dest: Path) -> int:
    """Unpack a GitHub source archive into `dest`, dropping its top folder.

    Every member is checked: regular files and folders only, no absolute
    paths, nothing that climbs out of `dest`. Files already in `dest` are
    overwritten; files that are not in the archive — a .venv — are left alone.
    """
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    written = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            parts = PurePosixPath(member.name).parts[1:]
            if not parts:
                continue
            if member.name.startswith("/") or ".." in parts:
                raise RuntimeError(f"Unsafe path in the System One Studio archive: {member.name}")
            target = (root / Path(*parts)).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe path in the System One Studio archive: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    continue
                with source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)
                if member.mode & 0o111:
                    target.chmod(0o755)
                written += 1
            # Links, devices and the rest are skipped: a source tree needs none.
    return written


# --- running ----------------------------------------------------------------


@dataclass
class Launch:
    command: list[str]
    how: str


def _works(python: str) -> bool:
    """Whether an interpreter actually starts: a pyenv shim for a version that is
    not installed is on PATH but fails the moment it runs."""
    done = _run(
        [python, "-c", "import sys; print(sys.version_info[:2] >= (3, 11))"], cwd=Path.home()
    )
    return done.returncode == 0 and done.stdout.strip() == "True"


def _python_for_venv() -> str | None:
    if sys.version_info[:2] >= MIN_PYTHON:
        return sys.executable
    for name in ("python3.13", "python3.12", "python3.11"):
        if (found := shutil.which(name)) and _works(found):
            return found
    return None


def _venv_bin(venv: Path, name: str) -> Path:
    return (
        venv
        / ("Scripts" if os.name == "nt" else "bin")
        / (f"{name}.exe" if os.name == "nt" else name)
    )


def _scripts(pyproject: Path) -> set[str]:
    """The commands a pyproject.toml's [project.scripts] table declares.

    A line scan rather than a TOML parser: Python 3.10 has none, and the studio's file is
    plain (one `name = "module:function"` per line).
    """
    try:
        lines = pyproject.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return set()
    names: set[str] = set()
    inside = False
    for line in lines:
        text = line.split("#", 1)[0].strip()
        if text.startswith("["):
            inside = text.replace(" ", "") == "[project.scripts]"
        elif inside and "=" in text:
            names.add(text.split("=", 1)[0].strip().strip("\"'"))
    return names


def command_in(app: Path) -> str:
    """The studio's command in the checkout at `app`: systemone-studio, or layastudio in a
    copy from before the rename, which has nothing else."""
    scripts = _scripts(app / "pyproject.toml")
    return OLD_COMMAND if OLD_COMMAND in scripts and COMMAND not in scripts else COMMAND


def prepare(app: Path, machine: hardware.Machine | None = None, *, log: Log = print) -> Launch:
    """The command that runs the studio in its own environment, creating it if needed.

    On Apple silicon, uv runs the studio against its lock file (MLX). Everywhere else the
    studio gets its own virtual environment with the PyTorch build this machine needs —
    CUDA 13 or 12.6, ROCm, Intel XPU or CPU — from PyTorch's (or AMD's) index, reinstalled
    only when the build or the studio's dependencies change. Without uv, a virtual
    environment is made with the first Python 3.11+ found.
    """
    build = (machine or hardware.detect()).torch_build
    if build.backend == "unsupported":
        raise RuntimeError(build.reason)
    uv = shutil.which("uv")
    name = command_in(app)
    if build.backend != "pypi":
        return _prepare_torch(app, build, uv, name, log)
    if uv:
        return Launch([uv, "run", "--project", str(app), "--extra", "systemone", name], "uv")

    python = _python_for_venv()
    if python is None:
        raise RuntimeError(
            "System One Studio needs Python 3.11 or newer. Install uv "
            "(https://docs.astral.sh/uv/), which fetches one for you, or install Python 3.11+, "
            "then run this again."
        )
    venv = app / ".venv"
    stamp = venv / ".systemone-deps"
    wanted = hashlib.sha256((app / "pyproject.toml").read_bytes()).hexdigest()
    if not _venv_bin(venv, "python").exists():
        log("Creating System One Studio's Python environment")
        made = _run([python, "-m", "venv", str(venv)], cwd=app)
        if made.returncode != 0:
            raise RuntimeError(f"Could not create a virtual environment: {made.stderr.strip()}")
    if not stamp.exists() or stamp.read_text().strip() != wanted:
        log("Installing System One Studio's dependencies (the first time takes a few minutes)")
        pip = subprocess.run(  # noqa: S603 - the environment's own python, no shell
            [
                str(_venv_bin(venv, "python")),
                "-m",
                "pip",
                "install",
                "--quiet",
                "-e",
                f"{app}[systemone]",
            ],
            cwd=app,
            check=False,
        )
        if pip.returncode != 0:
            raise RuntimeError(
                "Installing System One Studio's dependencies failed; see the output above."
            )
        stamp.write_text(wanted)
    return Launch([str(_venv_bin(venv, name))], "venv")


def _prepare_torch(
    app: Path, build: hardware.TorchBuild, uv: str | None, name: str, log: Log
) -> Launch:
    venv = app / ".venv"
    python = _venv_bin(venv, "python")
    stamp = venv / ".systemone-deps"
    wanted = hashlib.sha256(
        (app / "pyproject.toml").read_bytes() + f"{build.backend}|{build.requirement}".encode()
    ).hexdigest()
    log(f"PyTorch for this machine: {build.reason}")
    if not python.exists():
        log("Creating System One Studio's Python environment")
        if uv:
            made = _run([uv, "venv", "--python", "3.12", str(venv)], cwd=app)
        else:
            base = _python_for_venv()
            if base is None:
                raise RuntimeError(
                    "System One Studio needs Python 3.11 or newer. Install uv "
                    "(https://docs.astral.sh/uv/), which fetches one for you, or install "
                    "Python 3.11+, then run this again."
                )
            made = _run([base, "-m", "venv", str(venv)], cwd=app)
        if made.returncode != 0:
            raise RuntimeError(f"Could not create a virtual environment: {made.stderr.strip()}")
    launch = Launch([str(_venv_bin(venv, name))], f"PyTorch {build.backend}")
    if stamp.exists() and stamp.read_text().strip() == wanted:
        return launch

    log("Installing PyTorch and System One Studio (the first time downloads a few GB)")
    target = f"{app}[torch,systemone]"
    if uv:
        pip = [uv, "pip", "install", "--python", str(python)]
        if build.backend == "amd-windows":
            steps = [
                [
                    *pip,
                    "--index-url",
                    str(build.index_url),
                    "--index-strategy",
                    "unsafe-best-match",
                    build.requirement,
                ],
                [*pip, "-e", target],
            ]
        else:
            steps = [[*pip, "--torch-backend", build.backend, "-e", target]]
    else:
        pip = [str(python), "-m", "pip", "install"]
        steps = [
            [*pip, "--index-url", str(build.index_url), build.requirement or "torch"],
            [*pip, "-e", target],
        ]
    for step in steps:
        done = subprocess.run(step, cwd=app, check=False)  # noqa: S603 - installers, no shell
        if done.returncode != 0:
            raise RuntimeError(
                "Installing System One Studio's dependencies failed; see the output above."
            )
    stamp.write_text(wanted)
    return launch


def workspace_from_environment(environ: Mapping[str, str] | None = None) -> str | None:
    """The workspace the studio's variable names: $SYSTEMONE_STUDIO_HOME when it is set,
    else $LAYASTUDIO_HOME — the studio's own rule."""
    environ = os.environ if environ is None else environ
    if ENV_HOME in environ:
        return environ[ENV_HOME]
    return environ.get(OLD_ENV_HOME)


def studio_args(
    *,
    port: int | None,
    browser: bool,
    workspace: Path | None,
    extra: Sequence[str],
    managed: bool = True,
) -> list[str]:
    args: list[str] = []
    if port:
        args += ["--port", str(port)]
    if not browser:
        args.append("--no-browser")
    # A checkout run with --source keeps its own workspace beside the code, where its
    # runs already are; the managed copy keeps data outside the code it replaces.
    if workspace is not None:
        args += ["--workspace", str(workspace.expanduser().resolve())]
    elif managed and "--workspace" not in extra and not workspace_from_environment():
        # Absolute: the studio runs from its own folder, not the caller's.
        args += ["--workspace", str(DEFAULT_WORKSPACE.expanduser().resolve())]
    return args + list(extra)


def child_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment the studio runs in."""
    environ = os.environ if environ is None else environ
    # The studio has its own environment. Whatever virtualenv this CLI was
    # installed into must not leak into it — uv would warn, pip would mix them.
    env = {k: v for k, v in environ.items() if k != "VIRTUAL_ENV"}
    # A copy from before the rename reads only the old name: hand it the workspace a
    # current copy would use.
    if ENV_HOME in env:
        env[OLD_ENV_HOME] = env[ENV_HOME]
    return env


def run(command: Sequence[str], cwd: Path | None = None) -> int:
    """Run the studio in the foreground; Ctrl+C reaches it, and its exit code is ours."""
    # From the studio's own folder: a .python-version wherever the user happens
    # to be would otherwise steer pyenv shims and uv to a different Python.
    env = child_environment()
    process = subprocess.Popen(list(command), env=env, cwd=cwd)  # noqa: S603 - no shell
    try:
        return process.wait()
    except KeyboardInterrupt:
        # The terminal sent the same Ctrl+C to the studio; give it time to stop its jobs.
        try:
            return process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.terminate()
            return process.wait()
