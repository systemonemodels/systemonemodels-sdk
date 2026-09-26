"""`systemone run studio`: fetch Laya Studio, keep it current, and start it.

Laya Studio is the local fine-tuning app that publishes to System One Models.
It is a separate open-source project, so the CLI does not depend on it: this
module clones it once (or downloads its source when git is missing), updates
it on each run, gives it its own Python environment, and runs it. The studio
opens in the browser and publishes through this CLI's login.

Layout, under the platform's data directory:

    <data>/systemone/studio/app        the Laya Studio checkout (replaceable)

Datasets, runs and checkpoints stay in the studio's own workspace,
~/.layastudio/workspace, so updating or deleting the code never touches them.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from systemone.config import APP_NAME

STUDIO_REPO = "https://github.com/biplovgautam/LayaStudio"
ENV_REPO = "SYSTEMONE_STUDIO_REPO"
ENV_DIR = "SYSTEMONE_STUDIO_DIR"
DEFAULT_WORKSPACE = Path("~/.layastudio/workspace")
# Laya Studio's own floor.
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


def supported_here() -> tuple[bool, str]:
    """Whether Laya Studio runs on this machine today, and why not."""
    system, machine = platform.system(), platform.machine().lower()
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        return True, ""
    where = {"Darwin": "an Intel Mac", "Windows": "Windows", "Linux": "Linux"}.get(system, system)
    return (
        False,
        f"Laya Studio runs on Apple silicon Macs today; {where} support, with NVIDIA and AMD "
        "GPUs, is being built.",
    )


# --- fetching ---------------------------------------------------------------


def _run(cmd: Sequence[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    # Argument lists of known executables (git, python, uv), never a shell string.
    return subprocess.run(list(cmd), cwd=cwd, capture_output=True, text=True, check=False)  # noqa: S603


def fetch(dest: Path, *, update: bool = True, log: Log = print) -> str:
    """Make `dest` a current Laya Studio checkout. Returns what happened.

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
                    f"{dest} exists and is not a Laya Studio checkout; move it away."
                )
            log(f"Cloning Laya Studio from {repo_url()}")
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
                "Could not update Laya Studio (offline, or local changes); using the copy you have."
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
            log(f"Could not download Laya Studio ({exc}); using the copy you have.")
            return "kept"
        raise


def _download(dest: Path, log: Log) -> str:
    url = f"{repo_url()}/archive/refs/heads/main.tar.gz"
    if not url.startswith("https://"):
        raise RuntimeError("The Laya Studio source must be fetched over https.")
    log(f"Downloading Laya Studio from {url}")
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
                raise RuntimeError(f"Unsafe path in the Laya Studio archive: {member.name}")
            target = (root / Path(*parts)).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe path in the Laya Studio archive: {member.name}")
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


def _python_for_venv() -> str | None:
    if sys.version_info[:2] >= MIN_PYTHON:
        return sys.executable
    for name in ("python3.13", "python3.12", "python3.11"):
        if found := shutil.which(name):
            return found
    return None


def _venv_bin(venv: Path, name: str) -> Path:
    return (
        venv
        / ("Scripts" if os.name == "nt" else "bin")
        / (f"{name}.exe" if os.name == "nt" else name)
    )


def prepare(app: Path, *, log: Log = print) -> Launch:
    """The command that runs the studio in its own environment, creating it if needed.

    uv, when it is installed, does everything: it picks or downloads a suitable
    Python and keeps the environment in step with the studio's lock file.
    Otherwise a virtual environment is made with the first Python 3.11+ found,
    and reinstalled only when the studio's dependencies change.
    """
    if uv := shutil.which("uv"):
        return Launch(
            [uv, "run", "--project", str(app), "--extra", "systemone", "layastudio"], "uv"
        )

    python = _python_for_venv()
    if python is None:
        raise RuntimeError(
            "Laya Studio needs Python 3.11 or newer. Install uv (https://docs.astral.sh/uv/), "
            "which fetches one for you, or install Python 3.11+, then run this again."
        )
    venv = app / ".venv"
    stamp = venv / ".systemone-deps"
    wanted = hashlib.sha256((app / "pyproject.toml").read_bytes()).hexdigest()
    if not _venv_bin(venv, "python").exists():
        log("Creating Laya Studio's Python environment")
        made = _run([python, "-m", "venv", str(venv)])
        if made.returncode != 0:
            raise RuntimeError(f"Could not create a virtual environment: {made.stderr.strip()}")
    if not stamp.exists() or stamp.read_text().strip() != wanted:
        log("Installing Laya Studio's dependencies (the first time takes a few minutes)")
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
            check=False,
        )
        if pip.returncode != 0:
            raise RuntimeError(
                "Installing Laya Studio's dependencies failed; see the output above."
            )
        stamp.write_text(wanted)
    return Launch([str(_venv_bin(venv, "layastudio"))], "venv")


def studio_args(
    *, port: int | None, browser: bool, workspace: Path | None, extra: Sequence[str]
) -> list[str]:
    args: list[str] = []
    if port:
        args += ["--port", str(port)]
    if not browser:
        args.append("--no-browser")
    if "--workspace" not in extra and not os.environ.get("LAYASTUDIO_HOME"):
        args += ["--workspace", str((workspace or DEFAULT_WORKSPACE).expanduser())]
    return args + list(extra)


def run(command: Sequence[str]) -> int:
    """Run the studio in the foreground; Ctrl+C reaches it, and its exit code is ours."""
    # The studio has its own environment. Whatever virtualenv this CLI was
    # installed into must not leak into it — uv would warn, pip would mix them.
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    process = subprocess.Popen(list(command), env=env)  # noqa: S603 - the studio's own launcher, no shell
    try:
        return process.wait()
    except KeyboardInterrupt:
        # The terminal sent the same Ctrl+C to the studio; give it time to stop its jobs.
        try:
            return process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.terminate()
            return process.wait()
