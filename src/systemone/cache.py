"""Local cache, laid out the way Hugging Face's is and for the same reasons.

    ~/.cache/systemone/
      blobs/ab/abcdef…                          one file per distinct content
      models/biplov--snake/
        snapshots/0.3.0/coreml-int4/model.onnx  → link to a blob
        refs/latest                             "0.3.0"

Blobs are keyed by SHA-256 and shared by every repository and every version, so
the tokenizer that six exports of one model have in common is downloaded and
stored once. A snapshot is a directory of links into the blobs — the tree a
version describes, costing no extra disk.

Deterministic in the sense that matters: a version is immutable on the server,
so a snapshot for it is built once and never changes. A second pull of the same
version touches the network only to read the version's file list.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
from pathlib import Path

from systemone.config import cache_dir

CHUNK = 1024 * 1024


def blob_path(sha256: str) -> Path:
    return cache_dir() / "blobs" / sha256[:2] / sha256


def snapshot_root(repo: str, version: str) -> Path:
    namespace, _, name = repo.partition("/")
    return cache_dir() / "models" / f"{namespace}--{name}" / "snapshots" / version


def ref_path(repo: str, ref: str = "latest") -> Path:
    namespace, _, name = repo.partition("/")
    return cache_dir() / "models" / f"{namespace}--{name}" / "refs" / ref


def verify(path: Path, sha256: str) -> bool:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(CHUNK):
                digest.update(chunk)
    except OSError:
        return False
    return digest.hexdigest() == sha256


def has_blob(sha256: str) -> bool:
    """Present and intact.

    Checked by content rather than existence, because a blob that was
    truncated by a crash, or edited by someone poking around the cache, must be
    fetched again rather than silently handed out.
    """
    path = blob_path(sha256)
    return path.exists() and verify(path, sha256)


def adopt(downloaded: Path, sha256: str) -> Path:
    """Move a finished download into the blob store."""
    target = blob_path(sha256)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(downloaded, target)
    # Read-only, so a snapshot link cannot be used to modify a blob that other
    # snapshots share.
    with contextlib.suppress(OSError):
        target.chmod(0o444)
    return target


def link(blob: Path, target: Path) -> None:
    """Materialise a blob at a path inside a snapshot or a destination.

    Hard link first: no extra disk, and the file behaves as a real file to
    every tool that opens it. Symlink if the destination is on another volume.
    Copy as a last resort, which is what Windows without developer mode gets.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.link(blob, target)
        return
    except OSError:
        pass
    try:
        target.symlink_to(blob)
        return
    except OSError:
        pass
    shutil.copyfile(blob, target)


def write_ref(repo: str, version: str, ref: str = "latest") -> None:
    path = ref_path(repo, ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(version)


def read_ref(repo: str, ref: str = "latest") -> str | None:
    path = ref_path(repo, ref)
    return path.read_text().strip() if path.exists() else None


def size() -> int:
    root = cache_dir() / "blobs"
    if not root.exists():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def clear() -> int:
    """Delete the whole cache. Returns bytes freed."""
    freed = size()
    root = cache_dir()
    if root.exists():
        # Blobs are read-only by design; make them writable so rmtree can go.
        for path in root.rglob("*"):
            with contextlib.suppress(OSError):
                path.chmod(0o644 if path.is_file() else 0o755)
        shutil.rmtree(root)
    return freed
