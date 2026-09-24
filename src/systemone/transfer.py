"""Push and pull a model directory.

Kept separate from the CLI so the same logic is importable:

    from systemone import Client
    from systemone.transfer import push, pull
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from systemone import cache
from systemone.client import CHUNK, Client, sha256_file
from systemone.errors import SystemOneError
from systemone.inspect import walk

Progress = Callable[[str, int, int, bool], None]
"""(path, bytes done, bytes total, deduplicated)"""


def _read(
    path: Path, offset: int, length: int, on_bytes: Callable[[int], None] | None
) -> Iterator[bytes]:
    """A byte range of a file, a chunk at a time, reporting each as it goes."""
    with path.open("rb") as handle:
        handle.seek(offset)
        remaining = length
        while remaining > 0:
            chunk = handle.read(min(CHUNK, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            if on_bytes:
                on_bytes(len(chunk))
            yield chunk


def push_files(
    client: Client,
    repo: str,
    source: Path,
    files: Iterable[Path],
    prefix: str = "",
    on_progress: Progress | None = None,
    *,
    on_file: Callable[[str], None] | None = None,
    on_bytes: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
    """Upload each file and return the artifact records for a version.

    `on_file` hears each path as work on it starts, and `on_bytes` each chunk
    as it leaves, so a progress bar can move during a large file rather than
    only between files.
    """
    artifacts: list[dict[str, Any]] = []

    for path in files:
        relative = path.relative_to(source).as_posix()
        placed = f"{prefix.strip('/')}/{relative}" if prefix.strip("/") else relative
        size = path.stat().st_size
        if on_file:
            on_file(placed)

        # Hashed before anything is sent, so a file the registry already holds
        # is recognised and skipped. Streamed, so a 20GB checkpoint is not read
        # into memory to be hashed.
        digest = sha256_file(path)
        ticket = client.start_upload(repo, placed, size, digest)
        deduplicated = bool(ticket.get("already_present"))

        try:
            if deduplicated:
                result = client.finish_upload(repo, ticket["upload_id"], sha256=digest)
                if on_bytes:
                    on_bytes(size)
            elif ticket.get("url"):
                client.put(ticket["url"], _read(path, 0, size, on_bytes), size)
                result = client.finish_upload(repo, ticket["upload_id"], sha256=digest)
            else:
                part_size = ticket["part_size"]
                parts = []
                for part in ticket["parts"]:
                    offset = (part["part_number"] - 1) * part_size
                    length = max(0, min(part_size, size - offset))
                    etag = client.put(part["url"], _read(path, offset, length, on_bytes), length)
                    parts.append({"part_number": part["part_number"], "etag": etag})
                result = client.finish_upload(repo, ticket["upload_id"], parts=parts, sha256=digest)
        except Exception:
            # Release the reservation rather than leaving an abandoned multipart
            # upload for the bucket's lifecycle rule to find.
            with contextlib.suppress(Exception):
                # Best effort. The original failure is what the caller needs
                # to hear about, not a failed cleanup of it.
                client.abort_upload(repo, ticket["upload_id"])
            raise

        artifacts.append(
            {
                "kind": "r2",
                "uri": result["uri"],
                "path": result["path"],
                "filename": result["filename"],
                "size_bytes": result["size_bytes"],
                "sha256": digest,
                "storage_key": result["storage_key"],
            }
        )

        if on_progress:
            on_progress(placed, size, size, deduplicated)

    return artifacts


@dataclass
class PullResult:
    root: Path
    downloaded: int = 0
    from_cache: int = 0
    files: int = 0


def pull_version(
    client: Client,
    repo: str,
    destination: Path | None = None,
    version: str | None = None,
    variant: str | None = None,
    on_progress: Callable[[str, int, int, bool], None] | None = None,
) -> PullResult:
    """Materialise a version on disk, through the local cache.

    Every file with a known digest is served from the blob cache when it is
    already there and intact, and downloaded into the cache when it is not.
    The snapshot for the version is then a tree of links into that cache. With
    no destination, the snapshot itself is the result — the same shape as
    Hugging Face's snapshot_download. With one, the tree is linked there too.

    Callback receives (path, size, total files, came from cache).
    """
    detail = client.model(repo)
    wanted = version or detail.get("latest_version")
    if not wanted:
        raise SystemOneError(f"{repo} has no published version")

    versions = client.versions(repo)
    selected = next((v for v in versions if v["version"] == wanted), None)
    if selected is None:
        available = ", ".join(v["version"] for v in versions) or "none"
        raise SystemOneError(f"{repo} has no version {wanted}. Available: {available}")

    artifacts = selected["artifacts"]
    if variant:
        prefix = variant.strip("/") + "/"
        artifacts = [a for a in artifacts if (a.get("path") or "").startswith(prefix)]
        if not artifacts:
            folders = sorted(
                {
                    (a.get("path") or "").split("/")[0]
                    for a in selected["artifacts"]
                    if a.get("path")
                }
            )
            raise SystemOneError(
                f"no files under '{variant}'. Available: {', '.join(folders) or 'none'}"
            )

    snapshot = cache.snapshot_root(repo, wanted)
    result = PullResult(root=snapshot, files=len(artifacts))

    for artifact in artifacts:
        relative = artifact.get("path") or artifact["filename"]
        digest = artifact.get("sha256")
        size = artifact.get("size_bytes") or 0

        if digest and cache.has_blob(digest):
            blob = cache.blob_path(digest)
            result.from_cache += size
            cached = True
        elif digest:
            staging = cache.blob_path(digest).with_suffix(".incoming")
            for chunk in client.download(artifact["uri"], staging, digest):
                result.downloaded += chunk
            blob = cache.adopt(staging, digest)
            cached = False
        else:
            # No digest means nothing to key the cache on — an external
            # reference recorded without one. Fetched straight into the
            # snapshot, every time.
            target = snapshot / relative
            for chunk in client.download(artifact["uri"], target):
                result.downloaded += chunk
            if on_progress:
                on_progress(relative, size, len(artifacts), False)
            continue

        cache.link(blob, snapshot / relative)
        if on_progress:
            on_progress(relative, size, len(artifacts), cached)

    if manifest := detail.get("manifest_yaml"):
        (snapshot / "systemone.yaml").write_text(manifest)
    cache.write_ref(repo, wanted)

    if destination is not None:
        root = destination.expanduser().resolve() / repo.split("/")[-1]
        for source in snapshot.rglob("*"):
            if source.is_file() or source.is_symlink():
                relative_path = source.relative_to(snapshot)
                if variant and relative_path.parts[0] not in (variant.strip("/"), "systemone.yaml"):
                    continue
                cache.link(source.resolve(), root / relative_path)
        result.root = root

    return result


def snapshot_download(
    repo: str,
    version: str | None = None,
    variant: str | None = None,
    client: Client | None = None,
) -> Path:
    """Download a version into the cache and return its directory.

    from systemone import snapshot_download
    path = snapshot_download("biplov/snake-balanced-multilingual", variant="onnx-int8")
    """
    own = client is None
    registry = client or Client()
    try:
        return pull_version(registry, repo, None, version, variant).root
    finally:
        if own:
            registry.close()


def files_under(source: Path) -> list[Path]:
    return walk(source)
