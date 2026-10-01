"""Uploads stream from disk and report progress as bytes leave; pulls build snapshots."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from systemone import cache
from systemone.client import Client
from systemone.config import Config
from systemone.transfer import pull_version, push_files


def test_a_streamed_put_declares_its_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = request.headers
        seen["body"] = request.read()
        return httpx.Response(200, headers={"ETag": '"abc"'})

    real = httpx.Client

    def client_with_transport(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", client_with_transport)
    etag = Client(Config(endpoint="https://api.test")).put(
        "https://bucket.test/object", iter([b"abc", b"def"]), 6, {"x-amz-checksum-sha256": "Zm9v"}
    )

    assert etag == '"abc"'
    assert seen["body"] == b"abcdef"
    assert seen["headers"]["x-amz-checksum-sha256"] == "Zm9v"
    # A presigned PUT refuses chunked encoding, so the length must be explicit.
    assert seen["headers"]["content-length"] == "6"
    assert "transfer-encoding" not in seen["headers"]


def test_streaming_without_a_size_is_refused() -> None:
    with pytest.raises(ValueError, match="size"):
        Client(Config(endpoint="https://api.test")).put("https://x.test", iter([b"a"]))


class Storage:
    """Enough of the registry for push_files, keeping every byte it is sent."""

    def __init__(self, ticket: dict[str, Any]) -> None:
        self.ticket = ticket
        self.puts: list[bytes] = []
        self.headers: list[dict[str, str]] = []

    def start_upload(self, repo: str, path: str, size: int, digest: str | None) -> dict[str, Any]:
        return self.ticket

    def put(
        self,
        url: str,
        data: bytes | Iterable[bytes],
        size: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        body = data if isinstance(data, bytes) else b"".join(data)
        assert size is None or len(body) == size
        self.puts.append(body)
        self.headers.append(headers or {})
        return f'"{len(self.puts)}"'

    def finish_upload(self, repo: str, upload_id: str, **body: Any) -> dict[str, Any]:
        return {
            "uri": "r2://k",
            "path": "weights.bin",
            "filename": "weights.bin",
            "size_bytes": 10,
            "storage_key": "k",
        }

    def abort_upload(self, repo: str, upload_id: str) -> None:
        return None


def test_every_byte_is_reported_once(tmp_path: Path) -> None:
    (tmp_path / "weights.bin").write_bytes(b"0123456789")
    signed = {"content-length": "10", "x-amz-checksum-sha256": "abc="}
    storage = Storage({"upload_id": "u", "url": "https://bucket.test/k", "headers": signed})
    reported: list[int] = []

    push_files(
        storage,  # type: ignore[arg-type]
        "me/x",
        tmp_path,
        [tmp_path / "weights.bin"],
        on_bytes=reported.append,
    )

    assert storage.puts == [b"0123456789"]
    # The headers the signature covers go out exactly as the registry gave them.
    assert storage.headers == [signed]
    assert sum(reported) == 10


def test_multipart_uploads_send_each_range(tmp_path: Path) -> None:
    (tmp_path / "weights.bin").write_bytes(b"0123456789")
    parts = [{"part_number": n, "url": f"https://bucket.test/{n}"} for n in (1, 2, 3)]
    storage = Storage({"upload_id": "u", "part_size": 4, "parts": parts})
    reported: list[int] = []

    push_files(
        storage,  # type: ignore[arg-type]
        "me/x",
        tmp_path,
        [tmp_path / "weights.bin"],
        on_bytes=reported.append,
    )

    assert storage.puts == [b"0123", b"4567", b"89"]
    assert sum(reported) == 10


# ---- pulls --------------------------------------------------------------------------

SHIPPED = b'spec_version: "0.1"\nmodel: mira\nnamespace: sagea\n'
REGISTRY = "spec_version: '0.1'\nmodel: mira\nnamespace: sagea\nartifacts: []\n"


class Registry:
    """Enough of the registry for pull_version: one version, its files served from memory."""

    def __init__(self, files: dict[str, bytes], manifest: str | None = REGISTRY) -> None:
        self.files = files
        self.manifest = manifest
        self.downloads: list[str] = []

    def model(self, repo: str) -> dict[str, Any]:
        return {"latest_version": "0.1.4", "manifest_yaml": self.manifest}

    def versions(self, repo: str) -> list[dict[str, Any]]:
        artifacts = [
            {
                "uri": f"https://bucket.test/{path}",
                "path": path,
                "filename": path.rsplit("/", 1)[-1],
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for path, data in self.files.items()
        ]
        return [{"version": "0.1.4", "artifacts": artifacts}]

    def download(self, url: str, target: Path, expected_sha256: str | None = None) -> Iterator[int]:
        path = url.removeprefix("https://bucket.test/")
        self.downloads.append(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[path])
        yield len(self.files[path])


@pytest.fixture()
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SYSTEMONE_CACHE", str(tmp_path / "cache"))
    return tmp_path


def test_a_version_that_ships_its_manifest_keeps_it(isolated_cache: Path) -> None:
    """sagea/mira 0.1.4 ships systemone.yaml. Writing the registry's manifest over it went
    through the snapshot's link into the read-only blob and failed; had the blob been
    writable, it would have changed the bytes every snapshot shares."""
    registry = Registry({"systemone.yaml": SHIPPED, "model.safetensors": b"weights"})
    digest = hashlib.sha256(SHIPPED).hexdigest()

    for _ in range(2):  # the second pull comes from the cache
        result = pull_version(registry, "sagea/mira", isolated_cache / "out")  # type: ignore[arg-type]

    snapshot = cache.snapshot_root("sagea/mira", "0.1.4")
    assert (snapshot / "systemone.yaml").read_bytes() == SHIPPED
    assert (result.root / "systemone.yaml").read_bytes() == SHIPPED
    assert (result.root / "model.safetensors").read_bytes() == b"weights"
    assert cache.has_blob(digest)
    assert registry.downloads == ["systemone.yaml", "model.safetensors"]


def test_a_version_without_one_gets_the_registry_manifest(isolated_cache: Path) -> None:
    registry = Registry({"model.safetensors": b"weights"})
    result = pull_version(registry, "me/x", isolated_cache / "out")  # type: ignore[arg-type]
    snapshot = cache.snapshot_root("me/x", "0.1.4")
    assert (snapshot / "systemone.yaml").read_text() == REGISTRY
    assert (result.root / "systemone.yaml").read_text() == REGISTRY


def test_the_manifest_is_written_in_place_of_a_link_not_through_it(isolated_cache: Path) -> None:
    """A link left at the manifest's path is replaced; the blob behind it is untouched."""
    shared = b"a file other snapshots share"
    digest = hashlib.sha256(shared).hexdigest()
    staging = cache.blob_path(digest).with_suffix(".incoming")
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.write_bytes(shared)
    blob = cache.adopt(staging, digest)
    blob.chmod(0o644)  # even a writable blob must not be written through
    snapshot = cache.snapshot_root("me/x", "0.1.4")
    cache.link(blob, snapshot / "systemone.yaml")

    pull_version(Registry({"model.safetensors": b"weights"}), "me/x")  # type: ignore[arg-type]

    assert (snapshot / "systemone.yaml").read_text() == REGISTRY
    assert blob.read_bytes() == shared and cache.has_blob(digest)


def test_a_variant_pull_brings_the_shipped_manifest(isolated_cache: Path) -> None:
    registry = Registry(
        {"systemone.yaml": SHIPPED, "onnx/model.onnx": b"graph", "gguf/model.gguf": b"gguf"}
    )
    out = isolated_cache / "out"
    result = pull_version(registry, "me/x", out, variant="onnx")  # type: ignore[arg-type]
    files = sorted(p.relative_to(result.root).as_posix() for p in result.root.rglob("*"))
    assert files == ["onnx", "onnx/model.onnx", "systemone.yaml"]
    assert (result.root / "systemone.yaml").read_bytes() == SHIPPED
    assert "gguf/model.gguf" not in registry.downloads
