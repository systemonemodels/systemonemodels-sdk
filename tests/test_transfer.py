"""Uploads stream from disk and report progress as bytes leave."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx
import pytest

from systemone.client import Client
from systemone.config import Config
from systemone.transfer import push_files


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
