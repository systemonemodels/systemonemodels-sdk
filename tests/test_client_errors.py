"""Failures a user can do something about come back as one readable line."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from systemone.client import Client
from systemone.config import Config
from systemone.errors import ConnectionFailed, SystemOneError


def client_that(handler: httpx.MockTransport) -> Client:
    client = Client(Config(endpoint="https://api.test"))
    client._http = httpx.Client(base_url="https://api.test", transport=handler)
    return client


def test_unreachable_registry_names_the_endpoint() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 61] Connection refused", request=request)

    with pytest.raises(ConnectionFailed, match="Could not reach https://api.test"):
        client_that(httpx.MockTransport(refuse)).search("routing")


def test_timeouts_say_so() -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(ConnectionFailed, match="timed out"):
        client_that(httpx.MockTransport(slow)).whoami()


def test_failed_download_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def gone(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="denied")

    real = httpx.Client

    def patched(*args: object, **kwargs: object) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(gone)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "Client", patched)
    target = tmp_path / "model.onnx"
    with pytest.raises(SystemOneError, match="HTTP 403"):
        list(Client(Config(endpoint="https://api.test")).download("https://files.test/x", target))
    assert not target.exists()
    assert not target.with_suffix(".onnx.part").exists()
