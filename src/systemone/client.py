"""HTTP client for the registry.

Synchronous on purpose. The CLI is the primary consumer and an async CLI buys
nothing but an event loop to explain; uploads and downloads are I/O-bound on one
connection at a time either way, and httpx gives connection reuse without it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import httpx

from systemone import __version__
from systemone.config import Config, load
from systemone.errors import (
    ApiError,
    AuthError,
    ChecksumMismatch,
    ConnectionFailed,
    NotFound,
    SystemOneError,
)

# Cloudflare rejects the default library user agents on r2.dev as bot traffic,
# with a 403 that reads exactly like a permissions error. Identify properly.
USER_AGENT = f"systemone/{__version__} (+https://systemonemodels.tech)"

CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


class Client:
    def __init__(self, config: Config | None = None, timeout: float = 60.0):
        self.config = config or load()
        headers = {"user-agent": USER_AGENT, "accept": "application/json"}
        if self.config.token:
            headers["authorization"] = f"Bearer {self.config.token}"
        self._http = httpx.Client(
            base_url=self.config.endpoint,
            headers=headers,
            timeout=httpx.Timeout(timeout, read=300.0, write=900.0),
            follow_redirects=True,
        )

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    # --- plumbing ---------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.TransportError as exc:
            # DNS failures, refused connections, timeouts: one readable line
            # instead of a traceback, with the setting that most often causes it.
            raise ConnectionFailed(
                f"Could not reach {self.config.endpoint} ({_reason(exc)}). Check your "
                "connection, or choose the registry with `systemone login --endpoint URL` "
                "or SYSTEMONE_ENDPOINT."
            ) from exc

        if response.status_code == 401:
            raise AuthError("Not signed in. Run `systemone login`.")
        if response.status_code == 404:
            raise NotFound(_detail(response) or "not found")
        if response.status_code >= 400:
            payload = _json(response) or {}
            raise ApiError(
                response.status_code,
                payload.get("detail", response.text[:200]),
                payload.get("code", "error"),
                payload.get("errors", []),
            )

        return _json(response)

    # --- identity ---------------------------------------------------------

    def whoami(self) -> dict[str, Any] | None:
        result: dict[str, Any] | None = self._request("GET", "/v1/auth/me")
        return result

    def start_device_login(self, client_name: str) -> dict[str, Any]:
        result: dict[str, Any] = self._request(
            "POST", "/v1/auth/device", json={"client_name": client_name}
        )
        return result

    def tokens(self) -> list[dict[str, Any]]:
        """The account's live access tokens: id, name and display prefix."""
        found: list[dict[str, Any]] = self._request("GET", "/v1/tokens")
        return found

    def revoke_token(self, token_id: str) -> None:
        self._request("DELETE", f"/v1/tokens/{token_id}")

    def poll_device_login(self, device_code: str) -> dict[str, Any]:
        result: dict[str, Any] = self._request(
            "POST", "/v1/auth/device/token", json={"device_code": device_code}
        )
        return result

    # --- discovery --------------------------------------------------------

    def search(self, query: str | None = None, **filters: Any) -> dict[str, Any]:
        params: dict[str, Any] = {k: v for k, v in filters.items() if v not in (None, (), [])}
        if query:
            params["q"] = query
        results: dict[str, Any] = self._request("GET", "/v1/search/models", params=params)
        return results

    def model(self, repo: str) -> dict[str, Any]:
        detail: dict[str, Any] = self._request("GET", f"/v1/models/{repo}")
        return detail

    def versions(self, repo: str) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = self._request("GET", f"/v1/models/{repo}/versions")
        return entries

    def validate_manifest(self, manifest: str) -> dict[str, Any]:
        report: dict[str, Any] = self._request(
            "POST",
            "/v1/manifest/validate",
            content=manifest.encode(),
            headers={"content-type": "text/plain"},
        )
        return report

    # --- publishing -------------------------------------------------------

    def create_model(
        self, namespace: str, name: str, manifest: str | None, private: bool = False
    ) -> dict[str, Any]:
        created: dict[str, Any] = self._request(
            "POST",
            "/v1/models",
            json={
                "namespace": namespace,
                "name": name,
                "visibility": "private" if private else "public",
                "manifest_yaml": manifest,
            },
        )
        return created

    def start_upload(self, repo: str, path: str, size: int, digest: str | None) -> dict[str, Any]:
        ticket: dict[str, Any] = self._request(
            "POST",
            f"/v1/models/{repo}/uploads",
            json={"path": path, "size_bytes": size, "sha256": digest},
        )
        return ticket

    def finish_upload(self, repo: str, upload_id: str, **body: Any) -> dict[str, Any]:
        result: dict[str, Any] = self._request(
            "POST", f"/v1/models/{repo}/uploads/{upload_id}/complete", json=body
        )
        return result

    def abort_upload(self, repo: str, upload_id: str) -> None:
        self._request("DELETE", f"/v1/models/{repo}/uploads/{upload_id}")

    def publish_version(
        self,
        repo: str,
        version: str,
        manifest: str,
        artifacts: list[dict[str, Any]],
        notes: str | None = None,
        readme: str | None = None,
    ) -> dict[str, Any]:
        """Publish a version. A readme becomes the repository's model card."""
        body: dict[str, Any] = {
            "version": version,
            "manifest_yaml": manifest,
            "notes": notes,
            "artifacts": artifacts,
        }
        if readme is not None:
            body["readme"] = readme
        published: dict[str, Any] = self._request("POST", f"/v1/models/{repo}/versions", json=body)
        return published

    # --- bytes ------------------------------------------------------------

    def put(self, url: str, data: bytes | Iterable[bytes], size: int | None = None) -> str:
        """Upload to a presigned URL.

        A plain client, not self._http: the presigned URL carries its own
        authorization, and sending our Authorization header alongside it makes
        some S3 implementations reject the request as doubly signed.

        `data` may be chunks, so a large file is streamed from disk rather than
        held in memory. Its `size` is then required: an explicit Content-Length
        stops httpx from falling back to chunked encoding, which a presigned
        PUT does not accept.
        """
        headers = {"content-type": "application/octet-stream", "user-agent": USER_AGENT}
        if not isinstance(data, bytes):
            if size is None:
                raise ValueError("size is required when streaming an upload")
            headers["content-length"] = str(size)
        try:
            with httpx.Client(timeout=httpx.Timeout(60.0, write=1800.0)) as plain:
                response = plain.put(url, content=data, headers=headers)
                response.raise_for_status()
                return str(response.headers.get("ETag", ""))
        except httpx.HTTPStatusError as exc:
            raise SystemOneError(
                f"Storage refused the upload (HTTP {exc.response.status_code})."
            ) from exc
        except httpx.TransportError as exc:
            raise ConnectionFailed(f"Lost the connection to storage ({_reason(exc)}).") from exc

    def download(self, url: str, target: Path, expected_sha256: str | None = None) -> Iterator[int]:
        """Stream a file to disk, yielding bytes written.

        Written to a neighbouring .part file and renamed at the end, so an
        interrupted download never leaves something that looks complete.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        digest = hashlib.sha256()

        try:
            with (
                httpx.Client(
                    timeout=httpx.Timeout(60.0, read=1800.0), follow_redirects=True
                ) as plain,
                plain.stream("GET", url, headers={"user-agent": USER_AGENT}) as response,
            ):
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in response.iter_bytes(CHUNK):
                        handle.write(chunk)
                        digest.update(chunk)
                        yield len(chunk)
        except httpx.HTTPStatusError as exc:
            partial.unlink(missing_ok=True)
            raise SystemOneError(
                f"Download of {target.name} failed (HTTP {exc.response.status_code})."
            ) from exc
        except httpx.TransportError as exc:
            partial.unlink(missing_ok=True)
            raise ConnectionFailed(
                f"Lost the connection downloading {target.name} ({_reason(exc)})."
            ) from exc

        if expected_sha256 and digest.hexdigest() != expected_sha256:
            partial.unlink(missing_ok=True)
            raise ChecksumMismatch(f"{target.name} did not match its recorded SHA-256")

        partial.replace(target)


def _reason(exc: httpx.TransportError) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return "timed out"
    text = str(exc).strip()
    return text[:120] if text else type(exc).__name__


def _json(response: httpx.Response) -> Any:
    try:
        return response.json() if response.content else None
    except ValueError:
        return None


def _detail(response: httpx.Response) -> str | None:
    payload = _json(response)
    return payload.get("detail") if isinstance(payload, dict) else None
