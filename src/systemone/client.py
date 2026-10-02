"""HTTP client for the registry, and for the models it serves.

Synchronous on purpose. The CLI is the primary consumer and an async CLI buys
nothing but an event loop to explain; uploads and downloads are I/O-bound on one
connection at a time either way, and httpx gives connection reuse without it.
"""

from __future__ import annotations

import hashlib
import os
import sys
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import httpx

from systemone import __version__
from systemone.config import ENV_API_KEY, Config, load, web_url
from systemone.errors import (
    ApiError,
    AuthError,
    ChecksumMismatch,
    ConnectionFailed,
    NotFound,
    RateLimited,
    SystemOneError,
)

# Cloudflare rejects the default library user agents on r2.dev as bot traffic,
# with a 403 that reads exactly like a permissions error. Identify properly.
USER_AGENT = f"systemone/{__version__} (+https://systemonemodels.ai)"

CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


class Client:
    """The registry, and the models it serves through the inference API.

    Registry calls (search, pull, push) are made as the signed-in account: the
    token from `systemone login`, or SYSTEMONE_TOKEN. Calls to models
    (`decide`, `usage`) use an API key: the one given here, else
    SYSTEMONE_API_KEY, else that same login token. An API key never reaches the
    registry calls, so setting one cannot change what `systemone push` does.

        from systemone import Client

        client = Client("s1_pat_...")  # or set SYSTEMONE_API_KEY
        result = client.decide(
            "nokia/anyjev",
            "Customer: where is my parcel?",
            {"intent": {"type": "choice", "instructions": "What does the customer want?",
                        "criteria": ["refund", "track delivery", "cancel order"]}},
        )
        result["answers"]["intent"]["choice"]
    """

    def __init__(
        self,
        api_key: str | Config | None = None,
        timeout: float = 60.0,
        *,
        config: Config | None = None,
    ):
        if isinstance(api_key, Config):
            # Client(config), the way every release before 0.4 was called.
            config, api_key = api_key, None
        self.config = config or load()
        self.api_key = api_key or os.environ.get(ENV_API_KEY) or None
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

    def _request(
        self, method: str, path: str, *, auth_error: str | None = None, **kwargs: Any
    ) -> Any:
        """One call. `auth_error` explains a 401 when the server's own words do not."""
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

        _notice(response)
        if response.status_code == 401:
            if auth_error is not None:
                raise AuthError(_detail(response) or auth_error)
            raise AuthError("Not signed in. Run `systemone login`.")
        if response.status_code == 404:
            raise NotFound(_detail(response) or "not found")
        if response.status_code == 429:
            payload = _json(response) or {}
            raise RateLimited(
                429,
                payload.get("detail", "Too many requests. Wait a little and try again."),
                payload.get("code", "rate_limited"),
                payload.get("errors", []),
                retry_after=_seconds(response.headers.get("retry-after")),
            )
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

    # --- models, through the inference API ---------------------------------

    def _keyed(self) -> dict[str, str]:
        """The Authorization header for calls to models."""
        key = self.api_key or self.config.token
        if not key:
            raise AuthError(
                f"No API key. Pass Client(api_key) or set {ENV_API_KEY}; create one at "
                f"{web_url(self.config.endpoint)}/settings/api."
            )
        return {"authorization": f"Bearer {key}"}

    def decide(
        self,
        model: str,
        state: str,
        questions: Mapping[str, Mapping[str, Any]],
        *,
        checkpoint: str | None = None,
    ) -> dict[str, Any]:
        """Ask a model the inference API serves. Each question answered is one decision.

        `questions` maps an id of your choosing to a question: `type` (choice,
        score or noul), `instructions`, and `criteria` (choice: the options, as
        a list or {option: description}; score: the levels, lowest first; noul:
        nothing). The answer has `answers` by question id, `usage` and
        `latency_ms`. Raises RateLimited when over a limit or out of decisions.
        """
        body: dict[str, Any] = {
            "model": model,
            "state": state,
            "questions": {qid: dict(q) for qid, q in questions.items()},
        }
        if checkpoint is not None:
            body["checkpoint"] = checkpoint
        answer: dict[str, Any] = self._request(
            "POST",
            "/v1/systemone",
            json=body,
            headers=self._keyed(),
            auth_error="The API key was not accepted.",
        )
        return answer

    def served_models(self) -> list[dict[str, Any]]:
        """The models the inference API serves: id, state, checkpoints and limits."""
        listing: dict[str, Any] = self._request("GET", "/v1/systemone/models")
        models: list[dict[str, Any]] = listing.get("data", [])
        return models

    def usage(self, days: int = 30) -> dict[str, Any]:
        """Your plan, what is left of it, and your calls by day, model and key."""
        report: dict[str, Any] = self._request(
            "GET",
            "/v1/usage",
            params={"days": days},
            headers=self._keyed(),
            auth_error="The API key was not accepted.",
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

    def put(
        self,
        url: str,
        data: bytes | Iterable[bytes],
        size: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        """Upload to a presigned URL.

        A plain client, not self._http: the presigned URL carries its own
        authorization, and sending our Authorization header alongside it makes
        some S3 implementations reject the request as doubly signed.

        `data` may be chunks, so a large file is streamed from disk rather than
        held in memory. Its `size` is then required: an explicit Content-Length
        stops httpx from falling back to chunked encoding, which a presigned
        PUT does not accept.

        `headers` are the ones the registry's ticket says the signature covers
        — the size, and the file's SHA-256 — sent exactly as given so the store
        can refuse bytes that differ from what was declared.
        """
        sent = {"content-type": "application/octet-stream", "user-agent": USER_AGENT}
        sent.update(headers or {})
        if not isinstance(data, bytes):
            if size is None:
                raise ValueError("size is required when streaming an upload")
            sent["content-length"] = str(size)
        headers = sent
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


# A message the registry attaches to its answers, such as "this version of the CLI is
# out of date": shown once per run, on stderr so it never mixes with a command's output.
NOTICE_HEADER = "x-systemone-notice"
_shown_notices: set[str] = set()


def _notice(response: httpx.Response) -> None:
    text = response.headers.get(NOTICE_HEADER, "").strip()
    if text and text not in _shown_notices:
        _shown_notices.add(text)
        print(f"systemone: {text}", file=sys.stderr)


def _detail(response: httpx.Response) -> str | None:
    payload = _json(response)
    return payload.get("detail") if isinstance(payload, dict) else None


def _seconds(value: str | None) -> float | None:
    try:
        return float(value) if value else None
    except ValueError:
        return None
