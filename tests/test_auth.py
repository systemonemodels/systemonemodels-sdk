"""The CLI's side of browser login, against a scripted registry."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from systemone.auth import can_open_browser, device_login
from systemone.client import Client
from systemone.config import Config
from systemone.errors import LoginFailed

CODES: dict[str, Any] = {
    "device_code": "secret-device-code",
    "user_code": "BCDF-GHJK",
    "verification_uri": "https://systemonemodels.tech/device",
    "verification_uri_complete": "https://systemonemodels.tech/device?code=BCDF-GHJK",
    "expires_in": 600,
    "interval": 5,
}


def registry(polls: list[tuple[int, dict[str, Any]]]) -> tuple[Client, list[dict[str, Any]]]:
    """A client whose registry answers the start call, then each poll in turn."""
    seen: list[dict[str, Any]] = []
    answers = iter(polls)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        seen.append({"path": request.url.path, "body": body})
        if request.url.path == "/v1/auth/device":
            return httpx.Response(201, json=CODES)
        status, payload = next(answers)
        return httpx.Response(status, json=payload)

    client = Client(Config(endpoint="https://api.test"))
    client._http = httpx.Client(base_url="https://api.test", transport=httpx.MockTransport(handler))
    return client, seen


def pending(code: str = "authorization_pending") -> tuple[int, dict[str, Any]]:
    return 400, {"detail": code, "code": code, "errors": []}


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def clock(self) -> float:
        return self.now


def test_waits_for_approval_then_returns_the_token() -> None:
    granted = {"token": "s1_pat_x", "token_name": "CLI · box", "username": "biplov"}
    client, seen = registry([pending(), pending(), (200, granted)])
    shown: list[dict[str, Any]] = []
    t = FakeTime()

    result = device_login(client, "box", shown.append, sleep=t.sleep, clock=t.clock)

    assert result == granted
    assert shown == [CODES]
    assert seen[0]["body"] == {"client_name": "box"}
    # Every poll carries the secret device code, never the short user code.
    assert all(s["body"] == {"device_code": "secret-device-code"} for s in seen[1:])
    assert t.sleeps == [5, 5, 5]


def test_backs_off_when_told_to_slow_down() -> None:
    client, _ = registry([pending("slow_down"), pending(), (200, {"token": "t", "username": "u"})])
    t = FakeTime()
    device_login(client, "box", lambda _: None, sleep=t.sleep, clock=t.clock)
    assert t.sleeps == [5, 10, 10]


@pytest.mark.parametrize("code", ["access_denied", "expired_token"])
def test_denied_or_expired_is_a_clear_failure(code: str) -> None:
    client, _ = registry([pending(code)])
    t = FakeTime()
    with pytest.raises(LoginFailed):
        device_login(client, "box", lambda _: None, sleep=t.sleep, clock=t.clock)


def test_gives_up_when_the_code_expires() -> None:
    client, _ = registry([pending()] * 200)
    t = FakeTime()
    with pytest.raises(LoginFailed, match="expired"):
        device_login(client, "box", lambda _: None, sleep=t.sleep, clock=t.clock)
    assert t.now >= CODES["expires_in"]


def test_never_opens_a_browser_over_ssh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.1 22 10.0.0.2 22")
    assert can_open_browser() is False
