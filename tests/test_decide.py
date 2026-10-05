"""Calling models through the inference API: Client.decide, and `systemone decide`."""

from __future__ import annotations

import json
import re
from collections.abc import Callable

import httpx
import pytest
from typer.testing import CliRunner

from systemone import cli
from systemone.client import Client
from systemone.config import Config
from systemone.errors import ApiError, AuthError, CreditsRequired, NotFound, RateLimited

runner = CliRunner()

Handler = Callable[[httpx.Request], httpx.Response]

ANSWER = {
    "model": "nokia/anyjev",
    "version": "0.2.0-qwen3-1.7b",
    "checkpoint": "f16",
    "answers": {
        "intent": {
            "type": "choice",
            "choice": "refund",
            "probabilities": {"refund": 0.9, "track delivery": 0.1},
        }
    },
    "usage": {"input_tokens": 50, "output_tokens": 0, "decisions": 1},
    "latency_ms": 812.0,
}
QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "What does the customer want?",
        "criteria": ["refund", "track delivery"],
    }
}


@pytest.fixture(autouse=True)
def no_key_in_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SYSTEMONE_API_KEY", raising=False)


def plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def served(handler: Handler, *, api_key: str | None = None, token: str | None = None) -> Client:
    client = Client(api_key, config=Config(endpoint="https://api.test", token=token))
    client._http = httpx.Client(
        base_url="https://api.test",
        headers=dict(client._http.headers),
        transport=httpx.MockTransport(handler),
    )
    return client


def test_decide_sends_the_request_with_the_api_key() -> None:
    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=ANSWER)

    result = served(answer, api_key="s1_pat_key").decide(
        "nokia/anyjev", "Customer: I want my money back.", QUESTIONS, checkpoint="f16"
    )
    assert result == ANSWER
    [request] = seen
    assert (request.method, request.url.path) == ("POST", "/v1/systemone")
    assert request.headers["authorization"] == "Bearer s1_pat_key"
    assert json.loads(request.content) == {
        "model": "nokia/anyjev",
        "state": "Customer: I want my money back.",
        "questions": QUESTIONS,
        "checkpoint": "f16",
    }


def test_the_key_is_used_for_models_only() -> None:
    auth: dict[str, str | None] = {}

    def record(request: httpx.Request) -> httpx.Response:
        auth[request.url.path] = request.headers.get("authorization")
        if request.url.path == "/v1/systemone":
            return httpx.Response(200, json=ANSWER)
        return httpx.Response(200, json={"items": [], "total": 0})

    client = served(record, api_key="s1_pat_key", token="s1_pat_login")
    client.search("routing")
    client.decide("nokia/anyjev", "state", QUESTIONS)
    assert auth == {
        "/v1/search/models": "Bearer s1_pat_login",
        "/v1/systemone": "Bearer s1_pat_key",
    }


def test_the_key_comes_from_the_environment_then_the_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth: list[str | None] = []

    def record(request: httpx.Request) -> httpx.Response:
        auth.append(request.headers.get("authorization"))
        return httpx.Response(200, json=ANSWER)

    served(record, token="s1_pat_login").decide("nokia/anyjev", "state", QUESTIONS)
    monkeypatch.setenv("SYSTEMONE_API_KEY", "s1_pat_env")
    served(record, token="s1_pat_login").decide("nokia/anyjev", "state", QUESTIONS)
    served(record, api_key="s1_pat_given").decide("nokia/anyjev", "state", QUESTIONS)
    assert auth == ["Bearer s1_pat_login", "Bearer s1_pat_env", "Bearer s1_pat_given"]


def test_without_any_key_it_says_where_to_get_one() -> None:
    def never(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be sent")

    client = served(never)
    client.config.endpoint = "https://api.systemonemodels.ai"
    with pytest.raises(AuthError, match=r"systemonemodels\.ai/settings/api"):
        client.decide("nokia/anyjev", "state", QUESTIONS)


def test_limits_raise_rate_limited_with_the_wait() -> None:
    def over(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"Retry-After": "42"},
            json={"detail": "You have used today's 500 decisions.", "code": "quota_exceeded"},
        )

    with pytest.raises(RateLimited, match="500 decisions") as caught:
        served(over, api_key="k").decide("nokia/anyjev", "state", QUESTIONS)
    assert caught.value.code == "quota_exceeded"
    assert caught.value.retry_after == 42.0
    assert caught.value.status == 429


def test_a_gpu_model_without_credit_raises_credits_required() -> None:
    detail = (
        "acme/big runs on GPUs, which are paid from prepaid credit, and your account has "
        "none. Add credits in Settings → Billing: https://web.test/settings/billing."
    )

    def broke(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"detail": detail, "code": "credits_required"})

    with pytest.raises(CreditsRequired, match="Settings → Billing") as caught:
        served(broke, api_key="k").decide("acme/big", "state", QUESTIONS)
    assert caught.value.status == 402
    assert caught.value.code == "credits_required"
    # Code that already catches ApiError keeps working.
    assert isinstance(caught.value, ApiError)


def test_the_decide_command_says_where_to_add_credit(monkeypatch: pytest.MonkeyPatch) -> None:
    def broke(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            402,
            json={
                "detail": "Add credits in Settings → Billing: https://web.test/settings/billing.",
                "code": "credits_required",
            },
        )

    monkeypatch.setattr(cli, "client", lambda: served(broke, api_key="k"))
    result = runner.invoke(cli.app, ["decide", "acme/big"])
    assert result.exit_code == 1
    assert "https://web.test/settings/billing" in plain(result.output)


def test_a_refused_key_and_an_unknown_model_are_explained() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "This API key was not accepted: it expired."})

    with pytest.raises(AuthError, match="it expired"):
        served(refuse, api_key="k").decide("nokia/anyjev", "state", QUESTIONS)

    def unknown(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "There is no model named 'x/y'."})

    with pytest.raises(NotFound, match="no model named"):
        served(unknown, api_key="k").decide("x/y", "state", QUESTIONS)


def test_served_models_and_usage() -> None:
    def api(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/systemone/models":
            return httpx.Response(200, json={"object": "list", "data": [{"id": "nokia/anyjev"}]})
        assert request.url.path == "/v1/usage"
        assert request.url.params["days"] == "7"
        assert request.headers["authorization"] == "Bearer k"
        return httpx.Response(200, json={"plan": {"name": "free"}, "used": 3})

    client = served(api, api_key="k")
    assert client.served_models() == [{"id": "nokia/anyjev"}]
    assert client.usage(days=7)["used"] == 3


def test_a_config_passed_first_still_works() -> None:
    client = Client(Config(endpoint="https://api.test", token="s1_pat_login"))
    assert client.config.endpoint == "https://api.test"
    assert client.api_key is None
    assert client._http.headers["authorization"] == "Bearer s1_pat_login"


def test_the_decide_command(monkeypatch: pytest.MonkeyPatch) -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["state"] == "Customer: refund please"
        return httpx.Response(200, json=ANSWER)

    monkeypatch.setattr(cli, "client", lambda: served(answer, api_key="k"))
    args = ["decide", "nokia/anyjev", "--state", "Customer: refund please"]
    shown = runner.invoke(cli.app, [*args, "--questions", json.dumps(QUESTIONS)])
    assert shown.exit_code == 0, shown.output
    out = plain(shown.output)
    assert "refund" in out and "1 decisions" in out and "812 ms" in out

    raw = runner.invoke(cli.app, [*args, "--questions", json.dumps(QUESTIONS), "--json"])
    assert json.loads(raw.output) == ANSWER


def test_the_decide_command_explains_a_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    def over(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"detail": "Your Free plan allows 10 requests a minute."})

    monkeypatch.setattr(cli, "client", lambda: served(over, api_key="k"))
    result = runner.invoke(cli.app, ["decide", "nokia/anyjev"])
    assert result.exit_code == 1
    assert "10 requests a minute" in plain(result.output)
