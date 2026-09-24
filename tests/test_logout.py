"""`systemone logout` revokes what login stored, and nothing else."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from systemone import cli, config
from systemone.errors import AuthError, ConnectionFailed

runner = CliRunner()
STORED = "s1_pat_abcdefRESTOFTHETOKEN"


class Registry:
    def __init__(self, fail: Exception | None = None) -> None:
        self.fail = fail
        self.revoked: list[str] = []
        self.seen_tokens: list[str | None] = []

    def __call__(self, settings: config.Config) -> Registry:
        self.seen_tokens.append(settings.token)
        return self

    def __enter__(self) -> Registry:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def tokens(self) -> list[dict[str, Any]]:
        if self.fail:
            raise self.fail
        return [
            {"id": "other", "prefix": "s1_pat_zzzzzz"},
            {"id": "mine", "prefix": "s1_pat_abcdef"},
        ]

    def revoke_token(self, token_id: str) -> None:
        self.revoked.append(token_id)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SYSTEMONE_HOME", str(tmp_path))
    monkeypatch.delenv("SYSTEMONE_TOKEN", raising=False)
    monkeypatch.delenv("SYSTEMONE_ENDPOINT", raising=False)
    config.save(config.Config(endpoint="https://dev-api.example.test", token=STORED, username="me"))
    return tmp_path


def test_logout_revokes_the_stored_token_and_forgets_everything(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = Registry()
    monkeypatch.setattr(cli, "Client", registry)
    result = runner.invoke(cli.app, ["logout"])

    assert result.exit_code == 0, result.output
    assert registry.revoked == ["mine"]
    assert "revoked" in result.output
    after = config.load(environment=False)
    assert after.token is None
    # The registry's address goes too: the next login is back to the default.
    assert after.endpoint == config.DEFAULT_ENDPOINT


def test_an_unreachable_registry_still_signs_out_here(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "Client", Registry(fail=ConnectionFailed("down")))
    result = runner.invoke(cli.app, ["logout"])

    assert result.exit_code == 0
    assert config.load(environment=False).token is None
    assert "still valid" in result.output
    assert "settings/tokens" in result.output


def test_a_dead_token_is_simply_forgotten(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "Client", Registry(fail=AuthError("no")))
    result = runner.invoke(cli.app, ["logout"])
    assert "already stopped working" in result.output
    assert config.load(environment=False).token is None


def test_a_token_from_the_environment_is_never_revoked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYSTEMONE_HOME", str(tmp_path))
    monkeypatch.setenv("SYSTEMONE_TOKEN", "s1_pat_ciciciTOKEN")
    registry = Registry()
    monkeypatch.setattr(cli, "Client", registry)
    result = runner.invoke(cli.app, ["logout"])

    assert result.exit_code == 0
    assert registry.revoked == []
    assert registry.seen_tokens == []
    assert "Not signed in on this machine" in result.output
    assert "SYSTEMONE_TOKEN" in result.output
