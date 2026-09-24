"""Where the token lives, and who can read it."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from systemone import config


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYSTEMONE_HOME", str(tmp_path / "config"))
    monkeypatch.delenv("SYSTEMONE_TOKEN", raising=False)
    monkeypatch.delenv("SYSTEMONE_ENDPOINT", raising=False)


def test_round_trip() -> None:
    config.save(config.Config(endpoint="https://example.test", token="s1_pat_x", username="me"))
    loaded = config.load()
    assert loaded.token == "s1_pat_x"
    assert loaded.username == "me"
    assert loaded.authenticated


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_the_token_file_is_not_world_readable() -> None:
    path = config.save(config.Config(token="s1_pat_secret"))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_environment_beats_the_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """So CI can authenticate without writing a config."""
    config.save(config.Config(token="from-file", endpoint="https://file.test"))
    monkeypatch.setenv("SYSTEMONE_TOKEN", "from-env")
    monkeypatch.setenv("SYSTEMONE_ENDPOINT", "https://env.test/")

    loaded = config.load()
    assert loaded.token == "from-env"
    # Trailing slash removed, or every path would double up its separator.
    assert loaded.endpoint == "https://env.test"


def test_a_corrupt_config_does_not_break_the_client() -> None:
    path = config.config_dir() / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    assert config.load().token is None


def test_logout_removes_the_file() -> None:
    path = config.save(config.Config(token="x"))
    config.clear()
    assert not path.exists()
    assert not config.load().authenticated


def test_web_url_follows_the_api_host() -> None:
    from systemone.config import web_url

    assert web_url("https://api.systemonemodels.tech") == "https://systemonemodels.tech"
    assert web_url("https://dev-api.systemonemodels.tech") == "https://dev.systemonemodels.tech"
    assert web_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000"


def test_empty_environment_variables_do_not_blank_the_stored_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYSTEMONE_HOME", str(tmp_path))
    config.save(config.Config(endpoint="https://api.example.test", token="s1_pat_stored"))
    monkeypatch.setenv("SYSTEMONE_TOKEN", "")
    monkeypatch.setenv("SYSTEMONE_ENDPOINT", "")

    loaded = config.load()
    assert loaded.token == "s1_pat_stored"
    assert loaded.endpoint == "https://api.example.test"
