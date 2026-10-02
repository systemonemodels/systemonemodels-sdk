"""Where the CLI keeps its token and its cache.

Both follow the platform conventions rather than hard-coding ~/.systemone, so
the cache lands somewhere a user's backup tool already knows to skip and the
token lands somewhere their sync tool already knows not to share.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

APP_NAME = "systemone"
DEFAULT_ENDPOINT = "https://api.systemonemodels.ai"
# The API's former address, and where it went: a login stored by an older version,
# or a SYSTEMONE_ENDPOINT still naming it, is read as the new one.
MOVED = {"https://api.systemonemodels.tech": DEFAULT_ENDPOINT}
ENV_TOKEN = "SYSTEMONE_TOKEN"  # noqa: S105 - the variable name, not a token
ENV_ENDPOINT = "SYSTEMONE_ENDPOINT"
# An API key for calling models (Client.decide); registry calls never use it.
ENV_API_KEY = "SYSTEMONE_API_KEY"  # noqa: S105 - the variable name, not a key


def web_url(endpoint: str) -> str:
    """The website that goes with an API endpoint, for printing links.

    api.systemonemodels.ai -> systemonemodels.ai, and dev-api.X -> dev.X.
    Anything else — a self-hosted registry, localhost — is returned unchanged
    rather than guessed at.
    """
    scheme, sep, rest = endpoint.partition("://")
    host, slash, path = rest.partition("/")
    if host.startswith("api."):
        host = host[len("api.") :]
    elif "-api." in host:
        host = host.replace("-api.", ".", 1)
    return f"{scheme}{sep}{host}{slash}{path}".rstrip("/")


def config_dir() -> Path:
    if override := os.environ.get("SYSTEMONE_HOME"):
        return Path(override).expanduser()
    if os.name == "nt":
        base = os.environ.get("APPDATA", "~")
        return Path(base).expanduser() / APP_NAME
    return Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser() / APP_NAME


def cache_dir() -> Path:
    if override := os.environ.get("SYSTEMONE_CACHE"):
        return Path(override).expanduser()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA", "~")
        return Path(base).expanduser() / APP_NAME / "cache"
    return Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / APP_NAME


@dataclass
class Config:
    endpoint: str = DEFAULT_ENDPOINT
    token: str | None = None
    username: str | None = None

    @property
    def authenticated(self) -> bool:
        return bool(self.token)


def _path() -> Path:
    return config_dir() / "config.json"


def load(environment: bool = True) -> Config:
    """Environment beats the file, so CI can authenticate without writing one.

    `environment=False` reads only what `login` stored: the token `logout` may
    revoke, which must never be one a CI job was handed in SYSTEMONE_TOKEN.
    """
    config = Config()

    path = _path()
    if path.exists():
        try:
            data = json.loads(path.read_text())
            config.endpoint = data.get("endpoint", config.endpoint)
            config.token = data.get("token")
            config.username = data.get("username")
        except (OSError, json.JSONDecodeError):
            # A corrupt config should not make the CLI unusable; the user can
            # simply log in again.
            pass

    if not environment:
        config.endpoint = _current(config.endpoint)
        return config

    # `or`, not a default argument: an exported-but-empty variable, which is
    # what `export SYSTEMONE_TOKEN=` in a CI template leaves behind, must fall
    # back to the stored value rather than blank it out.
    config.endpoint = _current(os.environ.get(ENV_ENDPOINT) or config.endpoint)
    config.token = os.environ.get(ENV_TOKEN) or config.token or None
    return config


def _current(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    return MOVED.get(endpoint, endpoint)


def save(config: Config) -> Path:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"endpoint": config.endpoint, "token": config.token, "username": config.username}

    # Written 0600 before anything is in it. Creating the file and then
    # tightening it leaves a window where the token is world-readable.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2)
    return path


def clear() -> None:
    path = _path()
    if path.exists():
        path.unlink()
