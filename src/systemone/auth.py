"""Browser login: the device authorization grant (RFC 8628).

The CLI asks the registry for a pair of codes, shows the short one, and polls
with the long one while the person approves it on the website. No password
passes through the terminal, and it works over SSH: the browser can be on any
machine, as long as the person can type eight letters into it.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from typing import Any

from systemone.client import Client
from systemone.errors import ApiError, LoginFailed

SLOW_DOWN_STEP = 5


def device_login(
    client: Client,
    client_name: str,
    on_code: Callable[[dict[str, Any]], None],
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Run the whole flow and return `{token, token_name, username}`.

    `on_code` is called once with the codes and URLs, so the caller decides how
    to show them. `sleep` and `clock` exist so tests do not wait in real time.
    """
    started = client.start_device_login(client_name)
    on_code(started)

    interval = float(started["interval"])
    deadline = clock() + float(started["expires_in"])
    while clock() < deadline:
        sleep(interval)
        try:
            return client.poll_device_login(started["device_code"])
        except ApiError as exc:
            if exc.code == "authorization_pending":
                continue
            if exc.code == "slow_down":
                interval += SLOW_DOWN_STEP
                continue
            if exc.code in ("access_denied", "expired_token"):
                raise LoginFailed(exc.detail) from exc
            raise
    raise LoginFailed("The code expired before it was approved. Run `systemone login` again.")


def can_open_browser() -> bool:
    """Whether opening a browser here would reach the person at the keyboard.

    Over SSH it would open on the remote machine, if at all — and on a Linux
    box without a display, Python's webbrowser falls back to a text browser
    that takes over the terminal the login is running in.
    """
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return False
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True
