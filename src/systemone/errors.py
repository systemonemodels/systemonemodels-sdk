"""Errors the CLI can explain rather than re-raise."""

from __future__ import annotations


class SystemOneError(Exception):
    """Base for anything a user could plausibly fix."""


class AuthError(SystemOneError):
    pass


class LoginFailed(AuthError):
    """The browser login was denied, expired, or could not be completed."""


class NotFound(SystemOneError):
    pass


class ConnectionFailed(SystemOneError):
    """The registry, or the storage behind it, could not be reached at all."""


class ApiError(SystemOneError):
    def __init__(
        self,
        status: int,
        detail: str,
        code: str = "error",
        errors: list[dict[str, str]] | None = None,
    ):
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.code = code
        self.errors = errors or []


class ChecksumMismatch(SystemOneError):
    pass
