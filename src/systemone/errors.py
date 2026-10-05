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


class RateLimited(ApiError):
    """Too many requests a minute, or a plan's decisions used up (HTTP 429).

    `retry_after` is the number of seconds the server asked to wait, when it said.
    """

    def __init__(
        self,
        status: int,
        detail: str,
        code: str = "rate_limited",
        errors: list[dict[str, str]] | None = None,
        retry_after: float | None = None,
    ):
        super().__init__(status, detail, code, errors)
        self.retry_after = retry_after


class CreditsRequired(ApiError):
    """The model runs on GPUs, which are paid from prepaid credit, and the account has
    none left (HTTP 402, code `credits_required`). Add credits in Settings → Billing;
    models on the CPU servers stay free within the plan's decisions."""
