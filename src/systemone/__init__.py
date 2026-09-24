"""System One — client and CLI for the decision-model registry.

from systemone import Client

with Client() as registry:
    registry.search("routing", capability="route")
"""

__version__ = "0.2.0"

from systemone.client import Client  # noqa: E402
from systemone.errors import (  # noqa: E402
    ApiError,
    AuthError,
    ChecksumMismatch,
    ConnectionFailed,
    LoginFailed,
    NotFound,
    SystemOneError,
)
from systemone.transfer import push_files, snapshot_download  # noqa: E402

__all__ = [
    "ApiError",
    "AuthError",
    "ChecksumMismatch",
    "Client",
    "ConnectionFailed",
    "LoginFailed",
    "NotFound",
    "SystemOneError",
    "__version__",
    "push_files",
    "snapshot_download",
]
