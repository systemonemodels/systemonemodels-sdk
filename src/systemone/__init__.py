"""System One — client and CLI for the decision-model registry, and its inference API.

from systemone import Client

with Client() as registry:
    registry.search("routing", capability="route")

client = Client("s1_pat_...")  # an API key, or set SYSTEMONE_API_KEY
client.decide("nokia/anyjev", "Customer: where is my parcel?", {
    "intent": {"type": "choice", "instructions": "What does the customer want?",
               "criteria": ["refund", "track delivery", "cancel order"]},
})
"""

__version__ = "0.7.1"

from systemone.client import Client  # noqa: E402
from systemone.errors import (  # noqa: E402
    ApiError,
    AuthError,
    ChecksumMismatch,
    ConnectionFailed,
    CreditsRequired,
    LoginFailed,
    NotFound,
    RateLimited,
    SystemOneError,
)
from systemone.transfer import push_files, snapshot_download  # noqa: E402

__all__ = [
    "ApiError",
    "AuthError",
    "ChecksumMismatch",
    "Client",
    "ConnectionFailed",
    "CreditsRequired",
    "LoginFailed",
    "NotFound",
    "RateLimited",
    "SystemOneError",
    "__version__",
    "push_files",
    "snapshot_download",
]
