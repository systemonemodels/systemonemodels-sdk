"""The content-addressed cache."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from systemone import cache


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYSTEMONE_CACHE", str(tmp_path / "cache"))


def store(content: bytes) -> tuple[str, Path]:
    digest = hashlib.sha256(content).hexdigest()
    staging = cache.blob_path(digest).with_suffix(".incoming")
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.write_bytes(content)
    return digest, cache.adopt(staging, digest)


def test_blobs_are_keyed_by_content() -> None:
    digest, blob = store(b"tokenizer")
    assert blob.name == digest
    assert blob.parent.name == digest[:2]
    assert cache.has_blob(digest)


def test_a_corrupted_blob_is_not_trusted() -> None:
    """Checked by content rather than existence: a truncated or edited blob
    must be fetched again rather than silently handed out."""
    digest, blob = store(b"original bytes")
    blob.chmod(0o644)
    blob.write_bytes(b"tampered")
    assert not cache.has_blob(digest)


def test_blobs_are_read_only() -> None:
    """So a snapshot link cannot be used to modify bytes other snapshots share."""
    _, blob = store(b"shared")
    if os.name != "nt":
        assert not os.access(blob, os.W_OK) or os.geteuid() == 0


def test_two_snapshots_share_one_blob(tmp_path: Path) -> None:
    _, blob = store(b"the same tokenizer in every variant")
    first = tmp_path / "a" / "tokenizer.json"
    second = tmp_path / "b" / "tokenizer.json"
    cache.link(blob, first)
    cache.link(blob, second)

    assert first.read_bytes() == second.read_bytes()
    # Hard-linked where the filesystem allows it: same inode, no extra disk.
    if os.name != "nt":
        assert first.stat().st_ino == second.stat().st_ino == blob.stat().st_ino


def test_relinking_replaces_rather_than_failing(tmp_path: Path) -> None:
    _, old = store(b"version one")
    _, new = store(b"version two")
    target = tmp_path / "model.bin"
    cache.link(old, target)
    cache.link(new, target)
    assert target.read_bytes() == b"version two"


def test_refs_record_the_resolved_version() -> None:
    cache.write_ref("biplov/snake", "0.3.0")
    assert cache.read_ref("biplov/snake") == "0.3.0"
    assert cache.read_ref("biplov/never-pulled") is None


def test_snapshot_paths_are_stable() -> None:
    """Same repository and version, same directory, every time."""
    assert cache.snapshot_root("biplov/snake", "0.3.0") == cache.snapshot_root(
        "biplov/snake", "0.3.0"
    )
    assert "biplov--snake" in str(cache.snapshot_root("biplov/snake", "0.3.0"))


def test_clear_removes_read_only_blobs() -> None:
    store(b"one")
    store(b"two")
    assert cache.size() > 0
    freed = cache.clear()
    assert freed > 0
    assert cache.size() == 0
