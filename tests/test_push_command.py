"""`systemone push` end to end, against a registry that records what it is sent."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from systemone import cli
from systemone.config import Config
from systemone.errors import NotFound

runner = CliRunner()


class Registry:
    def __init__(self, existing: dict[str, list[str]] | None = None) -> None:
        self.config = Config(endpoint="https://api.test", token="t", username="me")
        self.existing = existing or {}
        self.created: list[tuple[str, bool]] = []
        self.published: list[dict[str, Any]] = []

    def __enter__(self) -> Registry:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def whoami(self) -> dict[str, Any]:
        return {"username": "me"}

    def versions(self, repo: str) -> list[dict[str, Any]]:
        if repo not in self.existing:
            raise NotFound("not found")
        return [{"version": v, "artifacts": []} for v in self.existing[repo]]

    def validate_manifest(self, document: str) -> dict[str, Any]:
        return {"valid": True, "issues": []}

    def create_model(self, namespace: str, name: str, document: str, private: bool) -> None:
        self.created.append((f"{namespace}/{name}", private))

    def publish_version(
        self,
        repo: str,
        version: str,
        manifest: str,
        artifacts: list[dict[str, Any]],
        notes: str | None = None,
        readme: str | None = None,
    ) -> None:
        self.published.append(
            {
                "repo": repo,
                "version": version,
                "manifest": yaml.safe_load(manifest),
                "paths": sorted(a["path"] for a in artifacts),
                "readme": readme,
            }
        )


def fake_push_files(
    registry: Any,
    repo: str,
    source: Path,
    files: list[Path],
    prefix: str = "",
    on_progress: Callable[[str, int, int, bool], None] | None = None,
    *,
    on_file: Callable[[str], None] | None = None,
    on_bytes: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
    out = []
    for path in files:
        relative = path.relative_to(source).as_posix()
        placed = f"{prefix}/{relative}" if prefix else relative
        if on_file:
            on_file(placed)
        if on_bytes:
            on_bytes(path.stat().st_size)
        out.append({"path": placed})
    return out


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> Registry:
    fake = Registry()
    monkeypatch.setattr(cli, "client", lambda: fake)
    monkeypatch.setattr(cli, "push_files", fake_push_files)
    monkeypatch.setattr(cli, "_interactive", lambda: False)
    return fake


def run_model(workspace: Path) -> Path:
    return workspace / "runs" / "snake-balanced-0923-005225" / "model"


def test_one_folder_publishes_with_its_card_and_measurements(
    workspace: Path, registry: Registry
) -> None:
    result = runner.invoke(cli.app, ["push", str(run_model(workspace)), "--repo", "me/laya-snake"])
    assert result.exit_code == 0, result.output

    assert registry.created == [("me/laya-snake", False)]
    [published] = registry.published
    assert published["repo"] == "me/laya-snake"
    assert published["version"] == "0.1.0"
    assert published["readme"].startswith("# Snake · balanced")
    assert published["manifest"]["evaluation"]["decision_accuracy"] == 0.9883
    assert published["manifest"]["base_model"] == "aac6fef/laya-multilingual-mlx"
    # Like a Hugging Face repository: the folder's files at the top.
    assert "model.safetensors" in published["paths"]


def test_a_workspace_needs_a_choice_when_nobody_can_answer(
    workspace: Path, registry: Registry
) -> None:
    result = runner.invoke(cli.app, ["push", str(workspace)])
    assert result.exit_code == 1
    assert "--all" in result.output
    assert registry.published == []


def test_all_publishes_every_model_with_variants_together(
    workspace: Path, registry: Registry
) -> None:
    result = runner.invoke(cli.app, ["push", str(workspace), "--all"])
    assert result.exit_code == 0, result.output

    assert [p["repo"] for p in registry.published] == ["me/guard", "me/snake-balanced"]
    snake = registry.published[1]
    assert "mlx/model.safetensors" in snake["paths"]
    assert "onnx-int8/model.onnx" in snake["paths"]
    assert "coreml/model.mlpackage/Manifest.json" in snake["paths"]
    guard = registry.published[0]
    assert guard["manifest"]["capabilities"] == ["classify"]
    assert guard["readme"].startswith("# Guard")


def test_picking_from_the_list(
    workspace: Path, registry: Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    result = runner.invoke(cli.app, ["push", str(workspace)], input="nine\n2-4\ny\n")
    assert result.exit_code == 0, result.output

    assert "not a number" in result.output
    [published] = registry.published
    assert published["repo"] == "me/snake-balanced"
    assert {p.split("/")[0] for p in published["paths"]} == {"mlx", "coreml", "onnx-int8"}


def test_declining_publishes_nothing(
    workspace: Path, registry: Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    result = runner.invoke(cli.app, ["push", str(run_model(workspace))], input="n\n")
    assert result.exit_code == 0
    assert registry.published == []


def test_an_existing_repository_gets_the_next_version(workspace: Path, registry: Registry) -> None:
    registry.existing["me/snake-balanced"] = ["0.1.0"]
    result = runner.invoke(cli.app, ["push", str(run_model(workspace)), "--yes"])
    assert result.exit_code == 0, result.output

    assert registry.created == []
    assert registry.published[0]["version"] == "0.2.0"


def test_a_dry_run_sends_nothing(workspace: Path, registry: Registry) -> None:
    result = runner.invoke(cli.app, ["push", str(workspace), "--all", "--dry-run"])
    assert result.exit_code == 0, result.output

    assert registry.created == []
    assert registry.published == []
    assert "systemone.yaml" in result.output
    assert "mlx/laya_finetune.json" in result.output


def test_one_repository_name_cannot_cover_several_models(
    workspace: Path, registry: Registry
) -> None:
    result = runner.invoke(cli.app, ["push", str(workspace), "--all", "--repo", "me/x"])
    assert result.exit_code == 1
    assert "--repo names one repository" in result.output


def test_a_folder_without_models_needs_a_name(tmp_path: Path, registry: Registry) -> None:
    (tmp_path / "notes.txt").write_text("hello")
    refused = runner.invoke(cli.app, ["push", str(tmp_path)])
    assert refused.exit_code == 1
    assert "no model found" in refused.output

    published = runner.invoke(cli.app, ["push", str(tmp_path), "--repo", "me/notes"])
    assert published.exit_code == 0, published.output
    assert registry.published[0]["paths"] == ["notes.txt"]


def test_organizations_are_a_flag_away(workspace: Path, registry: Registry) -> None:
    result = runner.invoke(
        cli.app, ["push", str(run_model(workspace)), "--namespace", "Laya", "--yes"]
    )
    assert result.exit_code == 0, result.output
    assert registry.published[0]["repo"] == "laya/snake-balanced"


def test_a_missing_card_file_is_a_clear_error(workspace: Path, registry: Registry) -> None:
    result = runner.invoke(
        cli.app, ["push", str(run_model(workspace)), "--readme", str(workspace / "nope.md")]
    )
    assert result.exit_code == 2
    # The message sits in a bordered panel that wraps long paths, and Typer
    # colours it when it detects CI; read it flat and plain.
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    flat = " ".join(plain.replace("│", " ").split())
    assert "Invalid value for '--readme'" in flat
    assert registry.published == []


def test_sizes_are_decimal_like_the_progress_bar() -> None:
    assert cli.human(999) == "999 B"
    assert cli.human(678_200_000) == "678.2 MB"
