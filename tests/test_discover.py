"""Finding the models under a folder, and choosing among them."""

from __future__ import annotations

from pathlib import Path

import pytest

from systemone.discover import describe, discover, model_name, next_version, parse_selection


def test_finds_checkpoints_and_exports_but_not_datasets_or_environments(workspace: Path) -> None:
    found = discover(workspace)
    assert [(c.name, c.variant, c.kind) for c in found] == [
        ("guard", "mlx", "laya-run"),
        ("snake-balanced", "mlx", "laya-run"),
        ("snake-balanced", "coreml", "laya-export"),
        ("snake-balanced", "onnx-int8", "laya-export"),
    ]


def test_an_export_is_named_after_its_run_and_titled_from_it(workspace: Path) -> None:
    export = describe(workspace / "exports" / "snake-balanced-0923-005225-onnx-int8")
    assert export is not None
    assert (export.name, export.variant, export.format) == ("snake-balanced", "onnx-int8", "onnx")
    assert export.title == "Snake · balanced"


def test_a_models_own_folders_are_not_further_models(workspace: Path) -> None:
    inner = workspace / "runs" / "snake-balanced-0923-005225" / "model" / "encoder"
    (inner / "model.onnx").write_bytes(b"x")
    names = [c.path for c in discover(workspace)]
    assert inner not in names


def test_any_folder_with_weights_counts(tmp_path: Path) -> None:
    (tmp_path / "My Router").mkdir()
    (tmp_path / "My Router" / "router.gguf").write_bytes(b"x")
    [found] = discover(tmp_path)
    assert (found.name, found.variant, found.kind) == ("my-router", "gguf", "folder")


def test_the_run_timestamp_is_not_part_of_the_name() -> None:
    assert model_name("snake-balanced-multilingual-0923-005225") == "snake-balanced-multilingual"
    assert model_name("emo-en-lora-proper") == "emo-en-lora-proper"


@pytest.mark.parametrize(
    ("text", "expected"),
    [("1", [0]), ("1,3-4", [0, 2, 3]), ("4-3", [2, 3]), ("2 1 2", [1, 0]), ("all", [0, 1, 2, 3])],
)
def test_selection(text: str, expected: list[int]) -> None:
    assert parse_selection(text, 4) == expected


@pytest.mark.parametrize("text", ["0", "5", "x", "", "1-x"])
def test_bad_selection_says_why(text: str) -> None:
    with pytest.raises(ValueError, match=r"\S"):
        parse_selection(text, 4)


def test_next_version_is_the_next_minor() -> None:
    assert next_version([]) == "0.1.0"
    assert next_version(["0.1.0", "0.3.2", "v1.0.0", "latest"]) == "1.1.0"
    assert next_version(["latest"]) == "0.1.0"


def test_installed_packages_are_not_searched(tmp_path: Path) -> None:
    fixture = tmp_path / "lib" / "python3.12" / "site-packages" / "somepkg" / "tests"
    fixture.mkdir(parents=True)
    (fixture / "model.onnx").write_bytes(b"x")
    assert discover(tmp_path) == []


def test_a_hugging_face_checkpoint_is_a_model(tmp_path: Path) -> None:
    """Sharded safetensors, a config and a tokenizer: the usual HF layout."""
    repo = tmp_path / "clm-8b"
    repo.mkdir()
    (repo / "model-00001-of-00002.safetensors").write_bytes(b"x")
    (repo / "model-00002-of-00002.safetensors").write_bytes(b"x")
    (repo / "config.json").write_text("{}")
    [found] = discover(tmp_path)
    assert (found.name, found.format) == ("clm-8b", "safetensors")


def test_older_pytorch_checkpoints_count_too(tmp_path: Path) -> None:
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "pytorch_model.bin").write_bytes(b"x")
    [found] = discover(tmp_path)
    assert found.format == "pytorch"
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "data.bin").write_bytes(b"x")
    assert len(discover(tmp_path)) == 1
