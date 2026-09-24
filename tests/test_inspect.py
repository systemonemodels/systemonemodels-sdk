"""Manifest inference from a model directory."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from systemone.inspect import build_manifest, inspect, walk


def write_export(root: Path) -> None:
    """A minimal LayaStudio export, with the three files inference reads."""
    (root / "encoder").mkdir(parents=True)
    (root / "tokenizer").mkdir()
    (root / "encoder" / "config.json").write_text('{"model_type": "bert"}')
    (root / "tokenizer" / "tokenizer.json").write_text("{}")
    (root / "model.onnx").write_bytes(b"graph")

    (root / "questions.json").write_text(
        json.dumps({"move": {"type": "choice", "criteria": {"UP": "up", "DOWN": "down"}}})
    )
    (root / "rl_agent_config.json").write_text(
        json.dumps(
            {
                "encoder": "jhu-clsp/mmBERT-base",
                "fine_tuned": {"base_model": "hub:aac6fef/laya-multilingual-mlx"},
            }
        )
    )
    (root / "export.json").write_text(
        json.dumps(
            {
                "model": "run:snake-balanced-0923",
                "target": "onnx",
                "precision": "int8",
                "ms_per_decision_cpu": 101.26,
                "test": {"accuracy_exported": 0.98, "dataset": "snake-moves-970a6be8"},
            }
        )
    )


def test_walk_skips_platform_litter(tmp_path: Path) -> None:
    write_export(tmp_path)
    (tmp_path / ".DS_Store").write_bytes(b"junk")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"junk")

    names = {p.name for p in walk(tmp_path)}
    assert ".DS_Store" not in names
    assert "x.pyc" not in names
    assert "model.onnx" in names


def test_capability_comes_from_the_question_schema(tmp_path: Path) -> None:
    """questions.json is what makes this a decision model rather than a blob."""
    write_export(tmp_path)
    assert inspect(tmp_path).capabilities == ["choice"]


def test_base_model_loses_the_hub_prefix(tmp_path: Path) -> None:
    write_export(tmp_path)
    assert inspect(tmp_path).base_model == "aac6fef/laya-multilingual-mlx"


def test_evaluation_is_lifted_from_the_export(tmp_path: Path) -> None:
    write_export(tmp_path)
    found = inspect(tmp_path)
    assert found.evaluation["decision_accuracy"] == 0.98
    assert found.evaluation["median_latency_ms"] == 101.26
    assert found.evaluation["suite"] == "snake-moves-970a6be8"


def test_summary_drops_the_internal_run_prefix(tmp_path: Path) -> None:
    write_export(tmp_path)
    summary = inspect(tmp_path).summary or ""
    assert summary.startswith("snake-balanced-0923")
    assert "run:" not in summary


def test_manifest_is_valid_shaped_yaml(tmp_path: Path) -> None:
    write_export(tmp_path)
    document = yaml.safe_load(build_manifest(inspect(tmp_path), "biplov", "snake"))

    assert document["spec_version"] == "0.1"
    assert document["namespace"] == "biplov"
    assert document["model"] == "snake"
    assert document["category"] == "system-one"
    assert document["architecture"] == "laya"
    assert document["capabilities"] == ["choice"]
    assert document["runtime"]["framework"] == "onnx"
    assert document["base_model"] == "aac6fef/laya-multilingual-mlx"


def test_an_unrecognised_directory_still_produces_a_manifest(tmp_path: Path) -> None:
    """Not every model comes from LayaStudio. The result is a starting point
    rather than a refusal."""
    (tmp_path / "weights.bin").write_bytes(b"opaque")
    found = inspect(tmp_path)

    assert not found.recognised
    document = yaml.safe_load(build_manifest(found, "someone", "thing"))
    assert document["capabilities"] == ["choice"]
    assert "evaluation" not in document


def test_a_model_is_never_its_own_base(tmp_path: Path) -> None:
    write_export(tmp_path)
    (tmp_path / "rl_agent_config.json").write_text(
        json.dumps({"fine_tuned": {"base_model": "hub:biplov/snake"}})
    )
    document = yaml.safe_load(build_manifest(inspect(tmp_path), "biplov", "snake"))
    assert "base_model" not in document
