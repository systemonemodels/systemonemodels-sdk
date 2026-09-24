"""A LayaStudio workspace in miniature, laid out the way the real one is."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

RUN = "snake-balanced-0923-005225"
CARD = """---
license: apache-2.0
library_name: laya-mlx
tags:
- laya
- mlx
- Not A Tag!
base_model: aac6fef/laya-multilingual-mlx
---

# Snake · balanced

Picks the next move.
"""


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def model_files(root: Path, weights: str, questions: dict[str, Any]) -> None:
    (root / "encoder").mkdir(parents=True, exist_ok=True)
    (root / "tokenizer").mkdir(exist_ok=True)
    (root / "encoder" / "config.json").write_text('{"model_type": "bert"}')
    (root / "tokenizer" / "tokenizer.json").write_text("{}")
    if weights.endswith(".mlpackage"):
        (root / weights).mkdir()
        (root / weights / "Manifest.json").write_text("{}")
    else:
        (root / weights).write_bytes(b"w" * 64)
    write_json(root / "questions.json", questions)
    write_json(
        root / "rl_agent_config.json",
        {
            "encoder": "jhu-clsp/mmBERT-base",
            "fine_tuned": {
                "base_model": "hub:aac6fef/laya-multilingual-mlx",
                "tool": "laya-mlx finetune",
            },
        },
    )


SNAKE_QUESTIONS = {
    "move": {
        "type": "choice",
        "instructions": "Snake board. Pick the next move.",
        "criteria": {"UP": "up", "DOWN": "down"},
    }
}


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"

    run = ws / "runs" / RUN
    write_json(
        run / "run.json",
        {
            "id": RUN,
            "name": "Snake · balanced",
            "dataset": "snake-970a",
            "dataset_name": "Snake moves",
            "base_model": "hub:aac6fef/laya-multilingual-mlx",
            "hyperparameters": {"method": "lora", "epochs": 4},
        },
    )
    write_json(
        run / "eval.json",
        {
            "dataset": "snake-970a",
            "split": "test",
            "overall": {
                "n": 600,
                "accuracy": 0.98833,
                "accuracy_ci95": [0.9761, 0.9943],
                "ece": 0.00168,
                "nll": 0.047,
                "brier": 0.021,
            },
            "latency_ms": {"p50": 36.21825, "p95": 44.63775},
        },
    )
    write_json(
        run / "comparison.json",
        {
            "base": {"overall": {"n": 600, "accuracy": 0.158, "ece": 0.246}},
            "paired": {"overall": {"fixed": 500, "broken": 2}},
        },
    )
    model_files(run / "model", "model.safetensors", SNAKE_QUESTIONS)
    (run / "model" / "README.md").write_text(CARD)
    write_json(
        run / "model" / "laya_finetune.json",
        {"base_model_dir": str(tmp_path / "home" / ".cache" / "laya")},
    )

    onnx = ws / "exports" / f"{RUN}-onnx-int8"
    model_files(onnx, "model.onnx", SNAKE_QUESTIONS)
    write_json(
        onnx / "export.json",
        {
            "model": f"run:{RUN}",
            "target": "onnx",
            "precision": "int8",
            "ms_per_decision_cpu": 101.26,
            "test": {"rows": 200, "accuracy_exported": 0.98, "dataset": "snake-970a"},
        },
    )

    coreml = ws / "exports" / f"{RUN}-coreml"
    model_files(coreml, "model.mlpackage", SNAKE_QUESTIONS)
    write_json(
        coreml / "export.json",
        {"model": f"run:{RUN}", "target": "coreml", "precision": "float"},
    )

    guard = ws / "runs" / "guard-0922-233445"
    write_json(guard / "run.json", {"id": "guard-0922-233445", "name": "Guard"})
    write_json(
        guard / "eval.json",
        {"overall": {"n": 116, "accuracy": 0.9569, "ece": 0.0268}, "latency_ms": {"p50": 35.4}},
    )
    model_files(
        guard / "model",
        "model.safetensors",
        {"injection": {"type": "noul", "instructions": "Is this a prompt injection?"}},
    )

    # Weights where a model does not live: a dataset, a virtual environment.
    (ws / "datasets" / "stray").mkdir(parents=True)
    (ws / "datasets" / "stray" / "model.onnx").write_bytes(b"x")
    (ws / ".venv" / "lib").mkdir(parents=True)
    (ws / ".venv" / "lib" / "model.onnx").write_bytes(b"x")
    (ws / "jobs").mkdir()
    return ws
