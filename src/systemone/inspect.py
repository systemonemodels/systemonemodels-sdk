"""Read a model directory and work out what it is.

A LayaStudio export already records everything a manifest needs — questions.json
declares the decision primitive, rl_agent_config.json records the base model and
the encoder, export.json carries accuracy and latency. Asking the author to
retype it would be asking them to make a mistake.

Anything not recognised still publishes; it just needs a manifest written by
hand, and `systemone push --manifest` takes one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

IGNORED = {".DS_Store", "Thumbs.db", ".gitattributes"}
IGNORED_DIRS = {".git", "__pycache__", ".ipynb_checkpoints"}

QUESTION_TYPE_TO_CAPABILITY = {
    "choice": "choice",
    "score": "score",
    "rank": "rank",
    "classify": "classify",
    "extract": "extract",
}


@dataclass
class Inspected:
    files: list[Path] = field(default_factory=list)
    total_bytes: int = 0
    architecture: str | None = None
    capabilities: list[str] = field(default_factory=list)
    base_model: str | None = None
    summary: str | None = None
    tags: list[str] = field(default_factory=list)
    framework: str | None = None
    evaluation: dict[str, Any] = field(default_factory=dict)

    @property
    def recognised(self) -> bool:
        return bool(self.capabilities)


def walk(source: Path) -> list[Path]:
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file()
        and path.name not in IGNORED
        and not any(part in IGNORED_DIRS for part in path.relative_to(source).parts)
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def inspect(source: Path) -> Inspected:
    found = Inspected(files=walk(source))
    found.total_bytes = sum(p.stat().st_size for p in found.files)

    export = _read_json(source / "export.json")
    agent = _read_json(source / "rl_agent_config.json")
    questions = _read_json(source / "questions.json")

    found.capabilities = sorted(
        {
            QUESTION_TYPE_TO_CAPABILITY[q["type"]]
            for q in questions.values()
            if isinstance(q, dict) and q.get("type") in QUESTION_TYPE_TO_CAPABILITY
        }
    )

    if agent or export:
        found.architecture = "laya"

    # LayaStudio writes "hub:owner/name"; the registry wants "owner/name".
    base = (agent.get("fine_tuned") or {}).get("base_model")
    if isinstance(base, str):
        found.base_model = base.split(":", 1)[-1]

    found.framework = export.get("target")
    found.tags = sorted({t for t in (export.get("target"), export.get("precision")) if t})

    if model_name := export.get("model"):
        # "run:<name>" is an internal LayaStudio detail.
        clean = str(model_name).split(":", 1)[-1]
        target = export.get("target", "export")
        precision = export.get("precision", "fp32")
        found.summary = f"{clean} exported to {target} ({precision})."

    test = export.get("test") or {}
    evaluation: dict[str, Any] = {}
    if suite := test.get("dataset"):
        evaluation["suite"] = suite
    if (accuracy := test.get("accuracy_exported")) is not None:
        evaluation["decision_accuracy"] = accuracy
    if (latency := export.get("ms_per_decision_cpu")) is not None:
        evaluation["median_latency_ms"] = latency
    found.evaluation = evaluation

    return found


def build_manifest(
    found: Inspected, namespace: str, model: str, license_: str = "apache-2.0"
) -> str:
    """Render a manifest from what was found. Fields absent are left out."""
    document: dict[str, Any] = {
        "spec_version": "0.1",
        "model": model,
        "namespace": namespace,
        "category": "system-one",
        "architecture": found.architecture or "laya",
        "license": license_,
    }
    if found.summary:
        document["summary"] = found.summary
    if found.base_model and found.base_model != f"{namespace}/{model}":
        document["base_model"] = found.base_model

    document["capabilities"] = found.capabilities or ["choice"]
    if found.tags:
        document["tags"] = found.tags
    document["runtime"] = {"framework": found.framework or "onnx", "hardware": "cpu"}
    if found.evaluation:
        document["evaluation"] = found.evaluation

    return yaml.safe_dump(document, sort_keys=False, default_flow_style=False, allow_unicode=True)
