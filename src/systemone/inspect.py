"""Read a model directory and work out what it is.

A LayaStudio export already records everything a manifest needs — questions.json
declares the decision primitive, rl_agent_config.json records the base model and
the encoder, export.json carries accuracy and latency. A training run records
more: run.json names it, eval.json holds the held-out evaluation, and a model
card's front matter carries its licence and tags. Asking the author to retype
any of it would be asking them to make a mistake.

Anything not recognised still publishes; it just needs a manifest written by
hand, and `systemone push --manifest` takes one.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from systemone.discover import weights_format

IGNORED = {".DS_Store", "Thumbs.db", ".gitattributes"}
IGNORED_DIRS = {".git", "__pycache__", ".ipynb_checkpoints"}

QUESTION_TYPE_TO_CAPABILITY = {
    "choice": "choice",
    "score": "score",
    "rank": "rank",
    "classify": "classify",
    "extract": "extract",
    "route": "route",
    # Laya's yes/no question: the probability that the answer is yes.
    "noul": "classify",
}

# The licences spec 0.1 accepts. Anything else is published as "other", with
# the name the author gave kept in license_name.
KNOWN_LICENSES = frozenset(
    {
        "apache-2.0",
        "mit",
        "bsd-2-clause",
        "bsd-3-clause",
        "gpl-3.0",
        "agpl-3.0",
        "lgpl-3.0",
        "mpl-2.0",
        "cc-by-4.0",
        "cc-by-sa-4.0",
        "cc-by-nc-4.0",
        "cc0-1.0",
        "openrail",
        "proprietary",
        "other",
    }
)

MAX_TAGS = 20
_SLUG = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}[a-z0-9]$")
_REPO = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}[a-z0-9]/[a-z0-9][a-z0-9._-]{0,62}[a-z0-9]$")

FULL_PRECISION = {"float", "float32", "fp32", "full"}

FORMAT_FRAMEWORK = {
    "onnx": "onnx",
    "coreml": "coreml",
    "gguf": "gguf",
    "safetensors": "safetensors",
}


@dataclass
class Inspected:
    files: list[Path] = field(default_factory=list)
    total_bytes: int = 0
    architecture: str | None = None
    capabilities: list[str] = field(default_factory=list)
    base_model: str | None = None
    summary: str | None = None
    title: str | None = None
    tags: list[str] = field(default_factory=list)
    framework: str | None = None
    precision: str | None = None
    hardware: str | None = None
    license: str | None = None
    license_name: str | None = None
    evaluation: dict[str, Any] = field(default_factory=dict)
    # The model card: README.md with its front matter removed.
    readme: str | None = None
    questions: dict[str, Any] = field(default_factory=dict)
    # Numbers a generated card can show that the manifest has no field for:
    # the base model's scores, confidence intervals, training settings.
    details: dict[str, Any] = field(default_factory=dict)
    # "laya-run" for a trained checkpoint, "laya-export" for an export of one.
    origin: str | None = None

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
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Separate a model card's YAML front matter from its body.

    Hugging Face cards open with a metadata block between two `---` lines. It is
    data rather than prose: rendered as Markdown it would show up as a stray rule
    and a paragraph of YAML.
    """
    text = text.lstrip("﻿")
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, text
    for index in range(1, len(lines)):
        if lines[index].strip() in ("---", "..."):
            try:
                meta = yaml.safe_load("".join(lines[1:index])) or {}
            except yaml.YAMLError:
                meta = {}
            body = "".join(lines[index + 1 :]).lstrip("\r\n")
            return (meta if isinstance(meta, dict) else {}), body
    return {}, text


def slugs(values: Iterable[Any]) -> list[str]:
    """Keep the values that are valid tags, lowercased and without repeats."""
    out: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        tag = value.strip().lower()
        if _SLUG.match(tag) and tag not in out:
            out.append(tag)
    return out


def repo_ref(value: Any) -> str | None:
    """A base model reference as the registry wants it: "owner/name".

    LayaStudio writes "hub:owner/name", and a Hugging Face card may list several
    base models; the first is the one fine-tuned from.
    """
    if isinstance(value, list):
        value = value[0] if value else None
    if not isinstance(value, str):
        return None
    ref = value.split(":", 1)[-1].strip().lower()
    return ref if _REPO.match(ref) else None


def _number(value: Any, low: float = 0.0, high: float = math.inf) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and low <= number <= high else None


def _license(meta: dict[str, Any]) -> tuple[str | None, str | None]:
    value = meta.get("license")
    if not isinstance(value, str) or not value.strip():
        return None, None
    key = value.strip().lower()
    if key in KNOWN_LICENSES:
        name = meta.get("license_name")
        return key, name if key == "other" and isinstance(name, str) else None
    return "other", value.strip()


def run_dir(source: Path, export: dict[str, Any]) -> Path | None:
    """The LayaStudio training run a model folder came from, if it is on disk.

    A checkpoint is `runs/<run id>/model`; an export records its run in
    export.json and sits in `exports/`, beside `runs/`.
    """
    if export:
        run_id = str(export.get("model", "")).split(":", 1)[-1]
        candidate = source.parent.parent / "runs" / run_id
        return candidate if run_id and candidate.is_dir() else None
    if (source.parent / "run.json").is_file():
        return source.parent
    return None


def _metrics(overall: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("accuracy", "ece", "nll", "brier"):
        if (value := _number(overall.get(key))) is not None:
            out[key] = value
    interval = overall.get("accuracy_ci95")
    if isinstance(interval, list) and len(interval) == 2:
        low, high = _number(interval[0], 0, 1), _number(interval[1], 0, 1)
        if low is not None and high is not None:
            out["ci95"] = [low, high]
    if isinstance(overall.get("n"), int):
        out["n"] = overall["n"]
    return out


def _run_evaluation(
    evaluation: dict[str, Any], run: dict[str, Any], comparison: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The manifest's evaluation block and the card's extra numbers, from a run."""
    overall = evaluation.get("overall") or {}
    latency = evaluation.get("latency_ms") or {}
    manifest: dict[str, Any] = {}
    suite = run.get("dataset_name") or evaluation.get("dataset")
    if isinstance(suite, str) and suite:
        manifest["suite"] = suite
    if (accuracy := _number(overall.get("accuracy"), 0, 1)) is not None:
        manifest["decision_accuracy"] = round(accuracy, 4)
    if (ece := _number(overall.get("ece"), 0, 1)) is not None:
        manifest["calibration_error"] = round(ece, 4)
    if (p50 := _number(latency.get("p50"))) is not None:
        manifest["median_latency_ms"] = round(p50, 2)
    if (p95 := _number(latency.get("p95"))) is not None:
        manifest["p95_latency_ms"] = round(p95, 2)

    details: dict[str, Any] = {}
    if manifest:
        details["metrics"] = _metrics(overall)
        if isinstance(suite, str) and suite:
            details["dataset"] = suite
        if isinstance(evaluation.get("split"), str):
            details["split"] = evaluation["split"]
    base = comparison.get("base") or {}
    if isinstance(base, dict) and base.get("overall"):
        details["base_metrics"] = _metrics(base["overall"])
    paired = (comparison.get("paired") or {}).get("overall") or {}
    if isinstance(paired.get("fixed"), int) and isinstance(paired.get("broken"), int):
        details["fixed"], details["broken"] = paired["fixed"], paired["broken"]
    if isinstance(run.get("hyperparameters"), dict):
        details["hyperparameters"] = run["hyperparameters"]
    return manifest, details


def _export_evaluation(export: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    """What an export measured about itself: the numbers for that exact file."""
    test = export.get("test") or {}
    manifest: dict[str, Any] = {}
    suite = test.get("dataset")
    if suite and suite == run.get("dataset") and run.get("dataset_name"):
        suite = run["dataset_name"]
    if isinstance(suite, str) and suite:
        manifest["suite"] = suite
    if (accuracy := _number(test.get("accuracy_exported"), 0, 1)) is not None:
        manifest["decision_accuracy"] = round(accuracy, 4)
    if (latency := _number(export.get("ms_per_decision_cpu"))) is not None:
        manifest["median_latency_ms"] = round(latency, 2)
    return manifest


def _is_mlx(meta: dict[str, Any], agent: dict[str, Any], base: str | None) -> bool:
    hints = (
        str(meta.get("library_name") or ""),
        str((agent.get("fine_tuned") or {}).get("tool") or ""),
        base or "",
    )
    return any("mlx" in hint for hint in hints)


def _title(run: dict[str, Any], readme: str | None) -> str | None:
    if isinstance(run.get("name"), str) and run["name"].strip():
        return str(run["name"]).strip()
    for line in (readme or "").splitlines():
        if line.startswith("# "):
            return line[2:].strip() or None
    return None


def _summary(
    title: str | None,
    questions: dict[str, Any],
    base: str | None,
    export: dict[str, Any],
) -> str | None:
    parts: list[str] = []
    if title:
        parts.append(title.rstrip(".") + ".")
    elif name := export.get("model"):
        # "run:<name>" is an internal LayaStudio detail.
        clean = str(name).split(":", 1)[-1]
        target = export.get("target", "export")
        precision = export.get("precision", "fp32")
        parts.append(f"{clean} exported to {target} ({precision}).")
    first = next((q for q in questions.values() if isinstance(q, dict)), {})
    if isinstance(first.get("instructions"), str) and first["instructions"].strip():
        parts.append(first["instructions"].strip())
    if base:
        parts.append(f"Fine-tuned from {base}.")
    text = " ".join(parts)
    if len(text) > 280:
        text = text[:279].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return text or None


def inspect(source: Path) -> Inspected:
    found = Inspected(files=walk(source))
    found.total_bytes = sum(p.stat().st_size for p in found.files)

    export = _read_json(source / "export.json")
    agent = _read_json(source / "rl_agent_config.json")
    questions = _read_json(source / "questions.json")
    run_root = run_dir(source, export)
    run = _read_json(run_root / "run.json") if run_root else {}

    # The folder's own card first; an export without one borrows its run's —
    # the licence and base model hold for both, the tags describe the checkpoint.
    card = source / "README.md"
    borrowed = not card.is_file() and run_root is not None
    if borrowed and run_root is not None:
        card = run_root / "model" / "README.md"
    meta: dict[str, Any] = {}
    if card.is_file():
        try:
            meta, body = split_front_matter(card.read_text(errors="replace"))
        except OSError:
            body = ""
        found.readme = body if body.strip() else None

    found.questions = {k: v for k, v in questions.items() if isinstance(v, dict)}
    found.capabilities = sorted(
        {
            QUESTION_TYPE_TO_CAPABILITY[q["type"]]
            for q in found.questions.values()
            if q.get("type") in QUESTION_TYPE_TO_CAPABILITY
        }
    )

    laya = bool(agent or export or run)
    if laya:
        found.architecture = "laya"
        found.hardware = "either"
        found.origin = "laya-export" if export else "laya-run" if run_root else None

    # The card's author chose its base model deliberately; LayaStudio's own
    # records are the fallback.
    found.base_model = (
        repo_ref(meta.get("base_model"))
        or repo_ref((agent.get("fine_tuned") or {}).get("base_model"))
        or repo_ref(run.get("base_model"))
    )
    found.license, found.license_name = _license(meta)

    fmt = weights_format(source)
    if isinstance(export.get("target"), str):
        found.framework = export["target"]
    elif fmt == "safetensors" and laya and _is_mlx(meta, agent, found.base_model):
        found.framework = "mlx"
    elif fmt:
        found.framework = FORMAT_FRAMEWORK[fmt]
    # Full precision is the unremarkable case; only a reduced one is worth a tag.
    precision = export.get("precision")
    if isinstance(precision, str) and precision.lower() not in FULL_PRECISION:
        found.precision = precision

    card_tags = [] if borrowed else meta.get("tags") or []
    if not isinstance(card_tags, list):
        card_tags = [card_tags]
    found.tags = slugs([*card_tags, found.framework, found.precision])[:MAX_TAGS]
    found.title = _title(run, found.readme)
    found.summary = _summary(found.title, found.questions, found.base_model, export)

    evaluation = _read_json(run_root / "eval.json") if run_root else {}
    comparison = _read_json(run_root / "comparison.json") if run_root else {}
    found.evaluation, found.details = _run_evaluation(evaluation, run, comparison)

    # An export's own measurements describe that exact file — its accuracy after
    # quantising, its latency on a CPU — so they replace the run's rather than
    # sit beside them: a table mixing the two would compare different things.
    if export and (own := _export_evaluation(export, run)):
        found.evaluation = own
        found.details = {k: v for k, v in found.details.items() if k == "hyperparameters"}
        rows = (export.get("test") or {}).get("rows")
        if isinstance(rows, int) and not isinstance(rows, bool):
            found.details["metrics"] = {"n": rows}

    return found


def build_manifest(
    found: Inspected, namespace: str, model: str, license_: str | None = None
) -> str:
    """Render a manifest from what was found. Fields absent are left out."""
    licence = (license_ or found.license or "apache-2.0").strip().lower()
    document: dict[str, Any] = {
        "spec_version": "0.1",
        "model": model,
        "namespace": namespace,
        "category": "system-one",
        "architecture": found.architecture or "laya",
        "license": licence,
    }
    if licence == "other" and found.license_name and not license_:
        document["license_name"] = found.license_name
    if found.summary:
        document["summary"] = found.summary
    if found.base_model and found.base_model != f"{namespace}/{model}":
        document["base_model"] = found.base_model

    document["capabilities"] = found.capabilities or ["choice"]
    if found.tags:
        document["tags"] = found.tags[:MAX_TAGS]
    document["runtime"] = {
        "framework": found.framework or "other",
        "hardware": found.hardware or "cpu",
    }
    if found.evaluation:
        document["evaluation"] = found.evaluation

    return yaml.safe_dump(document, sort_keys=False, default_flow_style=False, allow_unicode=True)
