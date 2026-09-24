"""A model card for a repository that did not bring one.

The card is the first thing a visitor reads. When a folder has no README, the
facts the push already gathered — what the model decides, how it measured,
which folder holds which format — make a better first page than an empty one.
It says it was generated, and the author can replace it at any time.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from systemone.inspect import Inspected

FRAMEWORK_LABELS = {
    "mlx": "MLX (safetensors)",
    "onnx": "ONNX",
    "coreml": "Core ML",
    "gguf": "GGUF",
    "safetensors": "safetensors",
}

LAYA = "[Laya](https://github.com/NandhaKishorM/laya)"

BASE_HOSTS = {"huggingface": "https://huggingface.co", "github": "https://github.com"}


def _size(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB"):
        if n < 1000:
            return f"{n:.0f} {unit}" if unit in ("B", "kB") else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} TB"


def _percent(value: Any) -> str:
    return f"{value * 100:.1f}%" if isinstance(value, (int, float)) else ""


def _decimal(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else ""


def _format(found: Inspected) -> str:
    label = FRAMEWORK_LABELS.get(found.framework or "", found.framework or "—")
    return f"{label} {found.precision}" if found.precision else label


def _evaluation(found: Inspected) -> list[str]:
    metrics: dict[str, Any] = found.details.get("metrics") or {}
    base: dict[str, Any] = found.details.get("base_metrics") or {}
    evaluation = found.evaluation
    if not metrics and not evaluation:
        return []

    accuracy = metrics.get("accuracy", evaluation.get("decision_accuracy"))
    ece = metrics.get("ece", evaluation.get("calibration_error"))
    ci = metrics.get("ci95")
    ours = _percent(accuracy)
    if ours and ci:
        ours = f"**{ours}** [{_percent(ci[0])}–{_percent(ci[1])}]"
    elif ours:
        ours = f"**{ours}**"

    rows: list[tuple[str, str, str]] = [
        ("Decision accuracy", _percent(base.get("accuracy")), ours),
        ("Calibration error (ECE)", _decimal(base.get("ece")), _decimal(ece)),
        ("Log loss", _decimal(base.get("nll")), _decimal(metrics.get("nll"))),
        ("Brier score", _decimal(base.get("brier")), _decimal(metrics.get("brier"))),
    ]
    if (p50 := evaluation.get("median_latency_ms")) is not None:
        rows.append(("Median latency", "", f"{p50:.1f} ms"))
    if (p95 := evaluation.get("p95_latency_ms")) is not None:
        rows.append(("p95 latency", "", f"{p95:.1f} ms"))
    if metrics.get("n"):
        rows.append(("Decisions scored", str(base.get("n") or ""), str(metrics["n"])))
    rows = [row for row in rows if row[2]]

    lines = ["## Evaluation", ""]
    dataset = found.details.get("dataset") or evaluation.get("suite")
    if dataset:
        split = found.details.get("split")
        where = f"the held-out {split} split of " if split else ""
        lines += [f"Measured on {where}*{dataset}*.", ""]
    if base:
        lines += ["| Metric | Base model | This model |", "|---|---|---|"]
        lines += [f"| {name} | {theirs} | {mine} |" for name, theirs, mine in rows]
    else:
        lines += ["| Metric | Value |", "|---|---|"]
        lines += [f"| {name} | {mine} |" for name, _, mine in rows]
    if isinstance(found.details.get("fixed"), int):
        lines += [
            "",
            f"Fine-tuning fixed **{found.details['fixed']}** decisions the base model got wrong "
            f"and broke **{found.details['broken']}** it got right.",
        ]
    return [*lines, ""]


def render(found: Inspected, repo: str, parts: Sequence[tuple[str, Inspected]]) -> str:
    """Markdown for the model page. `parts` is (folder, what is in it) per folder."""
    title = found.title or repo.split("/")[-1]
    lines = [f"# {title}", ""]

    if found.architecture == "laya":
        intro = (
            f"A {LAYA} decision model. It answers the questions below in one forward pass, "
            "with calibrated probabilities and no generated text."
        )
    elif found.architecture:
        intro = f"A {found.architecture} decision model."
    else:
        intro = "A decision model."
    if found.base_model:
        host = BASE_HOSTS.get(found.base_model_source or "")
        base = f"`{found.base_model}`"
        intro += f" Fine-tuned from {f'[{base}]({host}/{found.base_model})' if host else base}."
    lines += [intro, ""]

    lines += _evaluation(found)

    folders = [(folder, part) for folder, part in parts if folder]
    if folders:
        lines += ["## Variants", "", "| Folder | Format | Size |", "|---|---|---|"]
        lines += [
            f"| `{folder}/` | {_format(part)} | {_size(part.total_bytes)} |"
            for folder, part in folders
        ]
        lines += [""]

    mlx = next((folder for folder, part in parts if part.framework == "mlx"), None)
    first = folders[0][0] if folders else None
    pull = f"systemone pull {repo}" + (f" --variant {mlx or first}" if first else "")
    lines += ["## Use it", "", "```bash", "pip install systemonemodels", pull, "```", ""]
    if found.architecture == "laya" and mlx is not None and found.questions:
        where = f', variant="{mlx}") / "{mlx}"' if mlx else ")"
        lines += [
            "```python",
            "import json",
            "import laya_mlx as laya  # pip install laya-mlx, on Apple silicon",
            "from systemone import snapshot_download",
            "",
            f'path = snapshot_download("{repo}"{where}',
            "agent = laya.load(str(path))",
            'questions = json.loads((path / "questions.json").read_text())',
            'print(agent.predict("your text here", questions)["answers"])',
            "```",
            "",
        ]

    if found.questions:
        lines += [
            "## Questions",
            "",
            "The questions it was trained to answer. Their wording is part of the model's "
            "input, so ask them as written.",
            "",
            "```json",
            json.dumps(found.questions, indent=2, ensure_ascii=False),
            "```",
            "",
        ]

    if hyperparameters := found.details.get("hyperparameters"):
        lines += [
            "## Training",
            "",
            "```json",
            json.dumps(hyperparameters, indent=2, ensure_ascii=False),
            "```",
            "",
        ]

    lines += [
        "---",
        "",
        "*Generated by `systemone push` from the files in this repository. Push a README.md, "
        "or edit the card on this page, to replace it.*",
        "",
    ]
    return "\n".join(lines)
