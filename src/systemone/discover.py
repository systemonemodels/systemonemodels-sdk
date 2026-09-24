"""Find the models worth publishing under a directory.

Point `systemone push` at a Laya Studio workspace — or any folder — and it
finds every model inside, however deep: exported variants (`exports/<run>-onnx-int8`),
trained checkpoints (`runs/<run>/model`), or any directory that holds weights.

A directory is a model when it directly contains weights: `model.onnx`,
`model.safetensors`, a `.gguf` file or a Core ML `.mlpackage`. The search stops
there — a model's own `encoder/` or `tokenizer/` folders are part of it, not
further models — and never enters datasets, virtual environments or caches.

Names come from what Laya Studio records: an export's `export.json` names the
training run it came from, and the run's own folder name is the model. Exports
of one run become variants of one repository, which is how the registry keeps
them: one repository, one folder per variant.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WEIGHT_FILES = ("model.onnx", "model.safetensors")
WEIGHT_SUFFIXES = (".gguf",)
WEIGHT_DIRS = (".mlpackage",)
SKIP_DIRS = {
    ".git",
    ".hg",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".cache",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "datasets",
    "dist",
    "build",
    # Installed packages ship test fixtures with weights in them, and macOS keeps
    # application data under ~/Library: neither is a model someone is publishing.
    "site-packages",
    "Library",
}
MAX_DEPTH = 6

# Laya Studio suffixes its run ids with a timestamp: "snake-balanced-multilingual-0923-005225".
_TIMESTAMP = re.compile(r"-\d{4}-\d{6}$")
_UNSAFE = re.compile(r"[^a-z0-9._-]+")


@dataclass(frozen=True)
class Candidate:
    path: Path
    name: str  # suggested repository name
    variant: str  # folder inside the repository, e.g. "onnx-int8"
    format: str  # onnx | coreml | safetensors | gguf
    size_bytes: int
    title: str | None = None  # a human name, when the workspace recorded one
    kind: str = "folder"  # laya-export | laya-run | folder

    def relative_to(self, root: Path) -> str:
        try:
            return self.path.relative_to(root).as_posix() or "."
        except ValueError:
            return str(self.path)


def slug(value: str) -> str:
    cleaned = _UNSAFE.sub("-", value.strip().lower()).strip("-._")
    return cleaned[:64] or "model"


def model_name(run_id: str) -> str:
    return slug(_TIMESTAMP.sub("", run_id))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def weights_format(directory: Path) -> str | None:
    """The format of the weights directly inside `directory`, or None."""
    try:
        children = list(directory.iterdir())
    except OSError:
        return None
    names = {child.name for child in children}
    if "model.onnx" in names:
        return "onnx"
    if any(c.is_dir() and c.name.endswith(WEIGHT_DIRS) for c in children):
        return "coreml"
    if "model.safetensors" in names:
        return "safetensors"
    if any(c.is_file() and c.name.endswith(WEIGHT_SUFFIXES) for c in children):
        return "gguf"
    return None


def _size(directory: Path) -> int:
    total = 0
    for path in directory.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def describe(directory: Path) -> Candidate | None:
    """What kind of model `directory` is, or None if it holds no weights."""
    fmt = weights_format(directory)
    if fmt is None:
        return None
    size = _size(directory)

    export = _read_json(directory / "export.json")
    if export:
        run_id = str(export.get("model", "")).split(":", 1)[-1] or directory.name
        # The export folder is "<run id>-<target>[-<precision>]"; the tail is the variant.
        if directory.name.startswith(f"{run_id}-"):
            variant = directory.name[len(run_id) + 1 :]
        else:
            variant = "-".join(p for p in (export.get("target"), export.get("precision")) if p)
        run = _read_json(directory.parent.parent / "runs" / run_id / "run.json")
        return Candidate(
            path=directory,
            name=model_name(run_id),
            variant=slug(variant or fmt),
            format=fmt,
            size_bytes=size,
            title=run.get("name"),
            kind="laya-export",
        )

    # A trained checkpoint: runs/<run id>/model, weights from laya-mlx.
    run = _read_json(directory.parent / "run.json")
    if run or (directory.name == "model" and (directory / "rl_agent_config.json").exists()):
        run_id = str(run.get("id") or directory.parent.name)
        return Candidate(
            path=directory,
            name=model_name(run_id),
            variant="mlx" if fmt == "safetensors" else fmt,
            format=fmt,
            size_bytes=size,
            title=run.get("name"),
            kind="laya-run",
        )

    name = directory.parent.name if directory.name == "model" else directory.name
    return Candidate(path=directory, name=slug(name), variant=fmt, format=fmt, size_bytes=size)


def discover(root: Path, max_depth: int = MAX_DEPTH) -> list[Candidate]:
    """Every model under `root`, without entering any model's own folders.

    Sorted so a model's variants sit together, the trained checkpoint first.
    """
    root = root.resolve()
    found: list[Candidate] = []
    frontier = [(root, 0)]
    while frontier:
        directory, depth = frontier.pop(0)
        candidate = describe(directory)
        if candidate is not None:
            found.append(candidate)
            continue
        if depth >= max_depth:
            continue
        try:
            children = sorted(p for p in directory.iterdir() if p.is_dir())
        except OSError:
            continue
        for child in children:
            if (
                child.name in SKIP_DIRS
                or child.name.startswith(".")
                or child.name.endswith(WEIGHT_DIRS)
            ):
                continue
            frontier.append((child, depth + 1))
    return sorted(found, key=lambda c: (c.name, c.kind != "laya-run", c.variant, str(c.path)))


def parse_selection(text: str, count: int) -> list[int]:
    """Parse "1,3-5", "all" or "*" into zero-based indexes. Raises ValueError."""
    text = text.strip().lower()
    if text in ("all", "*", "a"):
        return list(range(count))
    chosen: list[int] = []
    for part in filter(None, (p.strip() for p in text.replace(" ", ",").split(","))):
        start, dash, end = part.partition("-")
        try:
            first, last = (int(start), int(end)) if dash else (int(part), int(part))
        except ValueError:
            raise ValueError(f"'{part}' is not a number or a range like 3-5") from None
        if first > last:
            first, last = last, first
        numbers = range(first, last + 1)
        for n in numbers:
            if not 1 <= n <= count:
                raise ValueError(f"{n} is not in the list")
            if n - 1 not in chosen:
                chosen.append(n - 1)
    if not chosen:
        raise ValueError("nothing chosen")
    return chosen


def next_version(existing: list[str]) -> str:
    """The next minor version after the highest semantic version in `existing`."""
    best: tuple[int, int, int] | None = None
    for version in existing:
        match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", version.strip())
        if match:
            parsed = (int(match[1]), int(match[2]), int(match[3]))
            best = parsed if best is None or parsed > best else best
    if best is None:
        return "0.1.0"
    return f"{best[0]}.{best[1] + 1}.0"
