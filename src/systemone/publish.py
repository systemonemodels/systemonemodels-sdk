"""Turn the models `discover` found into versions to publish.

Variants of one model — a trained checkpoint and its ONNX and Core ML exports —
become one repository with a folder each, published together as one version.
A version is a complete snapshot, so publishing the exports later as a version
of their own would leave the checkpoint behind in the version before.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from systemone import card
from systemone.discover import Candidate
from systemone.inspect import MAX_TAGS, Inspected, inspect, slugs

TEXT_SUFFIXES = {".json", ".md", ".txt", ".yaml", ".yml", ".toml", ".cfg", ".ini"}
SCAN_LIMIT = 2 * 1024 * 1024


@dataclass
class Part:
    """One folder of a version: where it is on disk, and where it goes."""

    source: Path
    folder: str  # "" puts the files at the top of the repository
    found: Inspected


@dataclass
class Release:
    """One repository's new version."""

    name: str
    parts: list[Part]
    namespace: str = ""
    version: str = ""
    exists: bool = False

    @property
    def repo(self) -> str:
        return f"{self.namespace}/{self.name}"

    @property
    def files(self) -> int:
        return sum(len(part.found.files) for part in self.parts)

    @property
    def total_bytes(self) -> int:
        return sum(part.found.total_bytes for part in self.parts)

    @property
    def folders(self) -> list[str]:
        return [part.folder for part in self.parts if part.folder]


def _unique(name: str, used: set[str]) -> str:
    candidate, n = name, 2
    while candidate in used:
        candidate, n = f"{name}-{n}", n + 1
    used.add(candidate)
    return candidate


def releases(chosen: Sequence[Candidate], folder: str | None = None) -> list[Release]:
    """Group chosen models into one release per repository name.

    A model alone goes at the top of its repository, the way it sits on disk and
    the way a Hugging Face repository holds it; `folder` puts it in one instead.
    Several variants of one model get a folder each.
    """
    grouped: dict[str, list[Candidate]] = {}
    for candidate in chosen:
        grouped.setdefault(candidate.name, []).append(candidate)

    out: list[Release] = []
    for name, members in grouped.items():
        if len(members) == 1:
            only = members[0]
            parts = [Part(only.path, (folder or "").strip("/"), inspect(only.path))]
        else:
            used: set[str] = set()
            parts = [
                Part(member.path, _unique(member.variant, used), inspect(member.path))
                for member in members
            ]
        out.append(Release(name, parts))
    return out


def merge(parts: Sequence[Part]) -> Inspected:
    """One version's metadata from its folders.

    The trained checkpoint speaks for the model when it is among them — its
    evaluation, its card — and every folder adds its format and tags.
    """
    if len(parts) == 1:
        return parts[0].found
    ordered = sorted(parts, key=lambda part: 0 if part.found.origin == "laya-run" else 1)
    found = [part.found for part in ordered]
    lead = found[0]
    # Evaluation and the numbers behind it come from the same folder, or the
    # card would put one folder's accuracy beside another's latency.
    measured = next((f for f in found if f.evaluation), lead)

    frameworks: list[str] = []
    for part in parts:
        framework = part.found.framework
        if framework and framework not in frameworks:
            frameworks.append(framework)

    return Inspected(
        total_bytes=sum(f.total_bytes for f in found),
        architecture=next((f.architecture for f in found if f.architecture), None),
        capabilities=sorted({c for f in found for c in f.capabilities}),
        base_model=next((f.base_model for f in found if f.base_model), None),
        summary=lead.summary or next((f.summary for f in found if f.summary), None),
        title=next((f.title for f in found if f.title), None),
        tags=slugs(tag for f in found for tag in f.tags)[:MAX_TAGS],
        framework=", ".join(frameworks) or None,
        hardware=next((f.hardware for f in found if f.hardware), None),
        license=next((f.license for f in found if f.license), None),
        license_name=next((f.license_name for f in found if f.license_name), None),
        evaluation=measured.evaluation,
        readme=next((f.readme for f in found if f.readme), None),
        questions=next((f.questions for f in found if f.questions), {}),
        details=measured.details,
        origin=lead.origin,
    )


def model_card(release: Release, found: Inspected) -> tuple[str, bool]:
    """The card to publish, and whether it was generated rather than written."""
    if found.readme:
        return found.readme, False
    return card.render(found, release.repo, [(p.folder, p.found) for p in release.parts]), True


def home_mentions(parts: Sequence[Part], home: Path | None = None) -> list[str]:
    """Files that would publish the path of the author's home folder.

    LayaStudio records where things were on disk — export.json's `path`,
    laya_finetune.json's `base_model_dir` — and a public registry is the wrong
    place for a machine's username. Reported, never rewritten: changing a file
    on the way out would publish something other than what is on disk.
    """
    marker = str(home or Path.home())
    if len(marker) < 4:
        return []
    hits: list[str] = []
    for part in parts:
        for path in part.found.files:
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                if path.stat().st_size > SCAN_LIMIT or marker not in path.read_text(
                    errors="ignore"
                ):
                    continue
            except OSError:
                continue
            relative = path.relative_to(part.source).as_posix()
            hits.append(f"{part.folder}/{relative}" if part.folder else relative)
    return hits
