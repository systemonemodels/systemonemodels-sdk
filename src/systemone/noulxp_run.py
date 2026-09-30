"""`systemone run noulxp`: answer with a System One model on this machine, through NoulXP.

NoulXP, the open exchange protocol for decision models (github.com/systemonemodels/noulxp),
packages a System One model so that one open runtime answers for it, with no
code written for that model. This command:

1. looks for the model on this machine first: a version already pulled into the
   cache is used as it is;
2. otherwise pulls it from System One Models, only the checkpoint it needs;
3. uses the version's NoulXP package (a noulxp.json at its root or in a folder),
   or, for a model that has none yet but whose family NoulXP converts (Laya,
   Julia 1, Decider), builds one here from the model's own files, once;
4. answers through the `noulxp` command, in a Python environment of its own, so
   the CLI stays light and the runtime's libraries never meet yours.

Layout, under the platform's data directory:

    <data>/systemone/noulxp/venv                                   the runtime
    <data>/systemone/noulxp/packages/<ns>--<name>/<version>/<checkpoint>   built here
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from systemone import cache, studio
from systemone.errors import SystemOneError

MANIFEST = "noulxp.json"
# The manifest's name in packages made before the rename (OpenDXP, to 0.3.1).
MANIFESTS = (MANIFEST, "odxp.json")
MAIN = "main"
# A local checkout or wheel of noulxp to install instead of the published one.
ENV_SPEC = "SYSTEMONE_NOULXP_SPEC"
# Its name before the rename, still read.
LEGACY_ENV_SPEC = "SYSTEMONE_OPENDXP_SPEC"
REQUIREMENT = ">=0.4,<0.5"
LLAMA_WHEELS = "https://abetlen.github.io/llama-cpp-python/whl/{}"
LLAMA = "llama-cpp-python==0.3.35"
# Files that make a checkpoint of a family NoulXP can convert, per family.
FAMILIES = {
    "laya": ("rl_agent_config.json", "model.safetensors"),
    "julia": ("julia_config.json", "model.safetensors"),
    "decider": ("decider_config.json", "tokenizer.json"),
}
# What to ask when nothing else is given: a support message and three kinds of question.
EXAMPLE: dict[str, Any] = {
    "state": "Customer: I was charged twice for my order last week and I still "
    "haven't got a refund. This is the third time I'm writing.",
    "questions": {
        "intent": {
            "type": "choice",
            "instructions": "What does the customer want?",
            "criteria": {
                "refund": None,
                "cancel order": None,
                "track delivery": None,
                "other": None,
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["low", "medium", "high"],
        },
        "angry": {"type": "noul", "instructions": "The customer is angry."},
    },
}

Log = Callable[[str], None]


@dataclass(frozen=True)
class Checkpoint:
    """One runnable model in a version: a NoulXP package, or files NoulXP converts."""

    name: str  # "main" for the version's root, else its folder
    folder: str  # "" for the root
    package: bool  # a noulxp.json is there
    family: str | None  # a family NoulXP converts, when there is no package


# Beyond FAMILIES' marker files, what each family's converter reads.
FAMILY_FILES = {
    "laya": (
        "rl_agent_config.json",
        "model.safetensors",
        "encoder/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    ),
    "julia": (
        "julia_config.json",
        "config.json",
        "inference-policy.json",
        "model.safetensors",
        "encoder/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    ),
    "decider": ("decider_config.json", "config.json", "tokenizer.json", "tokenizer_config.json"),
}
GGUF_PREFERENCE = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M")


def wanted_files(checkpoint: Checkpoint, paths: Iterable[str], found: list[Checkpoint]) -> set[str]:
    """The files of a version this checkpoint needs, and only those.

    A package needs its folder. A root checkpoint needs the root, less the other
    checkpoints' folders. A checkpoint that is converted here needs its family's
    files: for Decider, its best GGUF build and not its multi-gigabyte safetensors.
    """
    prefix = f"{checkpoint.folder}/" if checkpoint.folder else ""
    others = [c.folder + "/" for c in found if c.folder and c.folder != checkpoint.folder]
    mine = {p for p in paths if p.startswith(prefix) and not any(p.startswith(o) for o in others)}
    if checkpoint.package:
        return mine
    named = {prefix + f for f in FAMILY_FILES[checkpoint.family or ""]}
    chosen = mine & named
    if checkpoint.family == "decider":
        ggufs = sorted(p for p in mine if p.endswith(".gguf") and "/" not in p[len(prefix) :])
        ranked = [g for q in GGUF_PREFERENCE for g in ggufs if q in g] or ggufs
        chosen |= set(ranked[:1])
    return chosen


def parse_ref(ref: str) -> tuple[str, str | None]:
    """`namespace/name[@version]`, or a registry URL, as (repo, version)."""
    text = ref.strip()
    for prefix in ("https://", "http://"):
        if text.startswith(prefix):
            text = text[len(prefix) :].split("/", 1)[-1]
    repo, _, version = text.partition("@")
    parts = [p for p in repo.strip("/").split("/") if p]
    if len(parts) != 2:
        raise SystemOneError(f"expected namespace/name, got {ref!r}")
    return f"{parts[0].lower()}/{parts[1].lower()}", version or None


def checkpoints(paths: Iterable[str]) -> list[Checkpoint]:
    """The runnable checkpoints in a version, from its file paths.

    A NoulXP package is a folder (or the root) with a noulxp.json. Without one,
    a folder (or the root) holding a family's files is a checkpoint NoulXP can
    build a package from.
    """
    files = set(paths)
    folders = sorted({""} | {p.split("/", 1)[0] for p in files if "/" in p})
    found = []
    for folder in folders:
        prefix = f"{folder}/" if folder else ""
        name = folder or MAIN
        if any(prefix + m in files for m in MANIFESTS):
            found.append(Checkpoint(name, folder, package=True, family=None))
            continue
        for family, needed in FAMILIES.items():
            if all(prefix + n in files for n in needed):
                if family == "decider" and not any(
                    p.startswith(prefix) and p.endswith(".gguf") and "/" not in p[len(prefix) :]
                    for p in files
                ):
                    continue
                found.append(Checkpoint(name, folder, package=False, family=family))
                break
    return found


def choose(found: list[Checkpoint], wanted: str | None) -> Checkpoint:
    if not found:
        raise SystemOneError(
            "this version has no NoulXP package, and none of its files is a model NoulXP "
            "can convert (Laya, Julia 1, Decider). Its maker can add one with `noulxp export`."
        )
    if wanted:
        for checkpoint in found:
            if checkpoint.name == wanted:
                return checkpoint
        names = ", ".join(c.name for c in found)
        raise SystemOneError(f"no checkpoint {wanted!r} here. Available: {names}")
    packaged = [c for c in found if c.package]
    pool = packaged or found
    return next((c for c in pool if c.name == MAIN), pool[0])


def local_snapshot(repo: str, version: str | None) -> tuple[str, Path] | None:
    """A version already pulled into the cache: the latest one, or the one asked for."""
    latest = cache.read_ref(repo)
    if latest is None or (version is not None and version != latest):
        return None
    root = cache.snapshot_root(repo, latest)
    return (latest, root) if root.is_dir() else None


def local_files(root: Path) -> list[str]:
    return sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()
    )


def has_checkpoint(root: Path, checkpoint: Checkpoint) -> bool:
    """Whether a snapshot holds everything this checkpoint needs, not just its name."""
    prefix = f"{checkpoint.folder}/" if checkpoint.folder else ""
    if checkpoint.package:
        manifest = manifest_in(root / prefix)
        if manifest is None:
            return False
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return all((root / prefix / path).is_file() for path in _manifest_paths(data))
    needed = FAMILIES[checkpoint.family or ""]
    if not all((root / prefix / n).is_file() for n in needed):
        return False
    if checkpoint.family == "decider":
        return any((root / prefix).glob("*.gguf")) if prefix else any(root.glob("*.gguf"))
    return True


def _manifest_paths(data: dict[str, Any]) -> list[str]:
    paths = []
    for key in ("weights", "tokenizer", "template", "prompt", "calibration", "conformance"):
        entry = data.get(key)
        if isinstance(entry, dict) and entry.get("path"):
            paths.append(str(entry["path"]))
            paths += [str(d["path"]) for d in entry.get("data", []) if d.get("path")]
    return paths


# --- the runtime's environment -------------------------------------------------------


def root_dir() -> Path:
    return studio.data_dir() / "noulxp"


def env_dir() -> Path:
    return root_dir() / "venv"


def package_dir(repo: str, version: str, checkpoint: str) -> Path:
    namespace, _, name = repo.partition("/")
    return root_dir() / "packages" / f"{namespace}--{name}" / version / checkpoint


def _requirement(extras: Iterable[str]) -> str:
    joined = ",".join(sorted(extras))
    source = os.environ.get(ENV_SPEC) or os.environ.get(LEGACY_ENV_SPEC)
    if source:
        return f"{source}[{joined}]" if joined else source
    return f"noulxp[{joined}]{REQUIREMENT}" if joined else f"noulxp{REQUIREMENT}"


def _apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def install_steps(venv: Path, needs: set[str], have: set[str], uv: str | None) -> list[list[str]]:
    """The commands that give the environment what it lacks, and nothing it has."""
    python = str(studio._venv_bin(venv, "python"))
    steps: list[list[str]] = []
    python_extras = {"onnx", "export", "laya"} & needs
    if python_extras - have or "base" not in have:
        requirement = _requirement(python_extras)
        if uv:
            command = [uv, "pip", "install", "--python", python]
            if "export" in python_extras:
                # Converting runs on the CPU: the small PyTorch build is enough.
                command += ["--torch-backend", "cpu"]
            steps.append([*command, requirement])
        else:
            command = [python, "-m", "pip", "install", "--quiet"]
            if "export" in python_extras and platform.system() != "Darwin":
                command += ["--extra-index-url", "https://download.pytorch.org/whl/cpu"]
            steps.append([*command, requirement])
    if "gguf" in needs and "gguf" not in have:
        # llama.cpp from a prebuilt wheel: PyPI ships only its source.
        channel = "metal" if _apple_silicon() else "cpu"
        steps.append(
            [
                python,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--prefer-binary",
                "--extra-index-url",
                LLAMA_WHEELS.format(channel),
                LLAMA,
            ]
        )
    return steps


def ensure_runtime(needs: set[str], *, log: Log = print) -> Path:
    """The `noulxp` command in its own environment, with the extras asked for."""
    venv = env_dir()
    stamp = venv / ".systemone-noulxp.json"
    have: set[str] = set()
    if stamp.is_file():
        try:
            recorded = json.loads(stamp.read_text())
            if recorded.get("source") == _requirement([]):
                have = set(recorded.get("extras", []))
        except (OSError, json.JSONDecodeError):
            have = set()
    uv = shutil.which("uv")
    if not studio._venv_bin(venv, "python").exists():
        log("Creating the NoulXP runtime's Python environment")
        venv.parent.mkdir(parents=True, exist_ok=True)
        if uv:
            made = studio._run([uv, "venv", "--seed", "--python", "3.12", str(venv)])
        else:
            base = studio._python_for_venv()
            if base is None:
                raise SystemOneError(
                    "NoulXP needs Python 3.11 or newer. Install uv "
                    "(https://docs.astral.sh/uv/), which fetches one for you, or Python 3.11+."
                )
            made = studio._run([base, "-m", "venv", str(venv)])
        if made.returncode != 0:
            raise SystemOneError(f"could not create an environment: {made.stderr.strip()}")
        have = set()
    steps = install_steps(venv, needs, have, uv)
    if steps:
        heavy = " (the first conversion downloads PyTorch)" if "export" in needs - have else ""
        log(f"Installing the NoulXP runtime{heavy}")
    for step in steps:
        done = subprocess.run(step, check=False)  # noqa: S603 - installers, no shell
        if done.returncode != 0:
            raise SystemOneError("installing the NoulXP runtime failed; see the output above")
    stamp.write_text(
        json.dumps({"source": _requirement([]), "extras": sorted(have | needs | {"base"})})
    )
    return studio._venv_bin(venv, "noulxp")


def needs_for(checkpoint: Checkpoint, profile: str | None) -> set[str]:
    """The extras a checkpoint needs: its runtime, and the converters when it has no package."""
    needs = {"onnx"}
    if checkpoint.family == "decider" or profile == "causal-letters":
        needs.add("gguf")
    if not checkpoint.package:
        needs.add("export")
        if checkpoint.family == "laya":
            needs.add("laya")
    return needs


def manifest_in(package: Path) -> Path | None:
    """A package's manifest: noulxp.json, or odxp.json from before the rename."""
    return next((package / m for m in MANIFESTS if (package / m).is_file()), None)


def profile_of(package: Path) -> str | None:
    manifest = manifest_in(package)
    if manifest is None:
        return None
    try:
        return str(json.loads(manifest.read_text(encoding="utf-8"))["profile"])
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def build_package(
    noulxp: Path, family: str, source: Path, out: Path, name: str, *, log: Log = print
) -> Path:
    """A NoulXP package from a checkpoint's own files, built once and kept."""
    if manifest_in(out) is not None:
        return out
    log(f"Building a NoulXP package from the {family} checkpoint (once)")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.with_name(out.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    done = subprocess.run(  # noqa: S603 - the environment's own noulxp, no shell
        [str(noulxp), "export", family, str(source), str(staging), "--name", name],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        shutil.rmtree(staging, ignore_errors=True)
        detail = (done.stderr or done.stdout).strip().splitlines()[-3:]
        raise SystemOneError("building the NoulXP package failed: " + " ".join(detail))
    shutil.rmtree(out, ignore_errors=True)
    staging.rename(out)
    return out


def answer(
    noulxp: Path, package: Path, request: dict[str, Any], *, device: str, threads: int | None
) -> dict[str, Any]:
    command = [str(noulxp), "run", str(package), "--request", "-", "--device", device]
    if threads:
        command += ["--threads", str(threads)]
    done = subprocess.run(  # noqa: S603 - the environment's own noulxp, no shell
        command, input=json.dumps(request), capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        detail = (done.stderr or done.stdout).strip().splitlines()[-3:]
        raise SystemOneError("the NoulXP runtime failed: " + " ".join(detail))
    try:
        result: dict[str, Any] = json.loads(done.stdout)
    except json.JSONDecodeError as exc:
        raise SystemOneError(f"the NoulXP runtime gave no JSON: {done.stdout[-200:]}") from exc
    return result


def load_request(request: Path | None, state: str | None, questions: str | None) -> dict[str, Any]:
    """The request to answer: a file, a state with questions, or the built-in example."""
    if request is not None:
        text = request.read_text(encoding="utf-8")
        loaded: dict[str, Any] = json.loads(text)
        return loaded
    if questions is not None:
        source = Path(questions)
        parsed = json.loads(source.read_text(encoding="utf-8") if source.is_file() else questions)
        return {"state": state or "", "questions": parsed}
    if state is not None:
        return {"state": state, "questions": EXAMPLE["questions"]}
    return EXAMPLE
