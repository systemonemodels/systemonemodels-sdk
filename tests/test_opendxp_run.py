"""`systemone run opendxp`: find a model here or pull it, build its OpenDXP package once, answer."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from systemone import cache, cli, opendxp_run
from systemone.errors import SystemOneError

runner = CliRunner()

LAYA = [
    "rl_agent_config.json",
    "model.safetensors",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
]
LAYA_REPO = [
    *LAYA,
    *(f"multilingual/{p}" for p in LAYA),
    *(f"typed-decisions/{p}" for p in LAYA),
    "assets/logo.png",
    "README.md",
]
DECIDER_REPO = [
    "decider_config.json",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
    "decider-2b-v11-Q4_K_M.gguf",
    "decider-2b-v11-Q8_0.gguf",
]
PACKAGE = ["odxp/odxp.json", "odxp/model.onnx", "odxp/model.safetensors", "odxp/tokenizer.json"]


def plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_model_references() -> None:
    assert opendxp_run.parse_ref("Convai-Innovations/Laya") == ("convai-innovations/laya", None)
    assert opendxp_run.parse_ref("mapika/decider@v11") == ("mapika/decider", "v11")
    assert opendxp_run.parse_ref("https://systemonemodels.tech/supersonic-labs/julia-1") == (
        "supersonic-labs/julia-1",
        None,
    )
    with pytest.raises(SystemOneError, match="namespace/name"):
        opendxp_run.parse_ref("laya")


def test_checkpoints_of_a_laya_repository() -> None:
    found = opendxp_run.checkpoints(LAYA_REPO)
    assert [(c.name, c.family, c.package) for c in found] == [
        ("main", "laya", False),
        ("multilingual", "laya", False),
        ("typed-decisions", "laya", False),
    ]
    assert opendxp_run.choose(found, None).name == "main"
    assert opendxp_run.choose(found, "typed-decisions").folder == "typed-decisions"
    with pytest.raises(SystemOneError, match="Available: main, multilingual"):
        opendxp_run.choose(found, "nope")


def test_a_package_comes_before_converting() -> None:
    found = opendxp_run.checkpoints([*LAYA, *PACKAGE])
    assert opendxp_run.choose(found, None) == opendxp_run.Checkpoint(
        "odxp", "odxp", package=True, family=None
    )


def test_nothing_to_run_says_why() -> None:
    with pytest.raises(SystemOneError, match="no OpenDXP package"):
        opendxp_run.choose(opendxp_run.checkpoints(["weights.bin", "README.md"]), None)


def test_only_the_files_a_checkpoint_needs() -> None:
    found = opendxp_run.checkpoints(LAYA_REPO)
    main = opendxp_run.wanted_files(found[0], LAYA_REPO, found)
    # The root checkpoint leaves the other checkpoints, and the assets, alone.
    assert main == set(LAYA)
    typed = opendxp_run.wanted_files(found[2], LAYA_REPO, found)
    assert typed == {f"typed-decisions/{p}" for p in LAYA}


def test_decider_pulls_its_best_gguf_and_not_its_safetensors() -> None:
    found = opendxp_run.checkpoints(DECIDER_REPO)
    assert [(c.name, c.family) for c in found] == [("main", "decider")]
    wanted = opendxp_run.wanted_files(found[0], DECIDER_REPO, found)
    assert "decider-2b-v11-Q8_0.gguf" in wanted
    assert "decider-2b-v11-Q4_K_M.gguf" not in wanted
    assert "model.safetensors" not in wanted


def test_a_model_already_here_is_used_as_it_is(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("SYSTEMONE_CACHE", str(tmp_path / "cache"))
    repo = "convai-innovations/laya"
    assert opendxp_run.local_snapshot(repo, None) is None
    root = cache.snapshot_root(repo, "2026.09.25")
    for path in LAYA:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text("x")
    cache.write_ref(repo, "2026.09.25")

    assert opendxp_run.local_snapshot(repo, None) == ("2026.09.25", root)
    assert opendxp_run.local_snapshot(repo, "2026.09.25") == ("2026.09.25", root)
    # Another version is not here, whatever its name.
    assert opendxp_run.local_snapshot(repo, "2026.09") is None

    found = opendxp_run.checkpoints(opendxp_run.local_files(root))
    assert opendxp_run.has_checkpoint(root, found[0])
    (root / "model.safetensors").unlink()
    assert not opendxp_run.has_checkpoint(root, found[0])


def test_installs_only_what_is_missing(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.delenv(opendxp_run.ENV_SPEC, raising=False)
    venv = tmp_path / "venv"
    steps = opendxp_run.install_steps(venv, {"onnx", "export", "laya"}, set(), "/bin/uv")
    assert steps[0][:3] == ["/bin/uv", "pip", "install"]
    assert "--torch-backend" in steps[0] and "cpu" in steps[0]
    assert steps[0][-1] == "opendxp[export,laya,onnx]>=0.1,<0.2"

    gguf = opendxp_run.install_steps(venv, {"onnx", "gguf"}, {"base", "onnx"}, "/bin/uv")
    assert len(gguf) == 1 and "--prefer-binary" in gguf[0] and gguf[0][-1] == opendxp_run.LLAMA

    assert opendxp_run.install_steps(venv, {"onnx"}, {"base", "onnx"}, "/bin/uv") == []


def test_a_local_opendxp_checkout_for_development(monkeypatch: Any) -> None:
    monkeypatch.setenv(opendxp_run.ENV_SPEC, "/src/opendxp")
    assert opendxp_run._requirement(["onnx"]) == "/src/opendxp[onnx]"


def test_requests(tmp_path: Path) -> None:
    assert opendxp_run.load_request(None, None, None) is opendxp_run.EXAMPLE
    asked = opendxp_run.load_request(
        None, "Customer: where is my order?", '{"q": {"type": "noul", "instructions": "Late."}}'
    )
    assert asked == {
        "state": "Customer: where is my order?",
        "questions": {"q": {"type": "noul", "instructions": "Late."}},
    }
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"state": "s", "questions": {}}))
    assert opendxp_run.load_request(request, None, None) == {"state": "s", "questions": {}}


ANSWER = {
    "answers": {
        "intent": {
            "type": "choice",
            "choice": "refund",
            "probabilities": {"refund": 0.83, "other": 0.17},
        },
        "angry": {"type": "noul", "noul": 0.71},
    },
    "usage": {"input_tokens": 90, "output_tokens": 0},
}


def test_a_package_folder_runs_directly(tmp_path: Path, monkeypatch: Any) -> None:
    package = tmp_path / "my-package"
    package.mkdir()
    (package / "odxp.json").write_text(json.dumps({"profile": "encoder-markers"}))
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        opendxp_run, "ensure_runtime", lambda needs, log: seen.setdefault("needs", needs)
    )

    def fake_answer(runtime: Any, pkg: Path, request: Any, **_: Any) -> dict[str, Any]:
        seen["package"] = pkg
        return ANSWER

    monkeypatch.setattr(opendxp_run, "answer", fake_answer)
    result = runner.invoke(cli.app, ["run", "opendxp", str(package)])
    out = plain(result.output)
    assert result.exit_code == 0, out
    assert seen["package"] == package.resolve()
    assert seen["needs"] == {"onnx"}
    assert "refund" in out and "83.0%" in out and "71% true" in out


class FakeRegistry:
    def __init__(self) -> None:
        self.closed = False

    def __enter__(self) -> FakeRegistry:
        return self

    def __exit__(self, *_: Any) -> None:
        self.closed = True

    def model(self, repo: str) -> dict[str, Any]:
        return {"latest_version": "2026.09.25"}

    def versions(self, repo: str) -> list[dict[str, Any]]:
        return [{"version": "2026.09.25", "artifacts": [{"path": p} for p in LAYA_REPO]}]


def test_a_registry_model_is_pulled_converted_once_and_answered(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setenv("SYSTEMONE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(cli, "client", FakeRegistry)
    pulled: dict[str, Any] = {}

    def fake_pull(
        registry: Any, repo: str, dest: Any, version: Any, variant: Any, tick: Any, select: Any
    ) -> Any:
        pulled["files"] = select(LAYA_REPO)
        root = cache.snapshot_root(repo, version)
        for path in pulled["files"]:
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text("x")
        cache.write_ref(repo, version)
        return type("Result", (), {"root": root})()

    built: list[Path] = []

    def fake_build(runtime: Any, family: str, source: Path, out: Path, name: str, log: Any) -> Path:
        built.append(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "odxp.json").write_text(json.dumps({"profile": "encoder-markers"}))
        return out

    needs_seen: list[set[str]] = []

    def fake_runtime(needs: set[str], log: Any) -> Path:
        needs_seen.append(set(needs))
        return tmp_path / "opendxp"

    monkeypatch.setattr(cli, "pull_version", fake_pull)
    monkeypatch.setattr(opendxp_run, "build_package", fake_build)
    monkeypatch.setattr(opendxp_run, "ensure_runtime", fake_runtime)
    monkeypatch.setattr(opendxp_run, "answer", lambda *a, **k: ANSWER)

    first = runner.invoke(
        cli.app, ["run", "opendxp", "convai-innovations/laya", "--checkpoint", "typed-decisions"]
    )
    assert first.exit_code == 0, plain(first.output)
    assert pulled["files"] == {f"typed-decisions/{p}" for p in LAYA}
    assert needs_seen[-1] == {"onnx", "export", "laya"}
    assert built and built[0].name == "typed-decisions"
    # Chosen by name, so no hint about the others.
    assert "also here" not in plain(first.output)

    # Second run: the model is on this machine and its package is built.
    monkeypatch.setattr(cli, "pull_version", lambda *a, **k: pytest.fail("pulled again"))
    second = runner.invoke(
        cli.app, ["run", "opendxp", "convai-innovations/laya", "--checkpoint", "typed-decisions"]
    )
    assert second.exit_code == 0, plain(second.output)
    assert "is on this machine" in plain(second.output)
    assert needs_seen[-1] == {"onnx"}
    assert len(built) == 1
