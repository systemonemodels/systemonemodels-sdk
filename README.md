# systemonemodels

[![PyPI](https://img.shields.io/pypi/v/systemonemodels.svg)](https://pypi.org/project/systemonemodels/)
[![Python](https://img.shields.io/pypi/pyversions/systemonemodels.svg)](https://pypi.org/project/systemonemodels/)
[![CI](https://github.com/systemonemodels/systemonemodels-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/systemonemodels/systemonemodels-sdk/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

The command line and Python client for [System One](https://systemonemodels.tech),
the registry for **decision models** — models that return a typed decision with
a calibrated confidence instead of generated text: choose, score, rank,
classify, route.

```bash
pip install systemonemodels
```

This installs the `systemone` command and the `systemone` Python package.
Python 3.10 or newer.

## Sign in

```bash
systemone login
```

The CLI shows a one-time code and opens the registry in your browser, where you
approve this machine. Over SSH, open the printed address on any device instead.
For CI, skip the browser:

```bash
export SYSTEMONE_TOKEN=s1_pat_...   # create one at systemonemodels.tech/settings/tokens
systemone whoami
```

## Find and pull models

```bash
systemone search routing
systemone search --capability route --architecture laya --sort downloads
systemone show biplov/snake-balanced-multilingual
systemone pull biplov/snake-balanced-multilingual --variant onnx-int8
```

Files are cached by content: a second pull downloads nothing, and variants that
share a tokenizer store it once. `--dest ./snake` links the files into a folder
of your choosing; `systemone cache info` and `systemone cache clear` manage the
cache.

## Publish

```bash
cd ~/my-workspace
systemone push
```

Run `push` in a folder and it finds every model underneath, however deep —
trained checkpoints, ONNX and Core ML exports, GGUF files, any folder holding
`model.onnx`, `model.safetensors`, a `.gguf` or an `.mlpackage` — lists them,
and asks which to publish (`1,3-5` or `all`). Datasets, virtual environments and
caches are never searched.

```
 #  MODEL                        VARIANT        SIZE  FOLDER
 1  banking77-balanced           mlx        646.8 MB  runs/banking77-balanced-0922-234657/model
 2  snake-balanced-multilingual  mlx        646.8 MB  runs/snake-balanced-multilingual-0923-005225/model
 3  snake-balanced-multilingual  onnx-int8  342.7 MB  exports/snake-balanced-multilingual-0923-005225-onnx-int8
```

Variants of one model are published together, as one version of one
repository with a folder each, so `systemone pull you/snake-balanced-multilingual
--variant onnx-int8` fetches only that folder. A model on its own keeps its
files at the top of the repository, the way it sits on disk.

To publish one folder under a name of your choosing:

```bash
systemone push ./runs/snake/model --repo you/laya-snake
```

What gets published, without retyping any of it:

- **The model card.** The folder's `README.md`, with its Hugging Face front
  matter read for the licence, tags and base model rather than shown. A model
  without one gets a card written from its evaluation, questions and variants.
- **The manifest.** `systemone.yaml` — what the model decides, its runtime, and
  its evaluation: accuracy, calibration error and latency from a
  [System One Studio](https://layastudio.biplovgautam.com.np/) run's `eval.json`, or
  an export's own measurements. A base model that came from Hugging Face is
  recorded as such, so the model page links to it. Pass `--manifest` to use your
  own, and check it first with `systemone validate systemone.yaml`.
- **The version.** The next minor version after the latest (`0.1.0`, `0.2.0`,
  …), or `--version`.

`push` shows the plan before sending anything and, in a terminal, asks first —
unless you named the repository with `--repo` or passed `--yes`. It warns when a
file would publish the path of your home folder. `--dry-run` prints the manifests,
cards and files and stops. `--namespace` publishes under an organization,
`--license` overrides the card's licence, and `--all --yes` publishes
everything found without a question, for scripts and CI.

Files the registry already holds are never sent again, so republishing with one
changed file sends one file, and large files stream from disk with a progress
bar that moves by the byte.

## What this machine can train on

```bash
systemone system          # OS, CPU, memory, GPUs and their driver stack, free disk
systemone system --json   # the same, for scripts
```

## Run a model on this machine

```bash
systemone run noulxp convai-innovations/laya --checkpoint typed-decisions
systemone run noulxp convai-innovations/laya --state "Customer: where is my parcel?" \
    --questions '{"intent": {"type": "choice", "instructions": "What does the customer want?",
                              "criteria": ["refund", "track delivery", "cancel order"]}}'
systemone run noulxp ./my-package --request request.json --json
```

Runs the model through [NoulXP](https://github.com/systemonemodels/noulxp), the
open standard that lets one runtime answer for any System One model. A model you
pulled before is used as it is; otherwise only the checkpoint you asked for is
downloaded. A model with no NoulXP package yet, from a family NoulXP converts
(Laya, Julia 1, Decider), gets one built once on this machine. The runtime gets a
Python environment of its own (the first conversion downloads PyTorch).
`--device` picks a backend (`cpu`, `cuda`, `coreml`, ...); the default uses CUDA
when it can and the CPU otherwise. NoulXP was called OpenDXP until 0.3.1, and
`systemone run opendxp` still works.

## Fine-tune locally with System One Studio

System One Studio was called Laya Studio until September 2026.

```bash
systemone run studio
```

Fetches [System One Studio](https://github.com/biplovgautam/LayaStudio) (formerly
Laya Studio) the first time,
updates it on later runs, gives it its own Python environment (through
[uv](https://docs.astral.sh/uv/) when it is installed, otherwise a virtual
environment on Python 3.11+) and opens it in your browser. It offers to sign you
in first, so the studio's Publish button can push a finished run to the registry.
Your datasets and runs stay in `~/.layastudio/workspace` (the folder keeps its old name); the code lives in the
platform's data folder and can be deleted at any time.

`--port`, `--no-browser`, `--no-update` and `--workspace` do what they say, and
anything after `--` goes to the studio: `systemone run studio -- --model
aac6fef/laya-mlx`. It runs on Apple silicon (MLX), and on Windows and Linux with
the PyTorch build for the machine's GPU — NVIDIA (CUDA 13 or 12.6), AMD (ROCm on
Linux; RX 7000 and 9000 on Windows), Intel Arc (XPU) — or the CPU. `systemone
system` shows what it found. `--source DIR` runs a System One Studio checkout instead of
the managed copy.

## Call a model through the API

Models with a live playground on System One Models can be called from your own
code. Create an API key at
[Settings → API](https://systemonemodels.tech/settings/api); each question a
model answers is one decision of your plan (the free plan has 500 a day).

```python
from systemone import Client

client = Client("s1_pat_...")  # or set SYSTEMONE_API_KEY
result = client.decide(
    "nokia/anyjev",
    "Customer: I was charged twice and want my money back.",
    {
        "intent": {
            "type": "choice",
            "instructions": "What does the customer want?",
            "criteria": ["refund", "track delivery", "cancel order"],
        }
    },
)
print(result["answers"]["intent"]["choice"])
```

From the terminal:

```bash
export SYSTEMONE_API_KEY=s1_pat_...
systemone decide nokia/anyjev --state "Customer: where is my parcel?" \
    --questions '{"intent": {"type": "choice", "instructions": "What does the customer want?", "criteria": ["refund", "track delivery"]}}'
```

`client.served_models()` lists what can be called and `client.usage()` what is
left of your plan. Over a limit, `decide` raises `RateLimited` with
`retry_after` in seconds. The request and answer format is in the
[Inference API docs](https://systemonemodels.tech/docs/inference).

## In Python

```python
from systemone import snapshot_download

path = snapshot_download("biplov/snake-balanced-multilingual", variant="onnx-int8")
```

`path` is a local directory with the version's files, served from the cache on
every call after the first.

```python
from systemone import Client

with Client() as registry:
    results = registry.search("routing", capability="route")
    model = registry.model("biplov/snake-balanced-multilingual")
```

## Configuration

| Variable | |
| --- | --- |
| `SYSTEMONE_TOKEN` | Access token. Takes precedence over a stored login, for CI. |
| `SYSTEMONE_API_KEY` | API key for calling models (`decide`); never used for registry calls. |
| `SYSTEMONE_ENDPOINT` | API address, for a self-hosted registry. |
| `SYSTEMONE_HOME` | Where the login is stored (`0600`). Defaults to the platform config directory. |
| `SYSTEMONE_CACHE` | Where downloaded files are cached. |

## Development

```bash
git clone https://github.com/systemonemodels/systemonemodels-sdk
cd systemonemodels-sdk
uv sync
uv run pytest
uv run ruff check . && uv run mypy
```

Issues and pull requests are welcome. For anything security-related, see
[SECURITY.md](SECURITY.md) instead of opening a public issue.

## Links

- Registry: <https://systemonemodels.tech/models>
- CLI guide: <https://systemonemodels.tech/docs/sdk>
- Manifest specification: <https://systemonemodels.tech/docs/manifest>
- Changelog: [CHANGELOG.md](CHANGELOG.md)
- Every System One model, compared: <https://systemonemodels.tech/system-one-models>
- System One Studio (formerly Laya Studio): <https://github.com/biplovgautam/LayaStudio>
- System One Models on [LinkedIn](https://www.linkedin.com/company/system-one-models/), [X](https://x.com/SystemoneModels), [Hugging Face](https://huggingface.co/systemonemodels) and [Instagram](https://www.instagram.com/systemonemodels.tech/)
- Contact: ceo@systemonemodels.tech

Licensed under [Apache-2.0](LICENSE).
