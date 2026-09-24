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
systemone create you/support-router
systemone push ./support-router --repo you/support-router --version 0.1.0
```

`push` uploads a directory as a new version and preserves its folder structure.
Files the registry already holds are never sent again, so republishing with one
changed file sends one file. `--variant onnx-int8` places the files under that
folder, so one repository can hold several exports of a model.

A `systemone.yaml` manifest describes what the model decides, how it was
evaluated and what it was fine-tuned from. `push` infers one from a
[Laya Studio](https://layastudio.biplovgautam.com.np/) export; pass `--manifest`
to use your own, and check it first with `systemone validate systemone.yaml`.

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

Licensed under [Apache-2.0](LICENSE).
