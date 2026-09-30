# Changelog

## 0.5.0

- `systemone run noulxp`: OpenDXP, the standard it runs models through, is now
  NoulXP (another project already used the name). It installs `noulxp` 0.4
  into its own environment, in place of `opendxp` 0.1; the format and the
  answers are the same, and packages made before the rename (an `odxp.json`)
  still run. `systemone run opendxp` still works, with a note, and
  `SYSTEMONE_OPENDXP_SPEC` is still read (now `SYSTEMONE_NOULXP_SPEC`).

## 0.4.0

- Call the models System One Models serves, from your own code:
  `Client(api_key).decide(model, state, questions)` asks one through the
  inference API and returns its answers, one per question, with the usage and
  the time it took. The key is the one given, else `SYSTEMONE_API_KEY`, else
  your `systemone login`; it is used only for calls to models, never for
  registry calls, so setting it cannot change what `push` does. Create keys at
  systemonemodels.tech/settings/api.
- `client.served_models()` lists the models you can call, and `client.usage()`
  shows your plan, what is left of it and your calls by day, model and key.
- `RateLimited` (an `ApiError`) is raised on HTTP 429, over a per-minute limit
  or out of decisions, with `retry_after` in seconds.
- `systemone decide MODEL --state ... --questions ...` asks from the terminal,
  with `--request FILE` and `--json` like `run opendxp`; with no request it asks
  a built-in example.
- `Client(config)` still works as before; the first argument may now be an API
  key instead.

## 0.3.0

- `systemone run opendxp MODEL` answers with a model on this machine through
  [OpenDXP](https://github.com/systemonemodels/opendxp), the open standard for
  System One models. It uses the model if it was pulled before, and otherwise
  pulls only the checkpoint it needs. A model without an OpenDXP package, from
  a family OpenDXP converts (Laya, Julia 1, Decider), gets one built once. The
  runtime lives in an environment of its own. With no request it answers a
  built-in example; `--state` and `--questions`, or `--request`, ask your own,
  and a folder with an `odxp.json` runs directly.
- Laya Studio is now System One Studio. `systemone run studio` runs it as
  before; its workspace stays in `~/.layastudio/workspace`.

## 0.2.4

- `systemone run studio` fetches Laya Studio, keeps it up to date, sets up its
  Python environment (uv when installed, a virtual environment otherwise) and
  starts it in the browser, offering to sign in first so the studio can
  publish. The source is unpacked with every path checked; datasets and runs
  stay in `~/.layastudio/workspace`. Before starting it prints what the
  machine can train on.
- `run studio` works on Windows and Linux too. It installs the PyTorch build the
  machine needs, chosen from its own detection rather than uv's guess: CUDA 13
  for Turing and newer NVIDIA cards on driver 580+, CUDA 12.6 for older cards or
  drivers, ROCm 7.2 for AMD on Linux, AMD's ROCm 10 wheels for RX 7000/9000 on
  Windows, XPU for Intel Arc, and the CPU otherwise. Only Intel Macs are left
  out, since neither PyTorch nor MLX builds for them any more. `--source DIR`
  runs a Laya Studio checkout instead of the managed copy.
- `systemone system` shows the OS, CPU, memory, every GPU with its memory and
  driver stack (CUDA, ROCm, DirectML, Intel XPU, Apple Metal) and free disk;
  `--json` for scripts. Standard library only, so it runs before any ML stack
  is installed.
- The studio and its tools run from the studio's own folder, so a
  `.python-version` in the folder you happen to be in cannot send a pyenv shim
  looking for a Python that is not installed.

## 0.2.3

- `push` recognises Hugging Face checkpoint layouts: sharded `*.safetensors`,
  `pytorch_model.bin`, `.pt`/`.pth`/`.ckpt` and `.tflite` files all mark a
  folder as a model, so a repository downloaded with `huggingface-cli` can be
  published as it is.
- A yes/no question is published as the `noul` capability, the field's own
  word for it, rather than `classify`.

## 0.2.2

- Uploads send the headers the registry's ticket says its signature covers
  (the size, and the file's SHA-256), so the store itself refuses bytes that
  differ from what was declared. Needed to publish to registries from
  2026-09-25 on; older clients get "Storage refused the upload (HTTP 403)".

## 0.2.1

- `push` checks that you are signed in, and that the registry answers, before
  it scans the folder and asks which models to publish — not after.
- A registry that cannot be reached now points at
  `systemone login --endpoint URL`.
- `logout` revokes the stored token on the registry instead of leaving it
  valid, then forgets the login as before, registry address included. A token
  given in `SYSTEMONE_TOKEN` is never touched.
- A base model that came from Hugging Face — named by a Hugging Face model
  card, or by Laya Studio's `hub:` references — is published with
  `base_model_source: huggingface`, so the model page links to it until the
  registry holds it too. Generated model cards link to it as well.

## 0.2.0

- `systemone push` finds the models in a folder and its subfolders — trained
  checkpoints and their ONNX and Core ML exports, and any folder with
  `model.onnx`, `model.safetensors`, a `.gguf` or an `.mlpackage` — and asks
  which to publish. `--all` publishes every one. The folder argument and
  `--repo` are now optional: the repository defaults to your namespace and the
  model's name, `--namespace` picks an organization.
- Variants of one model are published as one version, a folder each.
- A model's `README.md` becomes its model card; Hugging Face front matter sets
  the licence, tags and base model instead of being shown. Models without a
  card get one generated from their evaluation.
- Laya Studio runs publish their held-out evaluation — accuracy, calibration
  error, median and p95 latency — from `eval.json`. Yes/no questions are
  published as the `classify` capability.
- The version defaults to the next minor after the latest published one.
- `push` shows its plan and asks first, `--yes` skips the question, and it warns
  when a file would publish your home folder's path.
- Uploads stream from disk instead of loading each file into memory, with
  byte-level progress.
- Sizes are shown in decimal units (1 MB = 1,000,000 bytes), as Finder and the
  progress bar count them.

## 0.1.0

First release.

- `systemone login` signs in through the browser with a one-time code, and
  works over SSH. `--token` and `SYSTEMONE_TOKEN` cover CI.
- `search`, `show`, `pull` and `push`, with files cached by content: a second
  pull downloads nothing, and variants that share files store them once.
- `create` and `validate` for repositories and `systemone.yaml` manifests,
  including manifests inferred from Laya Studio exports.
- `snapshot_download()` and `Client` for use from Python.
- Readable one-line errors when the registry cannot be reached.
