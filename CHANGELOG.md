# Changelog

## 0.2.1

- `push` checks that you are signed in, and that the registry answers, before
  it scans the folder and asks which models to publish — not after.
- A registry that cannot be reached now points at
  `systemone login --endpoint URL`.
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
