# Changelog

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
