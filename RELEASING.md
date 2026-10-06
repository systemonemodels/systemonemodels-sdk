# Releasing

Releases go to PyPI from GitHub Actions with Trusted Publishing: PyPI verifies
the workflow's identity directly, so no token is stored anywhere. Each file is
also signed with that identity, and PyPI shows the provenance on the release.

## One-time setup

1. On [pypi.org](https://pypi.org): **Your account → Publishing → Add a new
   pending publisher** (GitHub):

   | Field | Value |
   | --- | --- |
   | PyPI project name | `systemonemodels` |
   | Owner | `systemonemodels` |
   | Repository name | `systemonemodels-sdk` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   "Pending" because the project does not exist on PyPI until the first
   release creates it.

2. On GitHub: **Settings → Environments → New environment → `pypi`**. Under
   *Deployment protection rules*, tick **Required reviewers** and add yourself.
   Under *Deployment branches and tags*, allow only tags matching `v*`. Every
   release then waits for a click — worth having, because a version published
   to PyPI can never be replaced.

## Each release

```bash
# 1. bump the version in both places, and add a CHANGELOG entry
$EDITOR pyproject.toml src/systemone/__init__.py CHANGELOG.md
git commit -am "Release 0.1.1"

# 2. tag and push
git tag v0.1.1
git push origin main v0.1.1
```

The workflow checks the tag matches the package version, runs the tests on the
oldest and newest supported Python, builds, installs the wheel into a clean
environment to prove it runs, and then waits for your approval in the Actions
tab. PyPI has no review queue: a minute after approval, `pip install
systemonemodels` gets it.

The build runs in a job of its own, with nothing but uv and the build backend,
and records a digest of what it made. The tests and checks run their
third-party code in other jobs, and the publish job, the only one PyPI trusts,
uploads only files with that digest.

## Pinned tools

Nothing in the workflows moves on its own, so a hijacked release of a tool
cannot reach a published file:

- **Actions** are pinned by commit, with the version in a comment. Dependabot
  proposes new versions once a month, in one pull request.
- **uv** is `PINNED_UV` in each workflow, checked against `PINNED_UV_SHA256`:
  the sha256 of `uv-x86_64-unknown-linux-gnu.tar.gz` on that uv release.
- **The build backend** (hatchling and what it needs) is pinned with hashes in
  [.github/build-constraints.txt](.github/build-constraints.txt). To move it:

  ```bash
  echo hatchling | uv pip compile - --universal --python-version 3.10 \
    --generate-hashes -o .github/build-constraints.txt
  ```

- **twine** is pinned where it runs.

## Try a build locally

```bash
uv build -b .github/build-constraints.txt --require-hashes
uvx twine==7.0.0 check --strict dist/*
```
