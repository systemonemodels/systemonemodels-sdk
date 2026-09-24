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

The workflow checks the tag matches the package version, runs the tests on
every supported Python, builds, installs the wheel into a clean environment to
prove it runs, and then waits for your approval in the Actions tab. PyPI has no
review queue: a minute after approval, `pip install systemonemodels` gets it.

## Try a build locally

```bash
uv build
uvx twine check --strict dist/*
```
