#!/usr/bin/env bash
set -euo pipefail
# release.sh — bump version, commit, tag, build, and publish posetail to PyPI.
# Run inside the pixi env, e.g.:  pixi run ./release.sh 0.1.0
cd "$(dirname "$0")"

VERSION="${1:-}"
[[ -n "$VERSION" ]] || { echo "usage: $0 <version>  (e.g. $0 0.1.0)" >&2; exit 1; }
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.]+)?$ ]] \
  || { echo "error: '$VERSION' is not a valid version" >&2; exit 1; }
TAG="v$VERSION"

# preconditions
[[ -z "$(git status --porcelain)" ]] || { echo "error: working tree is dirty" >&2; exit 1; }
git rev-parse -q --verify "refs/tags/$TAG" >/dev/null \
  && { echo "error: tag $TAG already exists" >&2; exit 1; } || true

# update the single source of truth (top-level version only; ^ anchor avoids pixi/torch version = ...)
sed -i -E "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml
grep -qx "version = \"$VERSION\"" pyproject.toml \
  || { echo "error: failed to update version in pyproject.toml" >&2; exit 1; }

# commit + tag
git add pyproject.toml
git commit -m "bump to version $VERSION"
git tag "$TAG"

# build fresh artifacts + validate metadata (catches any stray direct-URL dep)
rm -rf dist/ build/ ./*.egg-info
python -m build
python -m twine check dist/*

# publish to PyPI (uses ~/.pypirc [pypi]) then push commit + tag
python -m twine upload dist/*
git push origin HEAD --follow-tags
echo "released $VERSION as $TAG"
