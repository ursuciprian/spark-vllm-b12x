#!/usr/bin/env bash
# ~/GEN-AI/build/publish.sh
#
# Tag and push our locally-built spark-vllm-b12x image to a registry the user
# controls. WRITTEN, NOT RUN -- pushing requires the user's own registry
# credentials; this script never runs docker push on its own. Nothing in this
# repo/session has push access, and it should stay that way.
#
# Usage:
#   1. Log in first (the user runs this themselves, with their own token):
#        echo "$GHCR_TOKEN" | docker login ghcr.io -u <github-username> --password-stdin
#   2. Then:
#        OWNER=<github-username-or-org> ./publish.sh spark-vllm-b12x:local-20260918-a8333658
set -euo pipefail

LOCAL_TAG="${1:?usage: OWNER=<owner> ./publish.sh <local-image-tag>}"
OWNER="${OWNER:?set OWNER=<your-github-username-or-org>}"
REMOTE_TAG="ghcr.io/${OWNER}/spark-vllm-b12x:${LOCAL_TAG#*:}"

echo "== sanity: image exists locally =="
docker image inspect "$LOCAL_TAG" >/dev/null

echo "== sanity: logged in to ghcr.io =="
# `docker login` writes a credential entry; this just checks one exists, it
# does not validate the token still works (that only happens on push).
if ! docker info 2>/dev/null | grep -q "ghcr.io" && \
   ! grep -q "ghcr.io" "${DOCKER_CONFIG:-$HOME/.docker}/config.json" 2>/dev/null; then
    echo "Not logged in to ghcr.io. Run this first (your own token, never share it):" >&2
    echo "  echo \"\$GHCR_TOKEN\" | docker login ghcr.io -u <github-username> --password-stdin" >&2
    exit 1
fi

echo "== tag =="
docker tag "$LOCAL_TAG" "$REMOTE_TAG"
echo "Tagged $LOCAL_TAG -> $REMOTE_TAG"

echo "== push (this is the only network-mutating step) =="
docker push "$REMOTE_TAG"

echo "== done =="
echo "Pushed $REMOTE_TAG"
echo "Point a recipe's 'container:' field at it once you have confirmed the push:"
echo "  container: $REMOTE_TAG"
