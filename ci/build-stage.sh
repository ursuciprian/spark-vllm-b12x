#!/usr/bin/env bash
# ci/build-stage.sh <target> [extra docker buildx build args...]
#
# Shared "clone eugr/spark-vllm-docker at the pinned ref, apply the submodule
# fix, build one Dockerfile target" step. Used by build-image-hosted.yml's
# three jobs (flashinfer-export, vllm-export, runner) so each fresh runner
# job builds against the exact same pinned/patched Dockerfile.
#
# NOT wired into build.sh: build.sh's dgx-01 flow does CONFIRM_BUILD gating,
# wheel-cache reuse across local runs, and ssh distribution that don't apply
# to a one-shot hosted-runner job, and self-hosted already gets build reuse
# for free from its persistent local BuildKit cache. Duplicating the ~6-line
# clone+patch here is smaller than bending build.sh's tested dgx-01 path to
# fit both callers.
#
# Env overrides (all optional):
#   SPARK_VLLM_DOCKER_REPO, SPARK_VLLM_DOCKER_REF -- eugr's Dockerfile repo/pin
#   WORKDIR                                       -- scratch checkout dir
set -euo pipefail

TARGET="${1:?usage: ci/build-stage.sh <target> [docker buildx build args...]}"
shift

SPARK_VLLM_DOCKER_REPO="${SPARK_VLLM_DOCKER_REPO:-https://github.com/eugr/spark-vllm-docker.git}"
SPARK_VLLM_DOCKER_REF="${SPARK_VLLM_DOCKER_REF:-798528a2}"
WORKDIR="${WORKDIR:-$(mktemp -d)/spark-vllm-docker}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== clone eugr/spark-vllm-docker @ $SPARK_VLLM_DOCKER_REF into $WORKDIR =="
rm -rf "$WORKDIR"
git clone --quiet "$SPARK_VLLM_DOCKER_REPO" "$WORKDIR"
git -C "$WORKDIR" checkout --quiet "$SPARK_VLLM_DOCKER_REF"

echo "== apply submodule/clone fixes (idempotent) =="
python3 "$REPO_ROOT/apply_submodule_fix.py" "$WORKDIR/Dockerfile"

if [ "$TARGET" = "runner" ]; then
    echo "== generate build-metadata.yaml (runner stage's last COPY expects it) =="
    BASE_IMAGE="$(grep -m1 '^FROM .* AS runner' "$WORKDIR/Dockerfile" | awk '{print $2}')"
    cat > "$WORKDIR/build-metadata.yaml" <<EOF
build_date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
build_script_commit: ci/build-stage.sh
gpu_arch: 12.1a
base_image: ${BASE_IMAGE:-unknown}
EOF
fi

echo "== docker buildx build --target $TARGET =="
( cd "$WORKDIR" && docker buildx build --target "$TARGET" -f Dockerfile "$@" . )
