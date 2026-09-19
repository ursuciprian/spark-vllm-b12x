#!/usr/bin/env bash
# build.sh
#
# Build our own spark-vllm-b12x image from patchable clones of our vLLM and
# b12x forks, then load it onto both cluster nodes (this head +
# $DISTRIBUTE_TO, if set).
#
# WRITTEN, NOT RUN unattended. Do not execute while a model is serving or
# another chain is booting on the build host: this is a >1h, high-RAM/CPU/disk
# compile. Check `docker ps` / `free -h` / the pair watchdog first (the CI
# workflow's preflight step does this automatically).
#
# Env overrides (all optional):
#   VLLM_REPO, VLLM_REF       -- our vLLM fork/branch (default: ursuciprian/vllm@dgx-spark)
#   B12X_REPO, B12X_REF       -- our b12x fork/branch (default: ursuciprian/b12x@dgx-spark)
#   SPARK_VLLM_DOCKER_REPO    -- eugr's Dockerfile repo (default: eugr/spark-vllm-docker)
#   SPARK_VLLM_DOCKER_REF     -- pinned commit (default: 798528a2, 2026-09-16)
#   DISTRIBUTE_TO             -- if set, ssh host to docker save|ssh|load the image onto
#   OWNER                     -- ghcr.io namespace for --push (default: ursuciprian)
set -euo pipefail

BUILD_ROOT="${BUILD_ROOT:-$HOME/GEN-AI/build}"
VLLM_REPO="${VLLM_REPO:-https://github.com/ursuciprian/vllm.git}"
VLLM_REF="${VLLM_REF:-dgx-spark}"
B12X_REPO="${B12X_REPO:-https://github.com/ursuciprian/b12x.git}"
B12X_REF="${B12X_REF:-dgx-spark}"
SPARK_VLLM_DOCKER_REPO="${SPARK_VLLM_DOCKER_REPO:-https://github.com/eugr/spark-vllm-docker.git}"
SPARK_VLLM_DOCKER_REF="${SPARK_VLLM_DOCKER_REF:-798528a2}"
VLLM_SRC="${VLLM_SRC:-$BUILD_ROOT/.src/vllm}"
B12X_SRC="${B12X_SRC:-$BUILD_ROOT/.src/b12x}"
PATCH_DIR="${PATCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/patches}"
WORKDIR="$BUILD_ROOT/spark-vllm-docker"      # our own scratch copy; never touch a checked-out spark-vllm-docker in place
DATE_TAG="$(date +%Y%m%d)"
PUSH=0
for arg in "$@"; do
    [ "$arg" = "--push" ] && PUSH=1
done

if [ "${CONFIRM_BUILD:-0}" != "1" ]; then
    echo "Refusing to run: this is a heavy build (compile, high RAM/disk) on a" >&2
    echo "host that may be serving or mid-boot. Re-run with CONFIRM_BUILD=1 once" >&2
    echo "the build host is confirmed idle (check 'docker ps', 'free -h', the pair watchdog)." >&2
    exit 1
fi

echo "== 0. clone/refresh vllm and b12x forks (skipped if a local clone is already present -- for hand-patching) =="
mkdir -p "$(dirname "$VLLM_SRC")"
[ -d "$VLLM_SRC/.git" ] || git clone "$VLLM_REPO" "$VLLM_SRC"
[ -d "$B12X_SRC/.git" ] || git clone "$B12X_REPO" "$B12X_SRC"

echo "== 1. sanity: clones present and clean =="
for d in "$VLLM_SRC" "$B12X_SRC"; do
    [ -d "$d/.git" ] || { echo "Missing clone: $d" >&2; exit 1; }
    if [ -n "$(git -C "$d" status --porcelain)" ]; then
        echo "Refusing: $d has uncommitted changes from a previous partial patch apply." >&2
        exit 1
    fi
done
git -C "$VLLM_SRC" fetch origin "$VLLM_REF" --quiet
git -C "$VLLM_SRC" checkout --quiet "FETCH_HEAD" 2>/dev/null || git -C "$VLLM_SRC" checkout --quiet "$VLLM_REF"
git -C "$B12X_SRC" fetch origin "$B12X_REF" --quiet
git -C "$B12X_SRC" checkout --quiet "FETCH_HEAD" 2>/dev/null || git -C "$B12X_SRC" checkout --quiet "$B12X_REF"

VLLM_COMMIT="$(git -C "$VLLM_SRC" rev-parse HEAD)"
B12X_COMMIT_PRE="$(git -C "$B12X_SRC" rev-parse HEAD)"
echo "vllm tip:  $VLLM_COMMIT ($VLLM_REPO@$VLLM_REF)"
echo "b12x tip:  $B12X_COMMIT_PRE ($B12X_REPO@$B12X_REF)"

echo "== 2. apply patches (if any) =="
shopt -s nullglob
for p in "$PATCH_DIR"/vllm-*.patch; do
    echo "applying $p to vllm"
    git -C "$VLLM_SRC" apply --check "$p"
    git -C "$VLLM_SRC" apply "$p"
done
for p in "$PATCH_DIR"/b12x-*.patch; do
    echo "applying $p to b12x"
    git -C "$B12X_SRC" apply --check "$p"
    git -C "$B12X_SRC" apply "$p"
done
shopt -u nullglob

# vLLM's commit id is asserted against VLLM_SOURCE_COMMIT inside the Dockerfile
# (see stage that uses `--build-context vllm_source=...`), so a patched tree
# needs a fresh commit on top rather than a dirty working tree.
if [ -n "$(git -C "$VLLM_SRC" status --porcelain)" ]; then
    git -C "$VLLM_SRC" commit -aqm "local: instrumentation/kernel patches (build.sh)"
fi
VLLM_SOURCE_COMMIT="$(git -C "$VLLM_SRC" rev-parse HEAD)"

# b12x has no equivalent local-source build-context in the stock Dockerfile
# (only vLLM gets the `FROM scratch AS vllm_source` bind-mount trick) --
# `git clone --branch $B12X_REF $B12X_REPO` is a plain remote clone. To build
# our patched b12x tree, apply patches/0001-dockerfile-b12x-local-source.patch
# (add it to patches/ if it does not exist yet) which mirrors the vllm_source
# stage for b12x. Until that patch exists, this script pins B12X_REF to the
# tip commit it cloned above and clones straight from GitHub, so a *ref* pin
# works today but a *local diff* to b12x does not reach the image yet.
B12X_REPO_ARG="$B12X_REPO"
B12X_REF_ARG="$B12X_COMMIT_PRE"
if [ -f "$PATCH_DIR/0001-dockerfile-b12x-local-source.patch" ]; then
    B12X_LOCAL_SOURCE=1
else
    B12X_LOCAL_SOURCE=0
    if [ -n "$(git -C "$B12X_SRC" log --oneline "$B12X_COMMIT_PRE"..HEAD)" ]; then
        echo "b12x has local patches but no Dockerfile local-source stage to carry" >&2
        echo "them into the image; add patches/0001-dockerfile-b12x-local-source.patch" >&2
        echo "(see patches/README.md) before patching b12x." >&2
        exit 1
    fi
fi

echo "== 3. fresh scratch copy of eugr/spark-vllm-docker, pinned =="
rm -rf "$WORKDIR"
git clone "$SPARK_VLLM_DOCKER_REPO" "$WORKDIR"
git -C "$WORKDIR" checkout --quiet "$SPARK_VLLM_DOCKER_REF"
echo "spark-vllm-docker pinned to $SPARK_VLLM_DOCKER_REF"

echo "== 3a. reapply submodule fix to fresh clone's Dockerfile (patch 2026-09-18: vllm-flash-attn's CMake FetchContent recursively clones AMD-only ROCm/aiter + ROCm/composable_kernel submodules over ssh with no creds -- fatal: could not read Username for https://github.com; idempotent so it survives every fresh clone) =="
python3 "$BUILD_ROOT/apply_submodule_fix.py" "$WORKDIR/Dockerfile"
grep -n "submodule.third_party/aiter.update none" "$WORKDIR/Dockerfile"

if [ "$B12X_LOCAL_SOURCE" = "1" ]; then
    git -C "$WORKDIR" apply --check "$PATCH_DIR/0001-dockerfile-b12x-local-source.patch"
    git -C "$WORKDIR" apply "$PATCH_DIR/0001-dockerfile-b12x-local-source.patch"
fi

VLLM_SHORT_SHA="$(git -C "$VLLM_SRC" rev-parse --short=8 HEAD)"
B12X_SHORT_SHA="$(git -C "$B12X_SRC" rev-parse --short=8 HEAD)"
IMAGE_TAG="spark-vllm-b12x:local-${DATE_TAG}-${B12X_SHORT_SHA}${IMAGE_TAG_SUFFIX:-}"

echo "== 4a. flashinfer wheel export (patch 2026-09-18: Dockerfile's runner stage mounts a build-context named flashinfer_wheels/vllm_wheels, not a pullable image -- original script omitted these two export phases that build-and-copy.sh normally runs first) =="
FI_DIR="$BUILD_ROOT/.wheel-cache/flashinfer"
if compgen -G "$FI_DIR"/flashinfer*.whl > /dev/null 2>&1; then
    echo "flashinfer wheels already present in $FI_DIR (from a prior run) -- skipping the ~65min rebuild:"
    ls -la "$FI_DIR"/*.whl
else
    mkdir -p "$FI_DIR"
    ( cd "$WORKDIR" && docker build --target flashinfer-export --output "type=local,dest=$FI_DIR" -f Dockerfile . )
fi

echo "== 4b. vllm wheel export (same reason; builds from our local vllm checkout) =="
VW_DIR="$BUILD_ROOT/.wheel-cache/vllm"
if compgen -G "$VW_DIR"/vllm-*.whl > /dev/null 2>&1 && [ "$(cat "$VW_DIR/.vllm-source-commit" 2>/dev/null)" = "$VLLM_SOURCE_COMMIT" ]; then
    echo "vllm wheel already present in $VW_DIR for commit $VLLM_SOURCE_COMMIT (from a prior run) -- skipping the ~20min rebuild:"
    ls -la "$VW_DIR"/*.whl
else
    rm -rf "$VW_DIR"; mkdir -p "$VW_DIR"
    ( cd "$WORKDIR" && docker build --target vllm-export --output "type=local,dest=$VW_DIR" \
        --build-arg "TORCH_VERSION=2.13.0" \
        --build-arg "TORCHVISION_VERSION=0.28.0" \
        --build-arg "TORCHAUDIO_VERSION=2.11.0" \
        --build-arg "VLLM_SOURCE_MODE=local" \
        --build-arg "VLLM_SOURCE_COMMIT=${VLLM_SOURCE_COMMIT}" \
        --build-arg "VLLM_APPLY_PRESET_PRS=0" \
        --build-arg "VLLM_PRESERVE_SM12X_TARGET=1" \
        --build-arg "VLLM_PATCH_B12X_C128A_ALIGNMENT=1" \
        --build-context "vllm_source=${VLLM_SRC}" \
        -f Dockerfile . )
    echo "$VLLM_SOURCE_COMMIT" > "$VW_DIR/.vllm-source-commit"
fi

echo "== 4b2. generate build-metadata.yaml (patch 2026-09-18: runner Dockerfile's last COPY step, 'COPY build-metadata.yaml /workspace/build-metadata.yaml', expects this file in the build context root -- build-and-copy.sh's generate_build_metadata() normally writes it, but we call docker build directly and skipped it) =="
BASE_IMAGE="$(grep -m1 '^FROM .* AS runner' "$WORKDIR/Dockerfile" | awk '{print $2}')"
cat > "$WORKDIR/build-metadata.yaml" <<EOF
build_date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
build_script_commit: local-build.sh
vllm_version: unknown
vllm_commit: ${VLLM_SOURCE_COMMIT}
flashinfer_commit: unknown
gpu_arch: 12.1a
base_image: ${BASE_IMAGE:-unknown}
build_args:
  vllm_repo: "local-source"
  vllm_ref: ${VLLM_SOURCE_COMMIT}
  torch_version: "2.13.0"
  torchvision_version: "0.28.0"
  torchaudio_version: "2.11.0"
  cutlass_dsl_version: "4.7.0"
  b12x_repo: "${B12X_REPO_ARG}"
  b12x_ref: "${B12X_REF_ARG}"
  b12x_from_pypi: 0
  transformers_5: false
  exp_mxfp4: false
  vllm_prs: ""
  build_jobs: 16
EOF
echo "Generated $WORKDIR/build-metadata.yaml"

echo "== 4c. docker build (direct -- build-and-copy.sh's --exp-b12x refuses --vllm-source-dir) =="
# Mirrors --exp-b12x's pins (EXP_B12X_TORCH_VERSION etc.) plus the local vLLM
# source-context mechanism build-and-copy.sh uses for --vllm-source-dir, which
# --exp-b12x's own CLI validation forbids combining -- so we call docker build
# directly instead of going through build-and-copy.sh.
BUILD_ARGS=(
    --build-arg "TORCH_VERSION=2.13.0"
    --build-arg "TORCHVISION_VERSION=0.28.0"
    --build-arg "TORCHAUDIO_VERSION=2.11.0"
    --build-arg "TORCH_CUDA_ARCH_LIST=12.1a"
    --build-arg "FLASHINFER_CUDA_ARCH_LIST=12.1a"
    --build-arg "VLLM_SOURCE_MODE=local"
    --build-arg "VLLM_SOURCE_COMMIT=${VLLM_SOURCE_COMMIT}"
    --build-arg "B12X_REPO=${B12X_REPO_ARG}"
    --build-arg "B12X_REF=${B12X_REF_ARG}"
    --build-arg "B12X_CACHEBUST=${DATE_TAG}"
    --build-context "vllm_source=${VLLM_SRC}"
    --build-context "flashinfer_wheels=${FI_DIR}"
    --build-context "vllm_wheels=${VW_DIR}"
    --build-context "draft_vocab=${BUILD_ROOT}/draft-vocab"
)
if [ "$B12X_LOCAL_SOURCE" = "1" ]; then
    BUILD_ARGS+=(--build-arg "B12X_SOURCE_MODE=local" --build-context "b12x_source=${B12X_SRC}")
fi

( cd "$WORKDIR" && docker build -t "$IMAGE_TAG" "${BUILD_ARGS[@]}" -f Dockerfile . )

echo "Built $IMAGE_TAG"
echo "  vllm source commit: $VLLM_SOURCE_COMMIT"
echo "  b12x ref:            $B12X_REF_ARG (local diff carried: $B12X_LOCAL_SOURCE)"

if [ -n "${DISTRIBUTE_TO:-}" ]; then
    echo "== 5. distribute to $DISTRIBUTE_TO (same docker save | ssh | docker load sparkrun itself uses) =="
    TMP_IMAGE="$(mktemp -t spark_vllm_b12x_local.XXXXXX)"
    docker save -o "$TMP_IMAGE" "$IMAGE_TAG"
    if ssh "$DISTRIBUTE_TO" "docker image inspect --format '{{.Id}}' '$IMAGE_TAG'" >/dev/null 2>&1; then
        echo "$IMAGE_TAG already present on $DISTRIBUTE_TO; skipping copy."
    else
        cat "$TMP_IMAGE" | ssh "$DISTRIBUTE_TO" "docker load"
    fi
    rm -f "$TMP_IMAGE"
else
    echo "== 5. DISTRIBUTE_TO not set, skipping cross-node distribution =="
fi

if [ "$PUSH" = "1" ]; then
    echo "== 6. push to ghcr.io (assumes 'docker login ghcr.io' already ran) =="
    OWNER="${OWNER:-ursuciprian}"
    REMOTE_SHA_TAG="ghcr.io/${OWNER}/spark-vllm-b12x:${VLLM_SHORT_SHA}-${B12X_SHORT_SHA}"
    REMOTE_LATEST_TAG="ghcr.io/${OWNER}/spark-vllm-b12x:latest"
    docker tag "$IMAGE_TAG" "$REMOTE_SHA_TAG"
    docker tag "$IMAGE_TAG" "$REMOTE_LATEST_TAG"
    docker push "$REMOTE_SHA_TAG"
    docker push "$REMOTE_LATEST_TAG"
    echo "Pushed $REMOTE_SHA_TAG and $REMOTE_LATEST_TAG"
fi

echo "== done =="
echo "Image '$IMAGE_TAG' is now present on this host${DISTRIBUTE_TO:+ and on $DISTRIBUTE_TO}."
echo "Point a recipe's 'container:' field at it (see recipes/eugr/eugr-agents-serve-local.yaml)"
echo "with build_args: [] so sparkrun uses it verbatim instead of pulling/building."
