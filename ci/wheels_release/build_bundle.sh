#!/usr/bin/env bash
# ci/wheels_release/build_bundle.sh
#
# Compile-once orchestrator for the wheels-release workflow: builds the
# flashinfer wheel, the vllm wheel and the b12x wheel on dgx-01 (self-hosted,
# real GPU), reuses build.sh's existing wheel-cache dirs so a second run for
# the same source commit is a no-op, bundles each as
# "<name>-cu134-<sha7>.tar.zst" + ".sha256", writes build-metadata.yaml, and
# publishes (or verifies-and-skips) a GitHub release.
#
# Env (all required unless noted):
#   VLLM_REF, B12X_REF, FLASHINFER_REF (informational; flashinfer wheel is
#     built from eugr's pinned Dockerfile stage, not a standalone flashinfer
#     checkout -- FLASHINFER_REF is recorded in metadata only)
#   TAG                 -- release tag to create/verify
#   REPO                -- "owner/name" for `gh release`
#   BUILD_ROOT          -- default $HOME/GEN-AI/build (shares build.sh's wheel-cache)
#   GH_TOKEN            -- must be exported by the caller for `gh`
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-$HOME/GEN-AI/build}"
VLLM_REPO="${VLLM_REPO:-https://github.com/ursuciprian/vllm.git}"
VLLM_REF="${VLLM_REF:?VLLM_REF required}"
B12X_REPO="${B12X_REPO:-https://github.com/ursuciprian/b12x.git}"
B12X_REF="${B12X_REF:?B12X_REF required}"
FLASHINFER_REF="${FLASHINFER_REF:-}"
TAG="${TAG:?TAG required}"
REPO="${REPO:?REPO required, e.g. ursuciprian/spark-vllm-b12x}"
SPARK_VLLM_DOCKER_REPO="${SPARK_VLLM_DOCKER_REPO:-https://github.com/eugr/spark-vllm-docker.git}"
SPARK_VLLM_DOCKER_REF="${SPARK_VLLM_DOCKER_REF:-798528a2}"

OUT_DIR="$(mktemp -d)/wheels-release"
mkdir -p "$OUT_DIR"
echo "== staging bundle in $OUT_DIR =="

echo "== 1. fresh scratch copy of eugr/spark-vllm-docker (patched), for the export stages =="
WORKDIR="$BUILD_ROOT/spark-vllm-docker"
rm -rf "$WORKDIR"
git clone --quiet "$SPARK_VLLM_DOCKER_REPO" "$WORKDIR"
git -C "$WORKDIR" checkout --quiet "$SPARK_VLLM_DOCKER_REF"
python3 "$REPO_ROOT/apply_submodule_fix.py" "$WORKDIR/Dockerfile"
DOCKERFILE_SHA="$(sha256sum "$WORKDIR/Dockerfile" | awk '{print $1}')"

echo "== 2. vllm source: clone, pin, apply patches (same as build.sh) =="
VLLM_SRC="$BUILD_ROOT/.src/vllm"
mkdir -p "$(dirname "$VLLM_SRC")"
[ -d "$VLLM_SRC/.git" ] || git clone --quiet "$VLLM_REPO" "$VLLM_SRC"
git -C "$VLLM_SRC" fetch --quiet origin "$VLLM_REF"
git -C "$VLLM_SRC" checkout --quiet FETCH_HEAD
# See build.sh's patch_already_carried()/apply_patch_if_needed() for the
# full explanation: a "<patch>.marker" file ("<path>:<grep-string>") flags a
# patch as already carried when the fork folds it into a commit that also
# touches nearby lines, so neither forward nor reverse apply cleanly.
patch_already_carried() {
    local src="$1" p="$2" marker="$p.marker"
    [ -f "$marker" ] || return 1
    local rule path needle
    rule="$(cat "$marker")"
    path="${rule%%:*}"
    needle="${rule#*:}"
    [ -f "$src/$path" ] && grep -q -- "$needle" "$src/$path"
}
shopt -s nullglob
if [[ "$VLLM_REPO" == *"ursuciprian/vllm"* ]] && [ "${APPLY_PATCHES:-0}" != "1" ]; then
    echo "VLLM_REPO is our fork (carries patches as commits) -- skipping patches/vllm-*.patch (set APPLY_PATCHES=1 to force)"
else
    for p in "$REPO_ROOT"/patches/vllm-*.patch; do
        if patch_already_carried "$VLLM_SRC" "$p"; then
            echo "marker matched ($(cat "$p.marker")), skipping: $p"
        elif git -C "$VLLM_SRC" apply --reverse --check "$p" 2>/dev/null; then
            echo "already applied (reverse-check matched), skipping: $p"
        elif git -C "$VLLM_SRC" apply --check "$p" 2>/dev/null; then
            echo "applying $p"
            git -C "$VLLM_SRC" apply "$p"
        else
            echo "patch does not apply forward or reverse, and no marker file matched -- source tree has diverged: $p" >&2
            exit 1
        fi
    done
fi
shopt -u nullglob
if [ -n "$(git -C "$VLLM_SRC" status --porcelain)" ]; then
    git -C "$VLLM_SRC" -c user.email=ci@localhost -c user.name=ci commit -aqm "ci: carried source patches (wheels-release)"
fi
VLLM_SOURCE_COMMIT="$(git -C "$VLLM_SRC" rev-parse HEAD)"
VLLM_SHORT_SHA="$(git -C "$VLLM_SRC" rev-parse --short=7 HEAD)"

echo "== 3. flashinfer wheel export (reuse build.sh's wheel-cache if present) =="
FI_DIR="$BUILD_ROOT/.wheel-cache/flashinfer"
if compgen -G "$FI_DIR"/flashinfer*.whl > /dev/null 2>&1; then
    echo "flashinfer wheel cache hit: $FI_DIR"
else
    mkdir -p "$FI_DIR"
    ( cd "$WORKDIR" && docker build --target flashinfer-export --output "type=local,dest=$FI_DIR" -f Dockerfile . )
fi

echo "== 4. vllm wheel export (reuse cache if commit matches) =="
VW_DIR="$BUILD_ROOT/.wheel-cache/vllm"
if compgen -G "$VW_DIR"/vllm-*.whl > /dev/null 2>&1 && [ "$(cat "$VW_DIR/.vllm-source-commit" 2>/dev/null)" = "$VLLM_SOURCE_COMMIT" ]; then
    echo "vllm wheel cache hit for $VLLM_SOURCE_COMMIT: $VW_DIR"
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

echo "== 5. b12x wheel: pure 'pip wheel', JIT-compiled kernels so no CUDA toolchain build needed (see ci/lil_wheels pattern) =="
# A fresh mktemp -d per run (not a fixed path under .src/) so nothing persists
# across runs to go stale/root-owned in the first place. Even so, `rm -rf` as
# the non-root nvidia runner user (no sudo) can still fail if the wheel
# build's container ever writes as root again (run 35461873991: bind-mounted
# build/ and *.egg-info came back root-owned) -- fall back to a throwaway
# root container to force-remove.
rm_rf_robust() {
    local dir="$1"
    [ -d "$dir" ] || return 0
    rm -rf "$dir" 2>/dev/null && return 0
    echo "plain rm -rf failed on $dir (likely root-owned leftovers) -- retrying via a root container" >&2
    docker run --rm -v "$dir:/w" alpine sh -c 'rm -rf /w/* /w/.[!.]* 2>/dev/null; true'
    rm -rf "$dir"
}
B12X_SRC="$(mktemp -d "$BUILD_ROOT/.wheel-cache/b12x-src.XXXXXX")"
trap 'rm_rf_robust "$B12X_SRC"' EXIT
git clone --quiet "$B12X_REPO" "$B12X_SRC"
git -C "$B12X_SRC" fetch --quiet origin "$B12X_REF"
git -C "$B12X_SRC" checkout --quiet FETCH_HEAD
B12X_COMMIT="$(git -C "$B12X_SRC" rev-parse HEAD)"
B12X_SHORT_SHA="$(git -C "$B12X_SRC" rev-parse --short=7 HEAD)"

# Resolve CUDA_IMAGE from the pinned Dockerfile's ARG default rather than
# grepping the runner stage's raw "FROM ${CUDA_IMAGE} AS runner" line, which
# returns the literal unresolved ARG reference and breaks `docker run` (see
# wheels-release run 35456233104: "invalid reference format: repository name
# (library/${CUDA_IMAGE}) must be lowercase"). Allow an env override; fail
# fast rather than let an empty value silently reach docker build/run.
CUDA_IMAGE="${CUDA_IMAGE:-$(sed -nE 's/^ARG CUDA_IMAGE=(.*)$/\1/p' "$WORKDIR/Dockerfile" | head -1)}"
: "${CUDA_IMAGE:?could not resolve ARG CUDA_IMAGE from $WORKDIR/Dockerfile and no CUDA_IMAGE env override set}"
echo "CUDA_IMAGE resolved to: $CUDA_IMAGE"

# Build the "base" stage (not the bare CUDA_IMAGE) so the b12x wheel is built
# against the same torch/CUDA env the final image uses; "base" already
# installs torch (Dockerfile ~line 81) so no separate builder stage is
# needed, and its own FROM ${CUDA_IMAGE} is resolved by docker build itself.
B12X_BASE_TAG="spark-vllm-base:${TAG}"
( cd "$WORKDIR" && docker build --target base -t "$B12X_BASE_TAG" \
    --build-arg "CUDA_IMAGE=${CUDA_IMAGE}" \
    --build-arg "TORCH_VERSION=2.13.0" \
    --build-arg "TORCHVISION_VERSION=0.28.0" \
    --build-arg "TORCHAUDIO_VERSION=2.11.0" \
    -f Dockerfile . )

B12X_WHEEL_DIR="$BUILD_ROOT/.wheel-cache/b12x"
rm -rf "$B12X_WHEEL_DIR"; mkdir -p "$B12X_WHEEL_DIR"
# Run as the host UID:GID (not root) so anything written into the bind-mounted
# $B12X_SRC/$B12X_WHEEL_DIR is owned by the runner user and stays deletable
# without sudo. A non-root UID has no /etc/passwd entry inside the container,
# so pip/setuptools need an explicit writable HOME (and pip cache dir) --
# a separate mktemp dir (not inside $B12X_SRC, so it can't end up swept into
# the wheel's sdist by an over-eager package-data glob), cleaned up with it.
B12X_HOME="$(mktemp -d "$BUILD_ROOT/.wheel-cache/b12x-home.XXXXXX")"
trap 'rm_rf_robust "$B12X_SRC"; rm_rf_robust "$B12X_HOME"' EXIT
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/home/build \
    -e PIP_CACHE_DIR=/home/build/pip-cache \
    -v "$B12X_SRC:/src" \
    -v "$B12X_HOME:/home/build" \
    -v "$B12X_WHEEL_DIR:/wheelhouse" \
    -w /src \
    "$B12X_BASE_TAG" \
    bash -lc 'python3 -m pip wheel --no-build-isolation --no-deps --wheel-dir /wheelhouse . && test "$(find /wheelhouse -maxdepth 1 -name "b12x-*.whl" | wc -l)" -eq 1'
echo "$B12X_COMMIT" > "$B12X_WHEEL_DIR/.b12x-source-commit"

echo "== 6. bundle each wheel dir as <name>-cu134-<sha7>.tar.zst + .sha256 =="
bundle() {
    local name="$1" src_dir="$2" sha7="$3"
    local archive="$OUT_DIR/${name}-cu134-${sha7}.tar.zst"
    tar --sort=name --owner=0 --group=0 --numeric-owner --zstd -C "$src_dir" -cf "$archive" $(cd "$src_dir" && find . -maxdepth 1 -name '*.whl' -printf '%P\n')
    sha256sum "$archive" | awk -v a="$(basename "$archive")" '{print $1"  "a}' > "$archive.sha256"
    echo "wrote $archive"
}
# flashinfer's actual resolved commit is only known inside the docker build
# (FLASHINFER_REF defaults to the Dockerfile's own ARG, "main", unless we
# override it, and we don't pass one to the flashinfer-export build above);
# the export stage writes it to /workspace/wheels/.flashinfer-commit, which
# flashinfer-export's "COPY --from=flashinfer-builder /workspace/wheels /"
# carries into FI_DIR -- including on a cache-hit run, since FI_DIR persists
# across runs from whenever it was last actually built. Fall back to
# git ls-remote of the pinned ref only if that marker is somehow missing.
# A non-hex sha7 would silently fail verify_release_assets.py's asset regex
# (exactly what broke run 35458650343's "unpinned" fallback), so validate.
if [ -f "$FI_DIR/.flashinfer-commit" ]; then
    FI_SHORT_SHA="$(cut -c1-7 "$FI_DIR/.flashinfer-commit")"
else
    echo ".flashinfer-commit marker missing in $FI_DIR -- falling back to git ls-remote" >&2
    FI_SHORT_SHA="$(git ls-remote https://github.com/flashinfer-ai/flashinfer.git "${FLASHINFER_REF:-main}" | awk '{print $1}' | head -1 | cut -c1-7)"
fi
[[ "$FI_SHORT_SHA" =~ ^[0-9a-f]{7}$ ]] || { echo "could not resolve a valid 7-hex flashinfer sha (got '$FI_SHORT_SHA')" >&2; exit 1; }

bundle flashinfer "$FI_DIR" "$FI_SHORT_SHA"
bundle vllm "$VW_DIR" "$VLLM_SHORT_SHA"
bundle b12x "$B12X_WHEEL_DIR" "$B12X_SHORT_SHA"

echo "== 6b. self-check: all 3 archives + sha256 sidecars present before writing metadata/verifying =="
for c in flashinfer vllm b12x; do
    n=$(compgen -G "$OUT_DIR/${c}-cu134-*.tar.zst" | wc -l)
    [ "$n" -eq 1 ] || { echo "expected exactly 1 ${c}-cu134-*.tar.zst in $OUT_DIR, found $n" >&2; exit 1; }
    archive="$(compgen -G "$OUT_DIR/${c}-cu134-*.tar.zst")"
    [ -f "$archive.sha256" ] || { echo "missing sidecar: $archive.sha256" >&2; exit 1; }
done
echo "all 3 component archives + sha256 sidecars present."

echo "== 7. build-metadata.yaml =="
cat > "$OUT_DIR/build-metadata.yaml" <<EOF
build_date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
build_script_commit: ci/wheels_release/build_bundle.sh
tag: ${TAG}
dockerfile_sha256: ${DOCKERFILE_SHA}
spark_vllm_docker_repo: ${SPARK_VLLM_DOCKER_REPO}
spark_vllm_docker_ref: ${SPARK_VLLM_DOCKER_REF}
torch_version: "2.13.0"
torchvision_version: "0.28.0"
torchaudio_version: "2.11.0"
vllm:
  repo: ${VLLM_REPO}
  ref: ${VLLM_REF}
  source_commit: ${VLLM_SOURCE_COMMIT}
b12x:
  repo: ${B12X_REPO}
  ref: ${B12X_REF}
  source_commit: ${B12X_COMMIT}
flashinfer:
  ref: ${FLASHINFER_REF}
EOF

echo "== 8. self-verify freshly built bundle =="
python3 "$REPO_ROOT/ci/wheels_release/verify_release_assets.py" --directory "$OUT_DIR"

echo "== 9. publish (or verify-and-skip if the tag already exists) =="
if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
    echo "release $TAG already exists on $REPO -- downloading to verify byte-identical assets"
    EXISTING_DIR="$(mktemp -d)"
    gh release download "$TAG" --repo "$REPO" --dir "$EXISTING_DIR"
    python3 "$REPO_ROOT/ci/wheels_release/verify_release_assets.py" --directory "$OUT_DIR" --reference-directory "$EXISTING_DIR"
    echo "existing release $TAG matches freshly built assets byte-for-byte; not re-publishing."
else
    gh release create "$TAG" --repo "$REPO" --prerelease \
        --title "spark-vllm-b12x wheels ${TAG}" \
        --notes "vllm ${VLLM_SHORT_SHA} / b12x ${B12X_SHORT_SHA} / flashinfer ${FI_SHORT_SHA}. See build-metadata.yaml for full pins." \
        "$OUT_DIR"/*.tar.zst "$OUT_DIR"/*.sha256 "$OUT_DIR/build-metadata.yaml"
    echo "published release $TAG on $REPO"
fi

echo "OUT_DIR=$OUT_DIR"
