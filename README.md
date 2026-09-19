# spark-vllm-b12x

Our own vLLM+b12x image for a 2x DGX Spark pair (Qwen3.8-Flash-Next),
CI-built on a self-hosted Spark runner. Built from eugr's Dockerfile plumbing,
on top of our own forks of local-inference-lab's vLLM and b12x, with three
source patches applied that used to live as post-install mods in the recipe
repo.

## Pins

- **Dockerfile / build plumbing**: [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)
  pinned to `798528a2` (2026-09-16), unchanged except for the submodule fix in
  `apply_submodule_fix.py` (vllm-flash-attn's CMake FetchContent recursively
  clones AMD-only ROCm submodules over ssh with no creds; this neuters that
  update). Override with `SPARK_VLLM_DOCKER_REPO` / `SPARK_VLLM_DOCKER_REF`.
- **vLLM fork**: [ursuciprian/vllm](https://github.com/ursuciprian/vllm)
  branch `dgx-spark`, based on local-inference-lab/vllm `8e1f1e58`
  (`dev/jovian-judgement`, 2026-09-16) plus three commits:
  1. `fix(parser): honor tool_choice=required for models on the shared ParserEngine`
  2. `fix(b12x-startup): bound PrefixStore waits in preparation exchange`
  3. `feat(mtp): optional reduced-vocab draft head via VLLM_QWEN_MTP_DRAFT_VOCAB_PATH`
     (experimental, inert unless the env var is set)
  Override with `VLLM_REPO` / `VLLM_REF`.
- **b12x fork**: [ursuciprian/b12x](https://github.com/ursuciprian/b12x)
  branch `dgx-spark` = `a8333658` (2026-09-17, the serving pin), plus a side
  branch `exp/fwd-57f3572` (`a8333658` + cherry-picked `57f3572`, MoE-only
  NVFP4 A16 autotuning restore) kept for experimentation, not the default.
  Override with `B12X_REPO` / `B12X_REF`.

## Image tag scheme

Local builds: `spark-vllm-b12x:local-<YYYYMMDD>-<b12x-short-sha>`.
Pushed (with `--push`): `ghcr.io/<owner>/spark-vllm-b12x:<vllm-short-sha>-<b12x-short-sha>`
and `:latest`.

## Usage

```bash
# local build only (no push, no cross-node distribution)
CONFIRM_BUILD=1 ./build.sh

# build + push to ghcr.io (needs `docker login ghcr.io` first)
CONFIRM_BUILD=1 ./build.sh --push

# also copy the built image onto a second cluster node
DISTRIBUTE_TO=192.168.100.53 CONFIRM_BUILD=1 ./build.sh
```

All of `VLLM_REPO`, `VLLM_REF`, `B12X_REPO`, `B12X_REF`,
`SPARK_VLLM_DOCKER_REPO`, `SPARK_VLLM_DOCKER_REF`, `DISTRIBUTE_TO`, and
`OWNER` (ghcr.io namespace, default `ursuciprian`) are environment
overridable; see the comment block at the top of `build.sh`.

`patches/` carries the one source patch that has not yet been folded into the
vLLM fork's own history in a way CI can apply automatically, plus notes on
every mod-to-source translation done for this build in
`patches/README.md`.

**`patches/` is only for building from an upstream ref that doesn't have the
change yet.** Once a patch is merged into a fork branch (`dgx-spark`, etc.) as
a real commit, applying it again would fail — every caller (`build.sh`,
`ci/wheels_release/build_bundle.sh`, `build-image-hosted.yml`) checks
`git apply --reverse --check` first and skips a patch that's already present,
so the same `patches/*.patch` files work unmodified whether the target ref
already carries the commit or not.

## How the recipes consume this

The [qwen3.8-flash-next-dgx-spark-tp-2](https://github.com/ursuciprian/qwen3.8-flash-next-dgx-spark-tp-2)
recipe repo's `eugr-agents-serve-local16-la.yaml` (and siblings) point their
`container:` field directly at a tag this repo builds, with `build_args: []`
so sparkrun uses the image verbatim instead of pulling or building it itself.

## CI

`.github/workflows/build-image.yml` runs on a self-hosted `dgx-spark` runner
(`workflow_dispatch` or a `v*` tag push), with a preflight step that refuses
to build while something is being served on the runner host (`localhost:8000/health`
returns 200) unless `allow_while_serving` is set — a full build is a >1h,
high-RAM/CPU/disk compile that would starve a serving process.
