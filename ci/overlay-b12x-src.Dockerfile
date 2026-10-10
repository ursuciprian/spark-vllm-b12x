# Thin overlay used by build-b12x-overlay.yml (hosted runner, no Spark involved):
#   b12x   swaps b12x inside an existing spark-vllm-b12x image for a b12x fork ref built from source and moves every
#          installed nvidia-cutlass-dsl* package to the version that ref pins.
#   vLLM   optional: copies the vllm/ files that differ between VLLM_BASE (the commit the base image's vLLM was built
#          from) and VLLM_REF over the installed package. Each file is checked against VLLM_BASE first, so a file
#          changed by an earlier layer (a mod) refuses the build instead of being overwritten silently.
#   seed   removes the baked torch AOT seed: AOT graphs carry b12x plan handles as constants and vLLM's compile key
#          does not include b12x, so a seed compiled against another b12x must never be loaded. The b12x plan-selection
#          seed stays; a b12x with another selection-cache schema reads its own file name and simply measures.
# Build context: the repo root.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG B12X_REPO=https://github.com/ursuciprian/b12x.git
ARG B12X_REF
ARG VLLM_REPO=https://github.com/ursuciprian/vllm.git
ARG VLLM_BASE=
ARG VLLM_REF=
COPY ci/cutlass_pins.py /tmp/cutlass_pins.py
RUN set -eux; test -n "$B12X_REF"; d=/tmp/b12x-src; git init -q $d; git -C $d remote add origin "$B12X_REPO"; \
    for i in 1 2 3 4 5; do git -C $d fetch -q --depth 1 origin "$B12X_REF" && break; sleep 10; done; \
    git -C $d checkout -q FETCH_HEAD; \
    pins=$(python3 /tmp/cutlass_pins.py $d/pyproject.toml); echo "cutlass pins: $pins"; test -n "$pins"; \
    uv pip install --no-cache --no-deps --reinstall $pins; \
    uv pip install --no-cache --no-deps --reinstall $d; \
    git -C $d rev-parse HEAD > /workspace/b12x-source-commit; \
    python3 -c "import importlib.metadata as m, b12x; print('b12x', m.version('b12x'), open('/workspace/b12x-source-commit').read().strip(), 'cutlass-dsl', m.version('nvidia-cutlass-dsl'))"; \
    rm -rf $d /tmp/cutlass_pins.py /opt/b12x-seed/torch_compile_cache; \
    echo "# /workspace/build-metadata.yaml predates this overlay: b12x and CUTLASS DSL are in b12x-source-commit and pip" >> /workspace/build-metadata.yaml || true
RUN set -eu; [ -n "$VLLM_REF" ] || { echo "no vLLM overlay"; exit 0; }; test -n "$VLLM_BASE"; \
    S=/usr/local/lib/python3.12/dist-packages; test -d $S/vllm; v=/tmp/vllm-src; git init -q $v; \
    git -C $v remote add origin "$VLLM_REPO"; \
    get() { for i in 1 2 3 4 5; do git -C $v fetch -q --filter=blob:none --depth 1 origin "$1" && { git -C $v rev-parse FETCH_HEAD; return 0; }; sleep 10; done; return 1; }; \
    base=$(get "$VLLM_BASE"); ref=$(get "$VLLM_REF"); \
    files=$(git -C $v diff --name-only --diff-filter=AM "$base" "$ref" -- vllm/); \
    gone=$(git -C $v diff --name-only --diff-filter=DR "$base" "$ref" -- vllm/); [ -z "$gone" ] || { echo "deleted/renamed files not supported: $gone"; exit 1; }; \
    for f in $files; do \
      if git -C $v cat-file -e "$base:$f" 2>/dev/null; then git -C $v show "$base:$f" | cmp -s - "$S/$f" || { echo "$S/$f differs from $base, refusing"; exit 1; }; \
      else [ ! -e "$S/$f" ] || { echo "$S/$f exists in the image but not at $base, refusing"; exit 1; }; fi; \
      mkdir -p "$(dirname "$S/$f")"; git -C $v show "$ref:$f" > "$S/$f"; chmod a+r "$S/$f"; echo "vLLM overlay: $f"; done; \
    echo "$ref" > /workspace/vllm-overlay-commit; rm -rf $v
