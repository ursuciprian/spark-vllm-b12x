import sys

p = sys.argv[1]
s = open(p).read()
changed = False

# Fix 1: vllm-flash-attn's CMake FetchContent recursively clones AMD-only
# ROCm/aiter + ROCm/composable_kernel submodules over ssh with no creds.
old1 = '''RUN --mount=type=cache,id=ccache,target=/root/.ccache \\
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv \\
    --mount=type=cache,id=cargo-registry,target=/opt/cargo/registry \\
    --mount=type=cache,id=cargo-git,target=/opt/cargo/git \\
    --mount=type=cache,id=vllm-rust-target,target=/workspace/vllm/vllm/target \\
    VLLM_REQUIRE_RUST_FRONTEND=1 CARGO_BUILD_JOBS=${MAX_JOBS} \\
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v'''
new1 = '''RUN --mount=type=cache,id=ccache,target=/root/.ccache \\
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv \\
    --mount=type=cache,id=cargo-registry,target=/opt/cargo/registry \\
    --mount=type=cache,id=cargo-git,target=/opt/cargo/git \\
    --mount=type=cache,id=vllm-rust-target,target=/workspace/vllm/vllm/target \\
    git config --global submodule.third_party/aiter.update none && \\
    git config --global url."https://github.com/".insteadOf "git@github.com:" && \\
    export GIT_TERMINAL_PROMPT=0 && \\
    VLLM_REQUIRE_RUST_FRONTEND=1 CARGO_BUILD_JOBS=${MAX_JOBS} \\
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v'''

if "submodule.third_party/aiter.update none" not in s:
    assert old1 in s, "submodule-fix anchor not found -- eugr changed the Dockerfile, needs a fresh look"
    s = s.replace(old1, new1)
    changed = True
    print("submodule fix applied")
else:
    print("submodule fix already present, skipping")

# Fix 2: `git clone --depth 1 --branch "$B12X_REF"` cannot resolve a bare
# 40-char commit SHA (only branch/tag refs) -- our B12X_REF is pinned to an
# exact commit for reproducibility. Use init+fetch+checkout instead, which
# GitHub supports for arbitrary SHAs even with --depth 1. Depth bumped to 200
# (not 1) so fix 3's `merge-base --is-ancestor` check below has enough local
# history to resolve PR #363's merge commit as an ancestor.
old2 = '''        git clone --depth 1 --branch "$B12X_REF" "$B12X_REPO" /tmp/b12x-source && \\'''
new2 = '''        mkdir -p /tmp/b12x-source && \\
        git -C /tmp/b12x-source init -q && \\
        git -C /tmp/b12x-source remote add origin "$B12X_REPO" && \\
        git -C /tmp/b12x-source fetch --depth 200 origin "$B12X_REF" && \\
        git -C /tmp/b12x-source checkout -q FETCH_HEAD && \\'''

if "git -C /tmp/b12x-source init -q" not in s:
    assert old2 in s, "b12x-clone-fix anchor not found -- eugr changed the Dockerfile, needs a fresh look"
    s = s.replace(old2, new2)
    changed = True
    print("b12x SHA-clone fix applied (depth 200)")
elif "fetch --depth 1 origin" in s and "git -C /tmp/b12x-source init -q" in s:
    # earlier version of this script used depth 1; bump it now that fix 3 needs history
    s = s.replace(
        '''git -C /tmp/b12x-source fetch --depth 1 origin "$B12X_REF" && \\''',
        '''git -C /tmp/b12x-source fetch --depth 200 origin "$B12X_REF" && \\''',
    )
    changed = True
    print("b12x SHA-clone fix depth bumped 1 -> 200")
else:
    print("b12x SHA-clone fix already present at sufficient depth, skipping")

# Fix 3: docker/b12x-pr363-small-tile-barriers.patch ("TEMPORARY: restore
# small-tile W4A8 occupancy until B12X PR #363 is merged", bundled from
# commit 9dc276f8) is REDUNDANT for our pinned b12x tip: PR #363 merged
# 2026-09-13T13:09:35Z as commit 05d2c43b35c81e18743b9e30e377fde1d49e2a8c
# (verified via `gh pr view 363 --repo local-inference-lab/b12x` -> MERGED;
# `compare/9dc276f8...a83336581` lists 05d2c43b among the 42 commits in
# between). The Dockerfile's own reverse-apply check fails to detect this
# because the file kept changing after the merge, so this adds an explicit
# ancestor check ahead of it.
old3 = '''        if git -C /tmp/b12x-source apply --reverse --check /tmp/b12x-pr363.patch >/dev/null 2>&1; then \\
            echo "B12X PR #363 is already applied; skipping."; \\'''
new3 = '''        if git -C /tmp/b12x-source merge-base --is-ancestor 05d2c43b35c81e18743b9e30e377fde1d49e2a8c HEAD 2>/dev/null; then \\
            echo "PR #363 already merged at 05d2c43b35c81e18743b9e30e377fde1d49e2a8c, skipping bundled patch."; \\
        elif git -C /tmp/b12x-source apply --reverse --check /tmp/b12x-pr363.patch >/dev/null 2>&1; then \\
            echo "B12X PR #363 is already applied; skipping."; \\'''

if "b12x-pr363" not in s:
    print("fix3: Dockerfile no longer bundles the PR#363 patch, nothing to do")
elif "PR #363 already merged at 05d2c43b" not in s:
    if old3 in s:
        s = s.replace(old3, new3)
        changed = True
        print("b12x PR#363 merge-base skip fix applied")
    elif "/tmp/b12x-pr363.patch" not in s:
        # eugr removed the whole bundled-patch reverse-apply step from the
        # Dockerfile (2026-09-19 check: b12x-source clone is now a plain
        # \`git clone --depth 1 --branch \$B12X_REF\` with no patch-apply
        # block at all). Nothing to skip-guard anymore -- fix 3 is moot.
        print("b12x PR#363 patch-apply step no longer present in Dockerfile; fix 3 is obsolete, skipping")
    else:
        raise AssertionError("b12x-pr363-merge-base-fix anchor not found -- eugr changed the Dockerfile, needs a fresh look")
else:
    print("b12x PR#363 merge-base skip fix already present, skipping")

if changed:
    open(p, "w").write(s)
else:
    print("no changes needed, file already fully patched")

# Fix 4 (2026-09-18, dv arm): copy the MTP draft-vocab table into the image at
# /opt/draft-vocab/draft_vocab_mia47k.pt so VLLM_QWEN_MTP_DRAFT_VOCAB_PATH
# resolves inside the container without a host bind-mount. Anchored right
# after the runner stage's last COPY step (build-metadata.yaml). The file
# itself comes in via a separate --build-context "draft_vocab=..." (added to
# build.sh phase 4c BUILD_ARGS), not copied into this scratch clone's tree,
# so the build context stays the plain eugr/spark-vllm-docker checkout.
old4 = "COPY build-metadata.yaml /workspace/build-metadata.yaml"
new4 = ("COPY build-metadata.yaml /workspace/build-metadata.yaml\n"
        "COPY --from=draft_vocab draft_vocab_mia47k.pt /opt/draft-vocab/draft_vocab_mia47k.pt")

if "COPY --from=draft_vocab draft_vocab_mia47k.pt" not in s:
    assert old4 in s, "draft-vocab COPY anchor not found -- eugr changed the runner stage, needs a fresh look"
    s = s.replace(old4, new4)
    changed = True
    print("draft-vocab COPY fix applied")
else:
    print("draft-vocab COPY fix already present, skipping")

if changed:
    open(p, "w").write(s)
else:
    print("no changes needed (fix 4 check), file already fully patched")
