# Patches for the local b12x/vLLM build

Drop `*.patch` files here (unified diff, `git apply` compatible) to be applied,
in filename sort order, to the matching clone before the image is built:

- `vllm-*.patch`  -> applied inside `~/GEN-AI/build/vllm` (branch dev/jovian-judgement)
- `b12x-*.patch`  -> applied inside `~/GEN-AI/build/b12x` (branch master)

`build.sh` applies with `git apply --check` first (abort on failure) then
`git apply`. Empty directory = no patches, clones build as-is.

Known future candidates (see kernel-sweep-2026-09-18.md item 4, not written yet):
- per-row debug dump in `b12x/attention/paged/_selected_forward.py` (the CuTe
  QSA verify+combine kernels, `kernel()` defs at lines 567/881/984) gated by
  `B12X_QSA_DEBUG_ROW` env var, writing request_id/query_position/partial_lse
  for one target row per launch.
- per-row trace in `b12x/attention/qsa/_draft_selection.py` `_record_kernel`
  (line 32) and `_prepare_kernel` (line 114): both are Triton, so a debug
  branch can `tl.store` into an extra debug tensor without touching the CuTe
  kernel at all.

## build.sh scratch-clone Dockerfile edit (2026-09-18, applied by build.sh's run, not a patches/*.patch file)
The vllm-builder stage's final `uv build --wheel` RUN line was hand-edited in
`~/GEN-AI/build/spark-vllm-docker/Dockerfile` (the scratch clone build.sh
creates fresh each run, NOT this shared `~/spark-vllm-docker`) to add:
`git config --global submodule.third_party/aiter.update none && git config
--global url."https://github.com/".insteadOf "git@github.com:" && export
GIT_TERMINAL_PROMPT=0 &&` immediately before the `uv build` invocation.
Root cause: vllm-flash-attn's CMake FetchContent recursively runs `git
submodule update --init --recursive`, which tries to clone AMD-only
submodules (ROCm/aiter, ROCm/composable_kernel) not needed for our CUDA
build; git failed with `fatal: could not read Username for
'https://github.com': No such device or address` because those subsubmodule
URLs use ssh (git@github.com:) with no credentials available in the build
container. `submodule.third_party/aiter.update none` skips that submodule
entirely; the insteadOf/GIT_TERMINAL_PROMPT lines are defensive in case any
other submodule in the tree also uses an ssh URL. build.sh clones a FRESH
scratch copy of spark-vllm-docker every run (`rm -rf "$WORKDIR"; git clone
--depth 1 ...`), so this Dockerfile edit does NOT persist automatically —
whoever re-runs build.sh from scratch must re-apply it (or turn this into a
real patches/*.patch file the README already documents applying, see the top
of this file for the convention).

## b12x-pr363-small-tile-barriers.patch skip fix (2026-09-18, applied by apply_submodule_fix.py fix 3)
The runner stage bundles `docker/b12x-pr363-small-tile-barriers.patch`
("TEMPORARY: restore small-tile W4A8 occupancy until B12X PR #363 is
merged", bundled from commit 9dc276f8105cfbe2d5882a6475e6e96c9533911c).
Verified via `gh pr view 363 --repo local-inference-lab/b12x`: state MERGED,
mergedAt 2026-09-13T13:09:35Z, merge commit
05d2c43b35c81e18743b9e30e377fde1d49e2a8c ("fix(moe): specialize input-pair
barriers for small tiles (#363)"); `compare/9dc276f8...a83336581` lists
05d2c43b among the 42 commits between the patch base and our pinned b12x tip
(a83336581a3a907076e60797df69ab66df5a2ff1). The patch is therefore redundant
for any B12X_REF at or after 05d2c43b, but the Dockerfile own
"already applied, reverse-check" branch cannot detect this because
b12x/moe/_shared/kernels/dynamic.py kept changing after the merge (neither
forward nor reverse apply is clean). Fix: added an explicit
`git merge-base --is-ancestor 05d2c43b... HEAD` check ahead of the existing
reverse-apply check, which skips the patch with an explicit log line when
the merge commit is an ancestor of our pinned ref. Also bumped the b12x
source fetch depth from 1 to 200 (fix 2 above) since this ancestor check
needs local history far enough back to include 05d2c43b -- 200 comfortably
covers the 42-commit gap. If PR #363 is ever reverted or a future B12X_REF
predates the merge, the original apply logic still runs unchanged.

## `vllm-dv-devicefix` -- dv arm (reduced-vocab MTP draft head), BLOCKED (2026-09-18)

`vllm-0001-mtp-draft-vocab.patch` was built into
`spark-vllm-b12x:local-20260918-a8333658-dv` (built clean, ~8 min thanks to a
warm flashinfer wheel cache; the vllm wheel cache correctly busted since
applying the patch produces a new source commit, `f852670` vs base `8e1f1e5`
-- no manual cache-bust needed, see build.sh phase 4b's commit-hash guard).
The image booted and the draft-vocab table
(`draft-vocab/draft_vocab_mia47k.pt`, 47,149 ids) baked in at
`/opt/draft-vocab/draft_vocab_mia47k.pt` load correctly, but the mechanism
itself failed three separate boots. Each was fixed as a runtime mod
(`recipes/eugr/mods/vllm-dv-devicefix/`, ships a corrected
`vllm/models/qwen3_8_flash_next/mtp.py` and copies it over the installed
file -- no rebuild needed to retest) so the arm could be gated without
burning another ~8min build cycle per iteration. The mod files are kept in
`mods/` (not deleted) for whoever resumes this.

**Boot 1** (original patch as authored): `RuntimeError: Expected all tensors
to be on the same device, but found at least two devices, cuda:0 and cpu!`
in `register_buffer("draft_id_to_target_id", target_ids -
torch.arange(self.draft_vocab_size, dtype=torch.int64), ...)` --
`torch.arange` defaulted to CPU while `target_ids` (loaded via `torch.load`
then moved to the model device) was already on `cuda:0`.
Fix: `device=target_ids.device` on the `torch.arange` call.

**Boot 2** (+ device fix, primary lm_head still resized to
`draft_vocab_size`): `NotImplementedError: checkpoint routing performed an
unsupported data transformation` at `b12x/loader/_checkpoint.py:395`. Root
cause: resizing the *primary* `lm_head`/`logits_processor` to
`draft_vocab_size` and then `index_select`-slicing the checkpoint's
`lm_head.weight` during `load_weights` gives b12x's checkpoint router a
declared param shape that does not match the checkpoint tensor's shape at
that name. b12x's router is not a general reshape/slice engine and refuses
any such mismatch outright, regardless of whether the slice is "valid" from
vLLM's own perspective.
Fix (redesign): keep the primary `lm_head`/`logits_processor` at full
`config.vocab_size` (checkpoint loads unmodified -> shapes match -> b12x's
router is happy). Add a separate `draft_lm_head` (`ParallelLMHead` sized
`draft_vocab_size`) that never appears in the checkpoint weight stream at
all (so b12x's router never inspects it, never rejects it), populated in a
new `_populate_draft_lm_head()` called at the end of `load_weights`: all
gather the now-fully-loaded primary `lm_head.weight` across TP, slice to
`config.vocab_size` real rows, `index_select` the draft-vocab rows (in
table order), then copy this rank's contiguous slice (per
`draft_lm_head.shard_indices.org_vocab_start_index`) into
`draft_lm_head.weight.data`. A new `get_top_tokens()` override on
`Qwen3_8FlashNextMTP` routes the draft argmax through `draft_lm_head`
instead of the (now full-vocab) primary `lm_head`, so
`LocalArgmaxMixin`'s D2T remap (`k + draft_id_to_target_id[k]`) still lines
up against a `draft_vocab_size`-wide `k`.

**Boot 3** (device fix + draft_lm_head split): `RuntimeError: The size of
tensor a (2560) must match the size of tensor b (1280) at non-singleton
dimension 1` -- a TP shard mismatch in the reduced draft head. 2560 = 2x1280
(TP=2 world size), so this is a full-vs-per-rank hidden-dimension mismatch
somewhere in the new `draft_lm_head`/`_populate_draft_lm_head` path: either
`self.lm_head.weight` (after `tensor_model_parallel_all_gather`) carries a
gathered *hidden* dimension where a per-rank one was expected (the fix in
boot 2 all-gathers on `dim=0` -- the vocab dimension -- but if `hidden_size`
is itself sharded on this checkpoint's `lm_head` layout, e.g. an
output-parallel scheme where dim 1 is also partition-dependent, gathering
only dim 0 leaves a dim-1 mismatch against `draft_lm_head`'s own per-rank
hidden allocation), or the `F.linear`/matmul inside
`LogitsProcessor.get_top_tokens`'s `_apply_head` multiplies a full-hidden
`hidden_states` against a half-hidden `draft_lm_head.weight` (or vice
versa) because `draft_lm_head`'s hidden dimension was declared/allocated
inconsistently with how `hidden_states` arrives at that call (TP output vs
TP input side of the attention/MoE stack -- this model's hidden state may
already be a TP-local shard by the time `get_top_tokens` runs, and a plain
`ParallelLMHead(draft_vocab_size, config.hidden_size, ...)` construction
assumes the *unsharded* `config.hidden_size`).

**Status: BLOCKED, not resolved tonight.** Three boot attempts is the stop
condition. `eugr-agents-serve-local16.yaml` is the fallback and is what is
serving now. Do not attempt a 4th dv boot without first tracing the actual
tensor shapes at the `draft_lm_head` call site (add a one-line shape print
or use `VLLM_SPEC_TRACE`-style instrumentation) rather than guessing further
from the error message alone -- the two live hypotheses above (all-gather
dimensionality vs hidden-state TP-locality mismatch) are not distinguished
by anything captured so far.

Files: `mods/vllm-dv-devicefix/{run.sh,mtp_fixed.py}` (boot-3 version, kept
for the next attempt), `patches/vllm-0001-mtp-draft-vocab.patch` (carries
both the boot-1 and boot-2 fixes, i.e. matches boot 3's state, still fails
the way boot 3 failed -- do not rebuild from it without also fixing the TP
shard issue first, or the same crash reproduces).

## dv arm 2026-09-19 follow-up: boot fixed (2 real bugs), gated, NOT PROMOTED (0% MTP acceptance)

Picked up the 2026-09-18 BLOCKED state (boot 3 crash: `RuntimeError: size of
tensor a (2560) must match ... b (1280) at non-singleton dimension 1` in
`_populate_draft_lm_head`). Root cause was **not** TP hidden-sharding as
boot 3's writeup speculated -- confirmed via one-shot shape-debug boot
(temporary log line, since removed): `self.lm_head.weight.shape=(124160,
1280) dtype=torch.uint8 quant_method=Nvfp4OnlineLinearMethod` vs
`draft_head.weight.shape=(23584, 2560) dtype=bf16`. `VLLM_MTP_NVFP4_LM_HEAD`
defaults to `1`, so the MTP module's own primary `lm_head` is
runtime-quantized to NVFP4 (`Nvfp4OnlineLinearMethod.process_weights_after_loading`
packs 2 4-bit values/byte along the hidden dim, physically halving the last
dim: 2560 logical -> 1280 packed bytes) *during* `AutoWeightsLoader`'s own
weight-loading pass, i.e. **before** `_populate_draft_lm_head()` ever reads
`self.lm_head.weight.data`, not in the later `process_weights_after_loading`
pass base_loader.py runs on the whole model. `draft_lm_head` was built
without a matching `lm_head_quantization=` kwarg, so it stayed plain bf16
(2560) -- a packed-vs-unpacked format mismatch, not a TP-sharding one.

**Fix 1 (boot4/5 fix, applied):** since `get_top_tokens` is fully overridden
to route the draft argmax through `draft_lm_head` (the speculator
(`spec_decode/speculator.py:374-388`) only falls back to
`compute_logits()`/the primary `lm_head` when `get_top_tokens` is absent),
the primary head's own NVFP4 quantization contributes nothing on the decode
hot path once `draft_lm_head` exists -- it's read exactly once, at
`_populate_draft_lm_head()` time. Set `self.has_own_lm_head = False`
immediately after the reduced-vocab-draft-head block in `__init__` so both
heads stay plain bf16 with matching hidden dims. This got the boot past the
tensor-shape crash cleanly (`_populate_draft_lm_head` logged "draft_lm_head
populated, rows [0:23584) of 47149" -- correct shard row-count for TP=2).

**Fix 2 (boot5 fix, applied, was actually boot-1's original unresolved bug):**
next crash, past weight-loading, during the CUDA-graph profile/dummy run:
`RuntimeError: indices should be either on cpu or on the same device as the
indexed tensor (cpu)` in `get_top_tokens`'s `top = top + d2t[top]`. Boot 1's
"device fix" (`device=target_ids.device` on the `torch.arange` in
`register_buffer("draft_id_to_target_id", ...)`) only made `torch.arange`
match `target_ids`'s device -- but `target_ids` itself was loaded via
`torch.load(draft_vocab_path, map_location="cpu", ...)` and *never moved to
GPU*, so `draft_id_to_target_id` was silently CPU-resident the whole time.
`register_buffer()` takes the device of the tensor handed to it; it does not
move to the module's ambient device context. Boots 1-3 never reached the
place this surfaces (`d2t[top]`, indexed with GPU indices) because they
crashed earlier in `__init__`/`load_weights`. Fix: move `target_ids` to
`next(self.model.parameters()).device` right after the dtype cast, before
building the buffer.

**Both fixes applied only in the hot-patch mod**
(`recipes/eugr/mods/vllm-dv-devicefix/mtp_fixed.py`) -- no image rebuild.
Diff against the boot-3 file is exactly these two changes (verified with
`diff` against the pre-existing `.bak-preinstrument` copy of the boot-3
file). `vllm-0001-mtp-draft-vocab.patch` was **not** touched this session
since the arm is not promoted (see below); fold both fixes into the patch
+ regenerate + rebuild only if a future session actually gets this arm
gated and promoted.

**Boot 4 (today, dv-boot6.log): PASSED.** Health 200, one chat completion
returned coherent text (haiku request, no garbage/token-spam), engine init
81.17s, no crash. This is the first-ever clean dv boot.

**Gate run (today, results/arms/dv/): started, aborted early -- decisive
failure on the primary gate criterion before the expensive stages finished.**
`/metrics` and a dedicated `straggler.py` probe (rebuilt this session --
the original `/tmp/straggler.py` referenced by `gate_arm.sh` was a prior
session's ephemeral `/tmp` file, not versioned in the repo, and no longer
exists; rebuilt to the same output format from `results/arms/la/straggler.log`,
not committed anywhere since `/tmp` isn't part of this repo) both show:

  MTP acceptance: **0.00 accepted / draft** at every concurrency tested
  (c=5,6,7,8,12,16 -- 0/79 in one early check, 7/151120 cumulative over the
  full gate run before it was stopped, ~1/24 on one isolated 8-token
  request). La's baseline is 61% cumulative / 81-62-48-37% per position.
  Wall-clock at c5 was 21.1s vs la's 5.4s (~4x slower) -- with 0%
  acceptance every draft round pays pure overhead for zero payoff, so this
  arm is a straightforward regression, not just "no speedup."

  Correctness held throughout: `fidelity_probe.py` 20/20 exact at all four
  depths tested (8k/32k/64k/128k) -- verification always corrects whatever
  the (broken) draft head proposes, so 0% acceptance costs speed, not
  output quality. One manual chat-completion sanity check also produced
  clean, non-garbled text.

**Root cause of the 0% acceptance: not found, arm remains BLOCKED on this
axis.** Investigated and ruled out, in order:

  1. NOT the TP-shard/quantization-packing bug from earlier boots (fixed,
     verified via clean weight-load log line).
  2. NOT the CPU-buffer device bug (fixed, verified -- no more device-mismatch
     crash, and per-request `/metrics` deltas do register nonzero drafts and
     occasional accepts, so the buffer is now live and on-device).
  3. NOT an out-of-range/padding vocab id issue: `target_ids_max=248076`
     initially looked suspicious (close to the padded per-rank-doubled
     124160*2=248320 seen in the earlier NVFP4 debug log) but this
     checkpoint's real `text_config.vocab_size` **is** 248320 (confirmed via
     the model's `config.json`), so ids up to 248076 are legitimately
     in-range, not garbage/padding rows.
  4. NOT an accidental re-quantization of `draft_lm_head` by the model's own
     `modelopt_mixed` static quant_config (`quant_config=self.quant_config`
     is passed to `draft_lm_head`'s `ParallelLMHead` constructor): traced
     `ModelOptMixedConfig._resolve_quant_algo()` / `get_quant_method()` --
     lookup is by exact checkpoint layer name in `quantized_layers`;
     `draft_lm_head` never appears in the checkpoint, so `quant_algo` is
     `None` and it falls through to plain `UnquantizedLinearMethod()`. No
     stray scale-factor-zero corruption.
  5. Attempted a live per-step trace (`VLLM_DV_TRACE=1` env, rank-0-only
     `logger.info` of the draft-local argmax index, the D2T-remapped target
     id, and target_ids min/max, gated on
     `not torch.cuda.is_current_stream_capturing()` after a first attempt
     without that guard broke CUDA-graph capture with "Cannot copy between
     CPU and CUDA tensors during CUDA graph capture"). **Inconclusive**: the
     only trace lines that fired were during the warmup/profile dummy-run
     pass building the CUDA graphs (interleaved with "Capturing prefill CUDA
     graphs" in the log) -- constant `draft_local_argmax=0` there, but that
     is expected/uninformative for a dummy-hidden-state warmup pass, not
     evidence of a real bug. Real decode after the server came up (verified
     via one actual `/v1/chat/completions` request) produces varying,
     nonzero `/metrics` deltas (1 accepted / 24 drafted on that one request),
     so the mechanism is *not* permanently stuck returning a constant -- it's
     just wrong far too often to be useful. No Python-level visibility into
     real per-step predictions was obtained because real decode replays a
     captured CUDA graph and never re-enters this Python code path.
     Diagnostic instrumentation was reverted after this session (see diff
     above); `mtp_fixed.py` is back to exactly the two real fixes, no trace
     code, no leftover env var in the recipe.

  **Live hypotheses for whoever resumes this** (none confirmed):
  - Run with `--enforce-eager` (accept the perf hit) specifically to get
    real per-step Python visibility into `get_top_tokens`'s draft-local
    argmax vs the D2T-remapped id vs the actual verified/accepted token,
    since CUDA-graph replay defeated that this session.
  - The `draft_vocab_mia47k.pt` table's row-order/content vs. this exact
    checkpoint's tokenizer is external, unverified provenance (MiaAI-Lab
    AGPL table, converted to `.pt` by a prior session) -- a row-order bug
    or tokenizer-vocab mismatch in that conversion would look exactly like
    this (structurally-sound remap math, near-chance-level real acceptance)
    and is not something a python-only `mtp.py` mod can fix or that this
    session had tooling to independently re-derive/verify.
  - Re-check whether `draft_lm_head`'s hidden-state input at inference time
     (post cudagraph capture, on the model's real forward path) is on the
    same normalization/projection basis as what `_populate_draft_lm_head`
    assumed when it sourced rows from the primary `lm_head`'s weight --
    the remap math and shapes are provably self-consistent by construction
    (same `shard_indices.org_vocab_start_index` used on both the populate
    and lookup sides), so if there is a remaining bug it's more likely in
    the *data* (the table, or a stray host mismatch between the row and
    hidden basis) than in this session's TP/index arithmetic.

**Verdict: NOT PROMOTED.** Gate criteria (fidelity within 5pts: PASS;
hardmode >= 88: not run, moot; MTP acceptance not worse than la: FAIL, 0%
vs 61-81%; straggler clean: FAIL, 4x wall-clock regression at c5) --
decisively failed on the acceptance/speed axis alone; the expensive
hardmode/categories/sweep stages were stopped early once this was clear
(coordinator direction, to avoid burning ~tens of minutes of tool-eval on a
already-decided outcome).

**Now serving:** `eugr-agents-serve-local16-la.yaml` (la), booted clean,
health 200. One `bench_sweep.py --levels 1` check: 87.5 tok/s then 85.4
tok/s (2 runs) -- about 11-13% below the documented la baseline of 98.2
tok/s, but within this pair's documented 15-25% boot-to-boot variance band;
not treated as a regression signal on a single boot.

Files touched this session: `recipes/eugr/mods/vllm-dv-devicefix/mtp_fixed.py`
(net: two real fixes, no diagnostic residue), `/tmp/straggler.py` on dgx-01
(rebuilt, not part of the repo, ephemeral). `recipes/eugr/eugr-agents-serve-local-dv.yaml`
and `patches/vllm-0001-mtp-draft-vocab.patch` unchanged (arm not promoted).
