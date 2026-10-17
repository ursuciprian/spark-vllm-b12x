"""Install a FlashInfer source checkout as a side package under another top-level name.

Used by overlay-b12x-src.Dockerfile (flashinfer_ref): copies <src>/flashinfer to <dest> (e.g. flashinfer_k114) and
rewrites its absolute imports, quoted module paths and torch custom-op namespaces to the new name, so it loads next to
the image's FlashInfer without touching it: the image's sampler and other FlashInfer users keep their build and AOT
jit-cache, and FlashInfer's `b12x` compatibility package (flashinfer/experimental/b12x/_compat) is dropped, so the
real b12x is never shadowed. No pip, no dependencies: the image already carries what the copied code imports.
Self-check: python3 ci/vendor_flashinfer.py --selftest
Usage: python3 ci/vendor_flashinfer.py <flashinfer checkout> <dest dir> <commit>
"""
import os
import re
import shutil
import sys

OLD = "flashinfer"


def rewrite(text: str, new: str) -> str:
    def imp(m):
        ind, mod, alias = m.group(1), m.group(2) or "", m.group(3)
        if alias:
            return f"{ind}import {new}{mod}{alias}"
        if not mod:
            return f"{ind}import {new} as {OLD}"
        return f"{ind}import {new}{mod}; import {new} as {OLD}"   # `import flashinfer.x` binds the name flashinfer
    text = re.sub(rf"(?m)^([ \t]*)import {OLD}((?:\.\w+)*)((?:[ \t]+as[ \t]+\w+)?)[ \t]*$", imp, text)
    text = re.sub(rf"(?m)^([ \t]*)from {OLD}(\.|\s)", rf"\1from {new}\2", text)
    text = re.sub(rf"""(["']){OLD}([.:"'])""", rf"\1{new}\2", text)   # "flashinfer.x", "flashinfer::op", "flashinfer"
    text = re.sub(r"""(["'])trtllm::""", rf"\1trtllm_{new.rsplit('_', 1)[-1]}::", text)
    text = re.sub(r"\btorch\.ops\.(flashinfer|trtllm)\.", lambda m: f"torch.ops.{new if m[1] == OLD else 'trtllm_' + new.rsplit('_', 1)[-1]}.", text)
    return text


def vendor(src: str, dest: str, commit: str) -> int:
    new = os.path.basename(dest.rstrip("/"))
    if os.path.exists(dest):
        raise SystemExit(f"{dest} exists")
    shutil.copytree(os.path.join(src, OLD), dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.rmtree(os.path.join(dest, "experimental", "b12x", "_compat"), ignore_errors=True)
    n = 0
    for root, _, files in os.walk(dest):
        for f in files:
            if f.endswith(".py"):
                p = os.path.join(root, f)
                t = open(p, encoding="utf-8").read()
                r = rewrite(t, new)
                if r != t:
                    open(p, "w", encoding="utf-8").write(r)
                    n += 1
    with open(os.path.join(dest, "_build_meta.py"), "w") as fh:   # a version unlike the image's: its cubin/jit-cache are not loaded
        fh.write(f'__version__ = "0.7.2.dev0+{commit[:8]}"\n__git_commit__ = "{commit}"\n')
    return n


def selftest():
    t = ("import flashinfer\nimport flashinfer.jit.env\nimport flashinfer.utils as fu\n    from flashinfer.jit import x\n"
         "from flashinfer import y\nm = importlib.import_module('flashinfer.sampling')\n"
         "@torch.library.custom_op(\"flashinfer::fp4_quantize\", mutates_args=())\n@torch.library.custom_op(\"trtllm::reswizzle_sf\")\n"
         "torch.ops.flashinfer.fp4_quantize(x); torch.ops.trtllm.reswizzle_sf(y)\nimport flashinfer_cubin\n"
         "url = 'https://flashinfer.ai/whl'\nfind_spec(\"flashinfer\")\n")
    r = rewrite(t, "flashinfer_k114")
    want = ("import flashinfer_k114 as flashinfer\nimport flashinfer_k114.jit.env; import flashinfer_k114 as flashinfer\n"
            "import flashinfer_k114.utils as fu\n    from flashinfer_k114.jit import x\nfrom flashinfer_k114 import y\n"
            "m = importlib.import_module('flashinfer_k114.sampling')\n"
            "@torch.library.custom_op(\"flashinfer_k114::fp4_quantize\", mutates_args=())\n@torch.library.custom_op(\"trtllm_k114::reswizzle_sf\")\n"
            "torch.ops.flashinfer_k114.fp4_quantize(x); torch.ops.trtllm_k114.reswizzle_sf(y)\nimport flashinfer_cubin\n"
            "url = 'https://flashinfer.ai/whl'\nfind_spec(\"flashinfer_k114\")\n")
    assert r == want, r
    print("vendor_flashinfer selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
        sys.exit(0)
    src, dest, commit = sys.argv[1:4]
    print(f"vendored {src} at {commit[:12]} to {dest}: {vendor(src, dest, commit)} files rewritten")
