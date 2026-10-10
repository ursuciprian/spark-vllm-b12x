"""Check that every `from b12x... import name` in the installed vLLM resolves against the installed b12x.

Runs inside an image (CPU is enough). Walks the vllm package with ast, so lazy imports inside functions count too.
Prints one line per unresolved import; exit 1 when one sits in a file on the Qwen3.8-Flash-Next serving path.
Usage: python3 b12x_import_check.py [vllm package dir]   Self-check: python3 b12x_import_check.py --selftest
"""
import ast
import importlib
import os
import re
import sys

QWEN_PATH = re.compile(
    r"(fused_moe/b12x|fused_moe/experts/flashinfer_b12x_moe|quantization/utils/b12x_moe|kernels/linear/|"
    r"mamba/gdn/qwen_gdn|mamba/ops/b12x_gdn|warmup/b12x_prepare|v1/worker/b12x_startup|models/qwen3_8_flash_next/|"
    r"vocab_parallel_embedding|logits_processor|b12x_roce_all_reduce|quantization/modelopt|layers/linear\.py|"
    r"utils/b12x|v1/attention/backends/gdn_attn)"
)


def imports(path):
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 and node.module.split(".")[0] == "b12x":
            yield node.lineno, node.module, [a.name for a in node.names if a.name != "*"]
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == "b12x":
                    yield node.lineno, a.name, []


def check(root):
    bad, n = [], 0
    for d, _, fs in os.walk(root):
        for f in fs:
            if not f.endswith(".py"):
                continue
            p = os.path.join(d, f)
            for line, mod, names in imports(p):
                n += 1
                try:
                    m = importlib.import_module(mod)
                except Exception as e:  # noqa: BLE001 - report every failure kind
                    bad.append((p, line, f"import {mod}: {type(e).__name__}: {e}"))
                    continue
                for name in names:
                    if not hasattr(m, name):
                        try:
                            importlib.import_module(f"{mod}.{name}")
                        except Exception:  # noqa: BLE001
                            bad.append((p, line, f"{mod} has no {name}"))
    return n, bad


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            open(os.path.join(t, "a.py"), "w").write("def f():\n    from b12x.x import y, z\nimport b12x.q\nfrom os import path\n")
            got = sorted(imports(os.path.join(t, "a.py")))
        assert got == [(2, "b12x.x", ["y", "z"]), (3, "b12x.q", [])], got
        assert QWEN_PATH.search("vllm/models/qwen3_8_flash_next/model.py") and not QWEN_PATH.search("vllm/models/glm5next/x.py")
        print("b12x_import_check selftest ok")
        sys.exit(0)
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(importlib.import_module("vllm").__file__)
    n, bad = check(root)
    hot = [b for b in bad if QWEN_PATH.search(b[0])]
    for p, line, msg in bad:
        print(f"{'QWEN ' if QWEN_PATH.search(p) else 'other'} {os.path.relpath(p, root)}:{line}: {msg}")
    print(f"{n} b12x imports checked, {len(bad)} unresolved, {len(hot)} on the Qwen path")
    sys.exit(1 if hot else 0)
