"""Print `name==version` for every installed nvidia-cutlass-dsl* package, at the version a b12x pyproject pins.

Used by overlay-b12x-src.Dockerfile. The image carries the [cu13] set, not every package b12x lists, so the installed
set decides which packages move. Self-check: python3 ci/cutlass_pins.py --selftest
"""
import importlib.metadata as md
import re
import sys
import tomllib


def pins(pyproject_text: str, installed: set[str]) -> list[str]:
    want = {}
    for dep in tomllib.loads(pyproject_text)["project"]["dependencies"]:
        m = re.fullmatch(r"(nvidia-cutlass-dsl[a-z0-9-]*)==([0-9.]+)", dep.replace(" ", ""))
        if m:
            want[m[1]] = m[2]
    if len(set(want.values())) != 1:
        raise SystemExit(f"expected one pinned cutlass-dsl version, got {want}")
    ver = next(iter(want.values()))
    return [f"{n}=={want.get(n, ver)}" for n in sorted(installed) if n.startswith("nvidia-cutlass-dsl")]


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        t = '[project]\ndependencies = ["torch>=2", "nvidia-cutlass-dsl==4.7.1", "nvidia-cutlass-dsl-libs-cu12==4.7.1"]\n'
        got = pins(t, {"nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-cu13", "torch"})
        assert got == ["nvidia-cutlass-dsl==4.7.1", "nvidia-cutlass-dsl-libs-cu13==4.7.1"], got
        print("cutlass_pins selftest ok")
        sys.exit(0)
    have = {d.metadata["Name"].lower().replace("_", "-") for d in md.distributions()}
    print(" ".join(pins(open(sys.argv[1]).read(), have)))
