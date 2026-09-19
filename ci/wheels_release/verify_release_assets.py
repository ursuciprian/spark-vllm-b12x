#!/usr/bin/env python3
"""ci/wheels_release/verify_release_assets.py

Contract checker for a spark-vllm-b12x wheels-release bundle, ported from the
local-inference-lab jovian/lil wheel-release verify_release_assets.py pattern
(same idea: exact asset-set match + sha256 verification, raise loudly on any
mismatch, silent zero-exit on success).

Bundle contract: for each component in {flashinfer, vllm, b12x} there must be
exactly one "<component>-cu134-<sha7>.tar.zst" and its "<...>.sha256" sidecar,
plus one "build-metadata.yaml". No extra files, no symlinks.

Usage:
  verify_release_assets.py --directory DIR [--reference-directory DIR2]

With --reference-directory, additionally requires DIR and DIR2 to contain the
byte-identical asset set (used to confirm an existing release tag's assets
match what we just rebuilt, before deciding to skip re-publishing).
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

COMPONENTS = ("flashinfer", "vllm", "b12x")
ARCHIVE_RE = re.compile(r"^(flashinfer|vllm|b12x)-cu134-[0-9a-f]{7}\.tar\.zst$")


def require(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sidecar_digest(sidecar: Path) -> str:
    # GNU sha256sum one-line format: "<hex>  <filename>"
    line = sidecar.read_text().strip().splitlines()[0]
    digest = line.split()[0]
    require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None, f"malformed sha256 sidecar: {sidecar}")
    return digest


def check_directory(directory: Path) -> dict:
    require(directory.is_dir(), f"not a directory: {directory}")
    entries = sorted(p.name for p in directory.iterdir())
    require(len(entries) == len(set(entries)), f"duplicate asset names in {directory}")

    found = {}
    for name in entries:
        p = directory / name
        require(p.is_file() and not p.is_symlink(), f"asset must be a regular file, not a symlink: {p}")
        m = ARCHIVE_RE.match(name)
        if m:
            found[m.group(1)] = name

    for component in COMPONENTS:
        require(component in found, f"missing archive for component '{component}' in {directory}")
        archive = found[component]
        sidecar = directory / f"{archive}.sha256"
        require(sidecar.exists(), f"missing sha256 sidecar for {archive}")
        expected = read_sidecar_digest(sidecar)
        actual = sha256_of(directory / archive)
        require(actual == expected, f"sha256 mismatch for {archive}: sidecar={expected} actual={actual}")

    require((directory / "build-metadata.yaml").is_file(), f"missing build-metadata.yaml in {directory}")
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--directory", required=True, type=Path)
    ap.add_argument("--reference-directory", type=Path, default=None)
    args = ap.parse_args()

    try:
        found = check_directory(args.directory)
        if args.reference_directory is not None:
            ref_found = check_directory(args.reference_directory)
            require(found == ref_found, "asset name sets differ between fresh build and existing release")
            for component, archive in found.items():
                for name in (archive, f"{archive}.sha256"):
                    a = sha256_of(args.directory / name)
                    b = sha256_of(args.reference_directory / name)
                    require(a == b, f"independent reference mismatch: {name} (fresh={a} released={b})")
    except ValueError as e:
        print(f"verify_release_assets: FAIL: {e}", file=sys.stderr)
        return 1

    print("verify_release_assets: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
