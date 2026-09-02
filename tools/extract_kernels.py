#!/usr/bin/env python3
"""Extract embedded CUDA/C++ sources from hetero-serve's paged_attn* modules
into standalone .cu / .h files.

The upstream kernels live as raw-string constants (_CUDA_SRC, _CPP_DECL) that
torch.utils.cpp_extension.load_inline compiles at runtime. For the SM121 port we
need them as real translation units: nvcc can then be driven directly, with
explicit -arch flags, and Nsight Compute can attribute source lines.

Parsing is done with ast, not regex, so the extracted text is byte-identical to
what load_inline would have received.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

# module stem -> (output basename, exported function names)
MODULES = {
    "paged_attn":         "paged_attn_v1",
    "paged_attn_v2":      "paged_attn_v2",
    "paged_attn_v3":      "paged_attn_v3",
    "paged_attn_prefill": "paged_attn_prefill",
    "paged_attn_wmma":    "paged_attn_wmma",
}

WANTED = ("_CUDA_SRC", "_CPP_DECL")


def string_constants(path: Path) -> dict[str, str]:
    """Return {name: value} for module-level assignments of plain string literals."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and tgt.id in WANTED:
                out[tgt.id] = node.value.value
    return out


def load_inline_call(path: Path) -> dict[str, object]:
    """Recover the load_inline(...) kwargs so we know the extension name and
    exported functions without having to import torch."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if name != "load_inline":
            continue
        kw: dict[str, object] = {}
        for k in node.keywords:
            try:
                kw[k.arg] = ast.literal_eval(k.value)
            except (ValueError, SyntaxError):
                kw[k.arg] = "<dynamic>"
        return kw
    return {}


HEADER = """// ---------------------------------------------------------------------------
// Extracted verbatim from hetero-serve: {src}  (constant: {const})
// by tools/extract_kernels.py -- do not hand-edit; edit the .cu and re-sync.
//
// Original build path: torch.utils.cpp_extension.load_inline(
//     name={ext_name!r}, functions={functions!r},
//     extra_cuda_cflags={flags!r})
// Note: no -arch/-gencode was passed upstream; load_inline inferred the target
// from the live device or TORCH_CUDA_ARCH_LIST. The SM121 port makes it explicit.
// ---------------------------------------------------------------------------

"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path,
                    default=Path("hetero-serve/heteroserve/model"),
                    help="directory holding the paged_attn*.py modules")
    ap.add_argument("--out", type=Path, default=Path("kernels"))
    args = ap.parse_args()

    if not args.src.is_dir():
        print(f"error: source dir not found: {args.src}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    total = 0
    for stem, base in MODULES.items():
        path = args.src / f"{stem}.py"
        if not path.exists():
            print(f"  ! missing {path}", file=sys.stderr)
            continue
        consts = string_constants(path)
        meta = load_inline_call(path)
        ext_name = meta.get("name", "?")
        functions = meta.get("functions", [])
        flags = meta.get("extra_cuda_cflags", [])

        for const, suffix in (("_CUDA_SRC", ".cu"), ("_CPP_DECL", ".h")):
            if const not in consts:
                print(f"  ! {stem}: no {const}", file=sys.stderr)
                continue
            dst = args.out / f"{base}{suffix}"
            hdr = HEADER.format(src=f"heteroserve/model/{stem}.py", const=const,
                                ext_name=ext_name, functions=functions, flags=flags)
            body = consts[const]
            dst.write_text(hdr + body.lstrip("\n"), encoding="utf-8", newline="\n")
            lines = body.count("\n")
            print(f"  {dst}  ({lines} lines, from {const})")
            total += 1

    print(f"\nextracted {total} files into {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
