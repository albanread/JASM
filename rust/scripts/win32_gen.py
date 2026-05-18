#!/usr/bin/env python3
"""
Generate JASM `.masm` binding files for the Win32 API — v2 (invoke).

v2 emits, for each function in `windows_api.db`:

  1. `@extern "DLL.dll" NAME(arg_count)` — declaration (same as v1).
  2. `@macro NAME(arg0, arg1, ...)` — an `invoke`-style wrapper that
     places each argument in the correct Win64 register or stack slot,
     reserves shadow space, aligns RSP, calls the function, restores
     RSP.

Functions with by-value struct params, variadics, or void params are
emitted as `@extern` only — no wrapper. The user can still call them
by hand if needed.

Usage:

    python scripts/win32_gen.py                 # default DLL set
    python scripts/win32_gen.py --all           # every DLL in the DB
    python scripts/win32_gen.py --dll USER32.dll GDI32.dll
    python scripts/win32_gen.py --db E:/windows_api/windows_api.db

Source of truth: ``E:\\windows_api\\windows_api.db`` — a SQLite database
NewM2 built from Microsoft's ``Windows.Win32.winmd`` metadata.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

DEFAULT_DLLS = [
    # Classic "fat" DLLs — what every Win32 program has linked against
    # since the 1990s. Most exports live here.
    "KERNEL32.dll",
    "USER32.dll",
    "GDI32.dll",
    "ADVAPI32.dll",
    "SHLWAPI.dll",
    "COMCTL32.dll",
    "COMDLG32.dll",
    "OLE32.dll",
    "SHELL32.dll",
    "WINMM.dll",
    "WS2_32.dll",
    "NTDLL.dll",
    # API-set forwarders for post-Win10 APIs that never got back-
    # forwarded into kernel32.dll. Add new api-ms-* names here as
    # consumers need them — each one is small (1-30 functions) and
    # ABI-stable; Windows's loader maps them to whatever physical
    # DLL actually implements the function. Targeting Win10 1803+.
    "api-ms-win-core-memory-l1-1-6.dll",   # VirtualAlloc2, VirtualAlloc2FromApp
]

# ── Win64 ABI helpers ────────────────────────────────────────────────

INT_REGS_BY_SIZE = {
    1: ("cl", "dl", "r8b", "r9b"),
    2: ("cx", "dx", "r8w", "r9w"),
    4: ("ecx", "edx", "r8d", "r9d"),
    8: ("rcx", "rdx", "r8", "r9"),
}
FLOAT_REGS = ("xmm0", "xmm1", "xmm2", "xmm3")


def classify_param(type_name, kind, indirection_level):
    """Return ('int' | 'float' | 'skip', size_in_bytes).

    Win64 places ints in GPRs, floats in xmm0–3. Position determines
    the slot (mixed int/float don't share slots — slot 0 is rcx-OR-xmm0,
    not both).
    """
    if indirection_level and indirection_level >= 1:
        return ("int", 8)
    if kind in ("pointer", "reference", "delegate"):
        return ("int", 8)
    if kind == "struct":
        return ("skip", 0)
    if kind == "enum":
        return ("int", 4)
    if kind == "primitive":
        if not type_name:
            return ("int", 8)
        n = type_name.lower()
        if n == "void":
            return ("skip", 0)
        if n in ("u8", "i8", "char"):
            return ("int", 1)
        if n == "bool":
            # Win32 BOOL is 32-bit, not 8-bit. Deliberate.
            return ("int", 4)
        if n in ("u16", "i16"):
            return ("int", 2)
        if n in ("u32", "i32"):
            return ("int", 4)
        if n in ("u64", "i64", "usize", "isize"):
            return ("int", 8)
        if n == "f32":
            return ("float", 4)
        if n == "f64":
            return ("float", 8)
        # Unknown primitive — be conservative (64-bit).
        return ("int", 8)
    # Unknown kind — skip the wrapper for this function.
    return ("skip", 0)


def stack_reservation(num_args):
    """Bytes to `sub rsp` by. 32 shadow + 8 per stack arg, rounded up
    to a multiple of 16 so RSP is 16-aligned before the CALL."""
    base = 32
    if num_args > 4:
        base += 8 * (num_args - 4)
    if base % 16:
        base += 8
    return base


# ── DB queries ───────────────────────────────────────────────────────


def list_functions_for_dll(conn, dll):
    """Walk every function in `dll`. For each, return:
       (function_name, [(param_kind, param_type_name, indirection_level), ...])
    Filtered: drops A-family duplicates of W pairs; drops variadics."""
    fns = conn.execute(
        """
        SELECT f.function_id, f.function_name, f.aw_family
          FROM functions f
         WHERE f.dll_name = ?
           AND f.is_variadic = 0
         ORDER BY f.function_name
        """,
        (dll,),
    ).fetchall()

    w_names = {row[1] for row in fns if row[2] == "W"}
    out = []
    for fid, name, aw in fns:
        if aw == "A":
            w_form = name[:-1] + "W" if name.endswith("A") else None
            if w_form in w_names:
                continue
        params = conn.execute(
            """
            SELECT p.ordinal, t.kind, t.type_name, COALESCE(t.indirection_level, 0)
              FROM function_params p
              JOIN types t ON t.type_id = p.type_id
             WHERE p.function_id = ?
             ORDER BY p.ordinal
            """,
            (fid,),
        ).fetchall()
        param_types = [(k, tn, il) for _ord, k, tn, il in params]
        out.append((name, param_types))
    return out


# ── Code emission ────────────────────────────────────────────────────


HEADER = """\
; ---------------------------------------------------------------------
; Generated from windows_api.db -- DLL: {dll}
; v2: @extern declarations + invoke-style wrapper macros.
; DO NOT EDIT BY HAND. Regenerate via:
;     python scripts/win32_gen.py --dll {dll}
;
; Each function gets two definitions:
;
;   @extern "{dll}" NAME(N)
;     The declaration the host pairs with LoadLibraryW + GetProcAddress.
;
;   @macro NAME(arg0, arg1, ...)
;     Win64-ABI-aware wrapper. Each arg goes in its register or stack
;     slot per the parameter's type; shadow space and 16-byte RSP
;     alignment are handled. Functions whose signatures cannot be
;     modelled (struct-by-value params, variadics) get only the
;     @extern -- call those by hand.
;
; Function count: {wrapped_count} wrapped, {extern_only_count} extern-only
; Source of truth: E:\\windows_api\\windows_api.db
; ---------------------------------------------------------------------

"""


def emit_wrapper(name, param_types):
    """Return the `@macro NAME(...)` body as a string, or None if the
    function can't be wrapped (struct-by-value param, etc.)."""
    n = len(param_types)
    slots = []
    for (kind, type_name, il) in param_types:
        cls, size = classify_param(type_name or "", kind or "", il or 0)
        if cls == "skip":
            return None
        slots.append((cls, size))

    if n == 0:
        macro_header = f"@macro {name}()"
    else:
        args = ", ".join(f"arg{i}" for i in range(n))
        macro_header = f"@macro {name}({args})"

    lines = [macro_header]

    # Order:
    #   1. save rsp into r12 (callee-saved on Win64)
    #   2. align rsp to 16
    #   3. sub rsp, N  (reserve shadow + stack args)
    #   4. place stack args (writes to [rsp+32], [rsp+40], ...)
    #   5. place register args (writes to rcx/rdx/r8/r9/xmm0..3)
    #   6. call
    #   7. restore rsp from r12
    #
    # The user-supplied &argN substitutions refer to their original
    # locations (memory operand, immediate, etc.); they're computed
    # before rsp moves, so steps 4 & 5 read stable values.

    reserve = stack_reservation(n)
    lines.append("    mov     r12, rsp")
    lines.append("    and     rsp, -16")
    lines.append(f"    sub     rsp, {reserve}")

    # Stack args first (positions 4+). All stack args are 8 bytes wide
    # on Win64 regardless of the declared type.
    for i in range(4, n):
        offset = 32 + 8 * (i - 4)
        lines.append(f"    mov     qword ptr [rsp + {offset}], &arg{i}")

    # Register args.
    for i in range(min(n, 4)):
        cls, size = slots[i]
        if cls == "int":
            reg = INT_REGS_BY_SIZE[size][i]
            lines.append(f"    mov     {reg}, &arg{i}")
        else:
            reg = FLOAT_REGS[i]
            mnem = "movss" if size == 4 else "movsd"
            lines.append(f"    {mnem}   {reg}, &arg{i}")

    lines.append(f"    call    {name}")
    lines.append("    mov     rsp, r12")
    lines.append("@endmacro")
    return "\n".join(lines)


def emit_dll(conn, dll, out_dir):
    """Write the binding file for `dll`. Returns (wrapped, extern_only)."""
    funcs = list_functions_for_dll(conn, dll)
    if not funcs:
        print(f"warning: no functions found for {dll}", file=sys.stderr)
        return (0, 0)

    out_path = out_dir / filename_for(dll)
    wrapped = 0
    extern_only = 0
    blocks = []
    for name, param_types in funcs:
        n = len(param_types)
        ext_line = f'@extern "{dll}" {name}({n})'
        wrapper = emit_wrapper(name, param_types)
        if wrapper is None:
            extern_only += 1
            blocks.append(ext_line + "  ; (no wrapper -- unsupported signature)")
        else:
            wrapped += 1
            blocks.append(ext_line + "\n" + wrapper)

    with out_path.open("w", encoding="utf-8") as f:
        f.write(
            HEADER.format(
                dll=dll, wrapped_count=wrapped, extern_only_count=extern_only
            )
        )
        f.write("\n\n".join(blocks))
        f.write("\n")
    print(f"wrote {out_path}  ({wrapped} wrapped, {extern_only} extern-only)")
    return (wrapped, extern_only)


def filename_for(dll):
    stem = dll.lower()
    if stem.endswith(".dll"):
        stem = stem[:-4]
    return f"{stem}.masm"


# ── Driver ───────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--db",
        default=r"E:\windows_api\windows_api.db",
        help="path to the windows_api.db SQLite (default: %(default)s)",
    )
    p.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "win32"),
        help="output directory (default: <repo>/win32)",
    )
    p.add_argument(
        "--dll",
        nargs="+",
        metavar="DLL.dll",
        help="generate only these DLLs (default: the practical core)",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="generate every DLL in the DB",
    )
    args = p.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        sys.exit(f"error: DB not found at {db_path}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)

    if args.all:
        dlls = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT dll_name FROM functions "
                "WHERE dll_name IS NOT NULL ORDER BY dll_name"
            )
        ]
    elif args.dll:
        dlls = args.dll
    else:
        dlls = DEFAULT_DLLS

    total_wrapped = 0
    total_extern_only = 0
    for dll in dlls:
        w, e = emit_dll(conn, dll, out_dir)
        total_wrapped += w
        total_extern_only += e
    print(
        f"\n{len(dlls)} DLLs: {total_wrapped} wrapped, "
        f"{total_extern_only} extern-only -> {out_dir}"
    )


if __name__ == "__main__":
    main()
