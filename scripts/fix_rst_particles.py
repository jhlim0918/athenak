#!/usr/bin/env python3
"""Comment out a stray <particles> block in an AthenaK restart file's text header.

A gas-only run built from the dust branch writes an empty <particles> block into every
restart it produces (pgen.cpp calls GetOrAddBoolean("particles","restart_insert"), and
GetOrAdd CREATES the block when it is absent; ParameterDump then serialises it).  On the
next restart MeshBlockPack sees the block, constructs the particles module, and dies in
Particles::Particles with "Parameter name 'particle_type' not found in block 'particles'".

This rewrites the block header "<particles>" as "#particles>" -- exactly the same number
of bytes, and ParameterInput skips any line whose first non-blank character is '#'
(parameter_input.cpp LoadFromStream).  Every byte offset in the file is preserved, so the
binary payload after <par_end> is untouched.  The orphaned "restart_insert = 0" line then
attaches to the preceding block and is reported as an unused parameter.

Usage:  python3 fix_rst_particles.py <file.rst> [more.rst ...]      (edits IN PLACE)
        python3 fix_rst_particles.py --dry-run <file.rst>
"""
import os, sys

OLD, NEW = b"<particles>", b"#particles>"
assert len(OLD) == len(NEW)

def fix(path, dry=False):
    size = os.path.getsize(path)
    with open(path, "r+b" if not dry else "rb") as f:
        head = f.read(1 << 22)                      # header is far inside the first 4 MB
        end = head.find(b"<par_end>")
        if end < 0:
            print(f"  {path}: no <par_end> in the first 4 MB -- not an AthenaK restart?")
            return False
        off = head.find(OLD, 0, end)
        if off < 0:
            print(f"  {path}: no <particles> block in the header (nothing to do)")
            return False
        if head.find(OLD, off + 1, end) >= 0:
            print(f"  {path}: more than one <particles> line -- refusing to guess")
            return False
        line_end = head.find(b"\n", off)
        body = head[off:line_end].decode("latin-1").strip()
        if body != "<particles>":
            print(f"  {path}: header line is {body!r}, not a bare block tag -- refusing")
            return False
        if dry:
            print(f"  {path}: would patch offset {off} (header ends at {end})")
            return True
        f.seek(off); f.write(NEW); f.flush(); os.fsync(f.fileno())
    assert os.path.getsize(path) == size, "file size changed -- this must never happen"
    print(f"  {path}: patched at offset {off}, size unchanged ({size} bytes)")
    return True

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    dry = "--dry-run" in sys.argv
    if not args:
        sys.exit(__doc__)
    n = sum(fix(a, dry) for a in args)
    print(f"{n}/{len(args)} file(s) {'would be ' if dry else ''}patched")
