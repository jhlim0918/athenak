#!/usr/bin/env python3
"""Reduce a directory of AthenaK .bin dumps to the two 2D maps a gravito-turbulence
movie needs, so the 60 MB-per-dump volumes never leave the cluster.

Per snapshot it writes
    sigma  (x, y)  =  int rho dz            vertically integrated surface density
    rho_xz (x, z)  =  <rho>_y               azimuthally averaged density

matching scripts/plot_bin_colden.jl and scripts/plot_bin_rhoxz.jl (SC14 Fig.-6
left panel) exactly, so numbers are comparable with the existing single-frame
figures.  SMR/AMR dumps are restricted onto the root grid first.

Output is a single .npz: for the 301-dump beta=10 run that is ~95 MB instead of
~22 GB, small enough to scp home and drive a movie from.

Usage (on the cluster, from anywhere):
    module load python3
    python3 extract_gt_slices.py <run_dir_or_bin_dir> -o gt_b10_slices.npz -j 16

  <run_dir_or_bin_dir>  a run directory (its bin/ subdirectory is used) or the
                        bin/ directory itself
  -o/--out              output .npz            (default <run_dir>/gt_slices.npz)
  -j/--nproc            worker processes       (default 8; the job is I/O bound)
  -v/--var              variable to reduce     (default dens)
  --stride N            keep every Nth dump    (default 1)
  --pattern GLOB        dump name filter       (default *.hydro_w.*.bin)
  --repo PATH           AthenaK checkout holding vis/python/bin_convert.py
                        (default: $ATHENAK, else inferred from this script)

Read back with
    d = np.load("gt_b10_slices.npz")
    d["t"], d["x1"], d["x2"], d["x3"], d["sigma"], d["rho_xz"]
where sigma has shape (nt, nx1, nx2) and rho_xz has shape (nt, nx1, nx3).
"""

import argparse
import os
import sys
from glob import glob
from multiprocessing import Pool

import numpy as np


def _load_reader(repo):
    """Import read_binary from the AthenaK checkout (one reader, one format)."""
    candidates = []
    if repo:
        candidates.append(repo)
    if os.environ.get("ATHENAK"):
        candidates.append(os.environ["ATHENAK"])
    here = os.path.dirname(os.path.abspath(__file__))
    candidates += [os.path.dirname(here), here, os.getcwd()]
    for root in candidates:
        path = os.path.join(root, "vis", "python")
        if os.path.isfile(os.path.join(path, "bin_convert.py")):
            sys.path.insert(0, path)
            from bin_convert import read_binary
            return read_binary
    raise SystemExit(
        "cannot find vis/python/bin_convert.py -- pass --repo /path/to/athenak "
        "or set $ATHENAK"
    )


# per-worker state: set by _init_worker so this works under both the fork start
# method (Linux) and spawn (macOS), where module globals are not inherited
_READ = None
_VAR = "dens"


def _init_worker(repo, var):
    global _READ, _VAR
    _READ = _load_reader(repo)
    _VAR = var


def header_get(header, block, key):
    cur = "<none>"
    for line in header:
        if line.startswith("<"):
            cur = line
            continue
        k, _, v = line.partition("=")
        if cur == block and k.strip() == key:
            return v.strip()
    return None


def restrict2(a):
    """Conservative 2x coarsening of a (nx3, nx2, nx1) block."""
    return 0.125 * (
        a[0::2, 0::2, 0::2] + a[1::2, 0::2, 0::2]
        + a[0::2, 1::2, 0::2] + a[0::2, 0::2, 1::2]
        + a[1::2, 1::2, 0::2] + a[1::2, 0::2, 1::2]
        + a[0::2, 1::2, 1::2] + a[1::2, 1::2, 1::2]
    )


def assemble_root(fd, var):
    """Full root-level grid (nx3, nx2, nx1); finer blocks restricted onto it."""
    out = np.full((fd["Nx3"], fd["Nx2"], fd["Nx1"]), np.nan)
    m1, m2, m3 = fd["nx1_out_mb"], fd["nx2_out_mb"], fd["nx3_out_mb"]
    for m in range(fd["n_mbs"]):
        lx1, lx2, lx3, lev = fd["mb_logical"][m]
        blk = np.asarray(fd["mb_data"][var][m], dtype=np.float64)
        for _ in range(lev):
            blk = restrict2(blk)
        s = 1 << lev
        i0, j0, k0 = lx1 * m1 // s, lx2 * m2 // s, lx3 * m3 // s
        out[k0:k0 + m3 // s, j0:j0 + m2 // s, i0:i0 + m1 // s] = blk
    if np.isnan(out).any():
        raise RuntimeError("gaps in the assembled root grid")
    return out


def reduce_one(fname):
    """-> (time, sigma (nx1,nx2), rho_xz (nx1,nx3), meta) for one dump."""
    fd = _READ(fname)
    rho = assemble_root(fd, _VAR)
    dz = (fd["x3max"] - fd["x3min"]) / fd["Nx3"]
    sigma = rho.sum(axis=0).T * dz            # (nx2,nx1) -> (nx1,nx2)
    rho_xz = rho.mean(axis=1).T               # (nx3,nx1) -> (nx1,nx3)
    meta = {k: fd[k] for k in
            ("Nx1", "Nx2", "Nx3", "x1min", "x1max", "x2min", "x2max",
             "x3min", "x3max")}
    meta["omega0"] = header_get(fd["header"], "<shearing_box>", "omega0")
    return (fd["time"], sigma.astype(np.float32), rho_xz.astype(np.float32), meta)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--out")
    ap.add_argument("-j", "--nproc", type=int, default=8)
    ap.add_argument("-v", "--var", default="dens")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--pattern", default="*.hydro_w.*.bin")
    ap.add_argument("--repo")
    args = ap.parse_args()

    _load_reader(args.repo)   # fail fast in the parent if the reader is missing

    root = args.run_dir.rstrip("/")
    bindir = root if os.path.basename(root) == "bin" else os.path.join(root, "bin")
    files = sorted(glob(os.path.join(bindir, args.pattern)))[::args.stride]
    if not files:
        raise SystemExit(f"no files matching {args.pattern} in {bindir}")
    out = args.out or os.path.join(os.path.dirname(bindir) or ".", "gt_slices.npz")
    print(f"{len(files)} dumps in {bindir}  ->  {out}   ({args.nproc} workers)")

    results = []
    with Pool(args.nproc, initializer=_init_worker,
              initargs=(args.repo, args.var)) as pool:
        for n, r in enumerate(pool.imap(reduce_one, files, chunksize=1), 1):
            results.append(r)
            if n % 25 == 0 or n == len(files):
                print(f"  {n}/{len(files)}  t = {r[0]:.2f}", flush=True)

    results.sort(key=lambda r: r[0])
    meta = results[0][3]
    t = np.array([r[0] for r in results])
    if np.any(np.diff(t) <= 0):
        dup = np.flatnonzero(np.diff(t) <= 0)
        print(f"  warning: {dup.size} non-increasing time step(s) "
              f"(restart overlap?) near t = {t[dup]}")

    def centers(lo, hi, n):
        e = np.linspace(lo, hi, n + 1)
        return 0.5 * (e[:-1] + e[1:])

    np.savez_compressed(
        out,
        t=t,
        sigma=np.stack([r[1] for r in results]),
        rho_xz=np.stack([r[2] for r in results]),
        x1=centers(meta["x1min"], meta["x1max"], meta["Nx1"]),
        x2=centers(meta["x2min"], meta["x2max"], meta["Nx2"]),
        x3=centers(meta["x3min"], meta["x3max"], meta["Nx3"]),
        omega0=float(meta["omega0"]) if meta["omega0"] else 1.0,
        var=args.var,
        source=os.path.abspath(bindir),
    )
    mb = os.path.getsize(out) / 1024**2
    print(f"wrote {out}  ({mb:.1f} MB, {len(t)} frames, "
          f"Ωt = {t[0]:.1f} .. {t[-1]:.1f})")


if __name__ == "__main__":
    main()
