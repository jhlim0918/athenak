#!/usr/bin/env python3
"""Recompute the SC14 stress diagnostics directly from paired .bin dumps.

For every (hydro_w, grav_phi) pair on the root grid this forms the stress
PER CELL first,

    w_xy = g_x g_y / (4 pi G)  +  rho v_x dv_y          (SC14 eq. 19 numerator)

with g = -grad(phi) differenced from the companion grav_phi dump, and only then
integrates over the domain -- the ordering the history file cannot express,
since it stores the gravitational and Reynolds integrals separately.  Both
component integrals are kept too, so the two orderings can be compared directly.

Volume integrals written per snapshot (dV = dx1 dx2 dx3, root grid):
    mass rho_cs rho_cs2 rho_prs rho_dv2      (Q, <cs>_rho, rms dv)
    drho_rms drho_rms_rw dv2_vol             SC14 eq.-17 density dispersion
                                             (volume- and density-weighted) and
                                             the volume-averaged <dv^2>
    rho_wgrv rho_wrey rho_wtot               density-weighted, eq. 19
    wgrv wrey wtot                           volume-weighted, eq. 20
    rho_wtot_int wtot_int                    same, x-edge columns dropped
Vertical profiles (nx3 each, horizontal averages):
    rho_z prs_z cs_z w_z mflux_z             mflux_z = <rho v_z>_xy

x-derivatives use centered differences inside and one-sided at the two x1 edge
columns (the shear-periodic wrap carries a y offset), matching
scripts/plot_gt_stress_z.jl.  The *_int integrals drop those two columns from
numerator and denominator alike as a sensitivity check on that choice.

Usage:
    python3 extract_gt_stress.py <run_dir> -o gt_b10_stress.npz -j 8
"""

import argparse
import os
import sys
from glob import glob
from multiprocessing import Pool

import numpy as np


def _load_reader(repo):
    candidates = [repo] if repo else []
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
    raise SystemExit("cannot find vis/python/bin_convert.py -- pass --repo or set $ATHENAK")


_READ = None


def _init_worker(repo):
    global _READ
    _READ = _load_reader(repo)


def _header_time(path):
    """Simulation time from a .bin file's small text header, without reading data."""
    with open(path, "rb") as fp:
        head = fp.read(4096).decode("latin-1")
    for line in head.splitlines():
        if line.strip().startswith("time"):
            return float(line.split("=")[1])
    raise RuntimeError(f"no time= line in the header of {path}")


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
    return 0.125 * (
        a[0::2, 0::2, 0::2] + a[1::2, 0::2, 0::2]
        + a[0::2, 1::2, 0::2] + a[0::2, 0::2, 1::2]
        + a[1::2, 1::2, 0::2] + a[1::2, 0::2, 1::2]
        + a[0::2, 1::2, 1::2] + a[1::2, 1::2, 1::2]
    )


def assemble_root(fd, var):
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


def stress_one(pair):
    fw, fp = pair
    fd = _READ(fw)
    fdp = _READ(fp)
    if abs(fd["time"] - fdp["time"]) > 1e-8:
        raise RuntimeError(f"time mismatch {fw} {fd['time']} vs {fp} {fdp['time']}")

    gamma = float(header_get(fd["header"], "<hydro>", "gamma"))
    fpg = float(header_get(fd["header"], "<gravity>", "four_pi_G"))
    nx1, nx2, nx3 = fd["Nx1"], fd["Nx2"], fd["Nx3"]
    dx = (fd["x1max"] - fd["x1min"]) / nx1
    dy = (fd["x2max"] - fd["x2min"]) / nx2
    dz = (fd["x3max"] - fd["x3min"]) / nx3
    dV = dx * dy * dz

    rho = assemble_root(fd, "dens")
    vx = assemble_root(fd, "velx")
    dvy = assemble_root(fd, "vely")          # FARGO frame: already non-Keplerian
    vz = assemble_root(fd, "velz")
    prs = (gamma - 1.0) * assemble_root(fd, "eint")
    phi = assemble_root(fdp, "grav_phi" if "grav_phi" in fdp["mb_data"] else "phi")

    gx = np.empty_like(phi)
    gx[:, :, 1:-1] = -(phi[:, :, 2:] - phi[:, :, :-2]) / (2 * dx)
    gx[:, :, 0] = -(phi[:, :, 1] - phi[:, :, 0]) / dx
    gx[:, :, -1] = -(phi[:, :, -1] - phi[:, :, -2]) / dx
    gy = -(np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * dy)

    w_grv = gx * gy / fpg
    w_rey = rho * vx * dvy
    w_tot = w_grv + w_rey                     # summed per cell, before integrating

    cs2 = gamma * prs / rho
    cs = np.sqrt(cs2)
    dv2 = vx**2 + dvy**2 + vz**2
    sl = np.s_[:, :, 1:-1]                    # interior in x1

    # SC14 eq. 17: drho = rho - rho_bar(z), quoted relative to the local mean
    zbar = rho.mean(axis=(1, 2))[:, None, None]
    drel2 = ((rho - zbar) / zbar) ** 2

    out = {
        "time": fd["time"],
        "mass": rho.sum() * dV,
        "rho_cs": (rho * cs).sum() * dV,
        "rho_cs2": (rho * cs2).sum() * dV,
        "rho_prs": (rho * prs).sum() * dV,
        "rho_dv2": (rho * dv2).sum() * dV,
        "prs": prs.sum() * dV,
        "rho_wgrv": (rho * w_grv).sum() * dV,
        "rho_wrey": (rho * w_rey).sum() * dV,
        "rho_wtot": (rho * w_tot).sum() * dV,
        "wgrv": w_grv.sum() * dV,
        "wrey": w_rey.sum() * dV,
        "wtot": w_tot.sum() * dV,
        "rho_wtot_int": (rho[sl] * w_tot[sl]).sum() * dV,
        "rho_prs_int": (rho[sl] * prs[sl]).sum() * dV,
        "wtot_int": w_tot[sl].sum() * dV,
        "rho_cs2_int": (rho[sl] * cs2[sl]).sum() * dV,
        "drho_rms": np.sqrt(drel2.mean()),
        "drho_rms_rw": np.sqrt((rho * drel2).sum() / rho.sum()),
        "dv2_vol": dv2.mean(),
    }
    prof = {
        "rho_z": rho.mean(axis=(1, 2)),
        "prs_z": prs.mean(axis=(1, 2)),
        "cs_z": (rho * cs).mean(axis=(1, 2)) / rho.mean(axis=(1, 2)),
        "w_z": w_tot.mean(axis=(1, 2)),
        "mflux_z": (rho * vz).mean(axis=(1, 2)),
    }
    meta = {k: fd[k] for k in ("Nx1", "Nx2", "Nx3", "x1min", "x1max",
                               "x2min", "x2max", "x3min", "x3max")}
    meta["gamma"] = gamma
    meta["four_pi_G"] = fpg
    return out, prof, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--out")
    ap.add_argument("-j", "--nproc", type=int, default=4)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--repo")
    args = ap.parse_args()
    _load_reader(args.repo)

    root = args.run_dir.rstrip("/")
    bindir = root if os.path.basename(root) == "bin" else os.path.join(root, "bin")
    hyd = sorted(glob(os.path.join(bindir, "*.hydro_w.*.bin")))[::args.stride]
    # pair by TIME, not by index: the two outputs may run at different cadences
    # (e.g. hydro_w every 0.25/Omega for a movie, grav_phi every 2/Omega), in which
    # case equal file numbers are different instants.  Times come from the header.
    phi_by_time = {}
    for pth in glob(os.path.join(bindir, "*.grav_phi.*.bin")):
        phi_by_time[round(_header_time(pth), 6)] = pth
    pairs = []
    for f in hyd:
        key = round(_header_time(f), 6)
        if key in phi_by_time:
            pairs.append((f, phi_by_time[key]))
        else:
            print(f"  skipping {os.path.basename(f)} (t={key}): no grav_phi at that time")
    if not pairs:
        raise SystemExit(f"no hydro_w/grav_phi pairs in {bindir}")
    out = args.out or os.path.join(os.path.dirname(bindir) or ".", "gt_stress.npz")
    print(f"{len(pairs)} pairs in {bindir}  ->  {out}   ({args.nproc} workers)")

    res = []
    with Pool(args.nproc, initializer=_init_worker, initargs=(args.repo,)) as pool:
        for n, r in enumerate(pool.imap(stress_one, pairs, chunksize=1), 1):
            res.append(r)
            if n % 5 == 0 or n == len(pairs):
                print(f"  {n}/{len(pairs)}  t = {r[0]['time']:.2f}", flush=True)
    res.sort(key=lambda r: r[0]["time"])

    scal = {k: np.array([r[0][k] for r in res]) for k in res[0][0]}
    prof = {k: np.stack([r[1][k] for r in res]) for k in res[0][1]}
    meta = res[0][2]
    e = np.linspace(meta["x3min"], meta["x3max"], meta["Nx3"] + 1)
    np.savez_compressed(out, x3=0.5 * (e[:-1] + e[1:]),
                        gamma=meta["gamma"], four_pi_G=meta["four_pi_G"],
                        Lx=meta["x1max"] - meta["x1min"],
                        Ly=meta["x2max"] - meta["x2min"],
                        **scal, **prof)
    print(f"wrote {out}  ({len(res)} snapshots, "
          f"t = {scal['time'][0]:.1f} .. {scal['time'][-1]:.1f})")


if __name__ == "__main__":
    main()
