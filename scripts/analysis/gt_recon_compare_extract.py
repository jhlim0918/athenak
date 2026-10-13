#!/usr/bin/env python3
"""Per-dump structure diagnostics of an SC14 gravito-turbulence run, for comparing runs
that differ only in numerics (e.g. plm vs ppm4 vs wenoz reconstruction).

Needs only the hydro_w dumps; a grav_phi dump at the same time (paired by header time)
adds the gravitational-stress profile for that snapshot.  For every hydro_w dump:

  scalars   time, mass, rho_max, drho_rms (SC14 eq. 17, volume-weighted), sigma_rms
            (rms/mean of Sigma), floor_frac (cells within 0.1% of the 1e-4 floor)
  Sigma     Sigma(x, y) map (float32)
  spectra   P_y(ky): |FFT_y(Sigma - <Sigma>)|^2 averaged over x (y is periodic);
            P_x(kx): the same along x with a Hann window (x is shear-periodic, so a
            plain FFT would see the seam); KE_y(ky): |FFT_y(sqrt(rho) v)|^2 summed over
            the three components, averaged over x and the |z| < H layer
  profiles  horizontal averages per z: rho, rho-weighted cs^2, Reynolds stress
            rho vx dvy, rho-weighted dv^2, mass flux rho vz, floor fraction;
            w_grv = gx gy / 4 pi G where a grav_phi pair exists (NaN otherwise)

Usage:  python3 gt_recon_compare_extract.py <run_dir> -o out.npz [-j 8]
(run_dir or run_dir/bin holds the *.hydro_w.*.bin / *.grav_phi.*.bin files)
"""

import argparse
import os
import sys
from glob import glob
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import extract_gt_stress as gts  # noqa: E402  (reader + root-grid assembly helpers)

DFLOOR = 1.0e-4


def one(pair):
    fw, fp = pair
    if gts._READ is None:
        gts._init_worker(None)
    fd = gts._READ(fw)
    gamma = float(gts.header_get(fd["header"], "<hydro>", "gamma"))
    nx1, nx2, nx3 = fd["Nx1"], fd["Nx2"], fd["Nx3"]
    dx = (fd["x1max"] - fd["x1min"]) / nx1
    dy = (fd["x2max"] - fd["x2min"]) / nx2
    dz = (fd["x3max"] - fd["x3min"]) / nx3
    zc = fd["x3min"] + (np.arange(nx3) + 0.5) * dz

    rho = gts.assemble_root(fd, "dens")
    vx = gts.assemble_root(fd, "velx")
    dvy = gts.assemble_root(fd, "vely")       # FARGO frame: already non-Keplerian
    vz = gts.assemble_root(fd, "velz")
    prs = (gamma - 1.0) * gts.assemble_root(fd, "eint")
    cs2 = gamma * prs / rho
    dv2 = vx**2 + dvy**2 + vz**2
    floor = rho <= DFLOOR * 1.001

    sig = rho.sum(axis=0) * dz                 # (ny, nx)
    dsig = sig - sig.mean()
    py = (np.abs(np.fft.rfft(dsig, axis=0)) ** 2).mean(axis=1) / nx2**2
    win = np.hanning(nx1)
    px = (np.abs(np.fft.rfft(dsig * win[None, :], axis=1)) ** 2).mean(axis=0) / (win**2).sum() / nx1
    lay = np.abs(zc) < 1.0
    sr = np.sqrt(rho[lay])
    ke = 0.0
    for v in (vx, dvy, vz):
        ke = ke + (np.abs(np.fft.rfft(sr * v[lay], axis=1)) ** 2).mean(axis=(0, 2)) / nx2**2

    zbar = rho.mean(axis=(1, 2))[:, None, None]
    out = dict(
        time=fd["time"], mass=rho.sum() * dx * dy * dz, rho_max=rho.max(),
        drho_rms=np.sqrt((((rho - zbar) / zbar) ** 2).mean()),
        sigma_rms=dsig.std() / sig.mean(), floor_frac=floor.mean(),
        sigma=sig.astype(np.float32), py=py, px=px, ke_y=ke,
        rho_z=rho.mean(axis=(1, 2)),
        cs2_z=(rho * cs2).sum(axis=(1, 2)) / rho.sum(axis=(1, 2)),
        wrey_z=(rho * vx * dvy).mean(axis=(1, 2)),
        dv2_z=(rho * dv2).sum(axis=(1, 2)) / rho.sum(axis=(1, 2)),
        mflux_z=(rho * vz).mean(axis=(1, 2)),
        floor_z=floor.mean(axis=(1, 2)),
        wgrv_z=np.full(nx3, np.nan),
    )
    if fp is not None:
        fdp = gts._READ(fp)
        fpg = float(gts.header_get(fd["header"], "<gravity>", "four_pi_G"))
        phi = gts.assemble_root(fdp, "grav_phi" if "grav_phi" in fdp["mb_data"] else "phi")
        gx = np.empty_like(phi)
        gx[:, :, 1:-1] = -(phi[:, :, 2:] - phi[:, :, :-2]) / (2 * dx)
        gx[:, :, 0] = -(phi[:, :, 1] - phi[:, :, 0]) / dx
        gx[:, :, -1] = -(phi[:, :, -1] - phi[:, :, -2]) / dx
        gy = -(np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * dy)
        out["wgrv_z"] = (gx * gy / fpg).mean(axis=(1, 2))
    meta = dict(Nx1=nx1, Nx2=nx2, Nx3=nx3, x1min=fd["x1min"], x1max=fd["x1max"],
                x2min=fd["x2min"], x2max=fd["x2max"], zc=zc,
                ky=2 * np.pi * np.fft.rfftfreq(nx2, dy), kx=2 * np.pi * np.fft.rfftfreq(nx1, dx))
    print(f"t={fd['time']:7.2f}  rho_max={out['rho_max']:8.2f}  grav={'yes' if fp else 'no'}",
          flush=True)
    return out, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("-j", "--nproc", type=int, default=4)
    args = ap.parse_args()
    gts._init_worker(None)

    files = []
    for sub in ("", "bin"):
        files += glob(os.path.join(args.run_dir, sub, "*.hydro_w.*.bin"))
    phis = {}
    for sub in ("", "bin"):
        for f in glob(os.path.join(args.run_dir, sub, "*.grav_phi.*.bin")):
            phis[round(gts._header_time(f), 6)] = f
    pairs = sorted(((f, phis.get(round(gts._header_time(f), 6))) for f in set(files)),
                   key=lambda p: gts._header_time(p[0]))
    if not pairs:
        raise SystemExit(f"no hydro_w dumps under {args.run_dir}")

    with Pool(args.nproc, initializer=gts._init_worker, initargs=(None,)) as pool:
        res = pool.map(one, pairs)
    outs = [r[0] for r in res]
    meta = res[0][1]
    np.savez_compressed(args.out, **{k: np.array([o[k] for o in outs]) for k in outs[0]},
                        **meta)
    print(f"wrote {args.out}: {len(outs)} dumps, "
          f"{sum(p[1] is not None for p in pairs)} with grav_phi")


if __name__ == "__main__":
    main()
