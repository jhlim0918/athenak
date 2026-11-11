#!/usr/bin/env python3
"""Initial (t = 0) velocity seed of an SC14 gravito_turb run: slices and spectra.

The pgen seeds each velocity component with independent cell-to-cell uniform noise of
amplitude pert_amp*cs0 for |z| < pert_zmax (src/pgen/fluids/gravito_turb.cpp, HashNoise).
This writes, from the hydro_w dump at t = 0:
  slices     vx, vy, vz in the midplane layer (cell just above z = 0), vz in the x-z plane
             at y = 0
  P_x, P_y   1D power per mode |v_hat|^2/N^2 along x and along y, per component, averaged
             over the other horizontal direction and over the perturbed layers
  E_k        2D shell spectrum in (kx, ky), summed over the three components and averaged
             over the perturbed layers: energy per unit k, normalised so sum E_k dk =
             <v^2>/2 in the layer
  cumfrac    cumulative fraction of that kinetic energy at wavenumbers below k

Usage: python3 gt_initial_velocity_extract.py <run_dir> -o out.npz
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import extract_gt_stress as gts  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    gts._init_worker(None)
    f = None
    for sub in ("bin", ""):
        p = os.path.join(args.run_dir, sub, "GravitoTurb.hydro_w.00000.bin")
        if os.path.isfile(p):
            f = p
            break
    if f is None:
        raise SystemExit(f"no t = 0 hydro_w dump under {args.run_dir}")
    fd = gts._READ(f)
    v = {c: gts.assemble_root(fd, n) for c, n in (("vx", "velx"), ("vy", "vely"), ("vz", "velz"))}
    nx1, nx2, nx3 = fd["Nx1"], fd["Nx2"], fd["Nx3"]
    dx = (fd["x1max"] - fd["x1min"]) / nx1
    dy = (fd["x2max"] - fd["x2min"]) / nx2
    dz = (fd["x3max"] - fd["x3min"]) / nx3
    zc = fd["x3min"] + (np.arange(nx3) + 0.5) * dz
    lay = np.where(np.any([np.any(v[c] != 0, axis=(1, 2)) for c in v], axis=0))[0]
    kmid = np.argmin(np.abs(zc - 0.5 * dz))

    kx = 2 * np.pi * np.fft.rfftfreq(nx1, dx)
    ky = 2 * np.pi * np.fft.rfftfreq(nx2, dy)
    px = {c: (np.abs(np.fft.rfft(v[c][lay], axis=2)) ** 2).mean(axis=(0, 1)) / nx1**2 for c in v}
    py = {c: (np.abs(np.fft.rfft(v[c][lay], axis=1)) ** 2).mean(axis=(0, 2)) / nx2**2 for c in v}

    # 2D shell spectrum in the (kx, ky) plane
    KX = 2 * np.pi * np.fft.fftfreq(nx1, dx)[None, :]
    KY = 2 * np.pi * np.fft.fftfreq(nx2, dy)[:, None]
    kk = np.sqrt(KX**2 + KY**2)
    dk = 2 * np.pi / (nx1 * dx)
    edges = np.arange(0.5 * dk, kk.max() + dk, dk)
    p2 = 0.0
    for c in v:
        p2 = p2 + (np.abs(np.fft.fft2(v[c][lay], axes=(1, 2))) ** 2).mean(axis=0) / (nx1 * nx2) ** 2
    p2 = 0.5 * p2                                       # kinetic energy per unit mass
    idx = np.digitize(kk.ravel(), edges)
    ek = np.bincount(idx, weights=p2.ravel(), minlength=len(edges) + 1)[1:len(edges)] / dk
    kc = 0.5 * (edges[1:] + edges[:-1])
    cum = np.cumsum(ek * dk) / p2.sum()

    out = dict(time=fd["time"], x1min=fd["x1min"], x1max=fd["x1max"], x2min=fd["x2min"],
               x2max=fd["x2max"], zc=zc, kmid=kmid, layers=lay, kx=kx, ky=ky, k=kc, E_k=ek,
               cumfrac=cum, ke_layer=p2.sum(), xz_vz=v["vz"][:, nx2 // 2, :].astype(np.float32))
    for c in v:
        out[f"mid_{c}"] = v[c][kmid].astype(np.float32)
        out[f"px_{c}"] = px[c]
        out[f"py_{c}"] = py[c]
        out[f"rms_{c}"] = v[c][lay].std()
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out}: t = {fd['time']}, perturbed layers {lay[0]}..{lay[-1]} "
          f"(z = {zc[lay[0]]:+.3f}..{zc[lay[-1]]:+.3f}), midplane slice k = {kmid}")


if __name__ == "__main__":
    main()
