#!/usr/bin/env python3
"""Total-energy budget of an SC14 gravito_turb stage from paired hydro_w + grav_phi dumps,
including the energy carried out through the z faces.

For the gas in the box, with E_tot = K + U + E_tid + E_sg,
    K     = int (1/2) rho v^2            (v = vx, dvy, vz: FARGO frame, shear removed)
    U     = int e_int
    E_tid = int (1/2) rho Omega^2 z^2    (linearised vertical tidal potential)
    E_sg  = (1/2) int rho Phi            (self-gravity; Phi is linear in rho)
the energy equation integrates to
    dE_tot/dt = S - C - F_z + R
    S   = q Omega int w_xy,  w_xy = rho vx dvy + gx gy / 4 pi G   (stress work)
    C   = int U / beta                                             (beta cooling)
    F_z = net outward flux through the two z faces of
          rho vz [ (1/2) v^2 + gamma P / ((gamma-1) rho) + (1/2) Omega^2 z^2 + Phi ]
          (outermost active cells, cell-centred: an estimate of the face flux)
    R   = residual: floors, the non-conservative self-gravity coupling, numerics.
Also the mass flux through the faces vs dM/dt.  Rates are averaged over the dumps in
[tmin, tmax]; dE_tot/dt and dM/dt come from linear fits to the dump series.

Usage: python3 gt_energy_budget.py <run_dir> --beta 80 --tmin 950 --tmax 1150 [-j 16]
"""

import argparse
import os
import sys
from glob import glob
from multiprocessing import Pool

import numpy as np

for p in (os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
          os.path.join(os.environ.get("ATHENAK", ""), "scripts")):
    if os.path.isfile(os.path.join(p, "extract_gt_stress.py")):
        sys.path.insert(0, p)
        break
import extract_gt_stress as gts  # noqa: E402

QSH = 1.5
OMEGA = 1.0


def one(pair):
    fw, fp = pair
    if gts._READ is None:
        gts._init_worker(None)
    fd, fdp = gts._READ(fw), gts._READ(fp)
    if abs(fd["time"] - fdp["time"]) > 1e-8:
        raise RuntimeError(f"time mismatch {fw} {fp}")
    gamma = float(gts.header_get(fd["header"], "<hydro>", "gamma"))
    fpg = float(gts.header_get(fd["header"], "<gravity>", "four_pi_G"))
    nx1, nx2, nx3 = fd["Nx1"], fd["Nx2"], fd["Nx3"]
    dx = (fd["x1max"] - fd["x1min"]) / nx1
    dy = (fd["x2max"] - fd["x2min"]) / nx2
    dz = (fd["x3max"] - fd["x3min"]) / nx3
    dV, dA = dx * dy * dz, dx * dy
    z = (fd["x3min"] + (np.arange(nx3) + 0.5) * dz)[:, None, None]

    rho = gts.assemble_root(fd, "dens")
    vx = gts.assemble_root(fd, "velx")
    vy = gts.assemble_root(fd, "vely")
    vz = gts.assemble_root(fd, "velz")
    eint = gts.assemble_root(fd, "eint")
    phi = gts.assemble_root(fdp, "grav_phi" if "grav_phi" in fdp["mb_data"] else "phi")
    prs = (gamma - 1.0) * eint
    v2 = vx**2 + vy**2 + vz**2
    tid = 0.5 * OMEGA**2 * z**2

    gx = np.empty_like(phi)
    gx[:, :, 1:-1] = -(phi[:, :, 2:] - phi[:, :, :-2]) / (2 * dx)
    gx[:, :, 0] = -(phi[:, :, 1] - phi[:, :, 0]) / dx
    gx[:, :, -1] = -(phi[:, :, -1] - phi[:, :, -2]) / dx
    gy = -(np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * dy)

    # specific energy carried across a face, and the outward mass / energy fluxes
    spec = 0.5 * v2 + gamma * prs / ((gamma - 1.0) * rho) + tid + phi
    out = {}
    for name, k, sgn in (("top", -1, +1.0), ("bot", 0, -1.0)):
        fm = sgn * rho[k] * vz[k]
        out[f"mflux_{name}"] = fm.sum() * dA
        out[f"eflux_{name}"] = (fm * spec[k]).sum() * dA
        # split of the energy flux by carrier, for the record
        out[f"eflux_{name}_kin"] = (fm * 0.5 * v2[k]).sum() * dA
        out[f"eflux_{name}_enth"] = (fm * gamma * prs[k] / ((gamma - 1.0) * rho[k])).sum() * dA
        out[f"eflux_{name}_tid"] = (fm * tid[k]).sum() * dA
        out[f"eflux_{name}_phi"] = (fm * phi[k]).sum() * dA
    out.update(
        time=fd["time"], mass=rho.sum() * dV,
        K=0.5 * (rho * v2).sum() * dV, U=eint.sum() * dV,
        Etid=(rho * tid).sum() * dV, Esg=0.5 * (rho * phi).sum() * dV,
        wrey=(rho * vx * vy).sum() * dV, wgrv=(gx * gy / fpg).sum() * dV,
    )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--tmin", type=float, required=True)
    ap.add_argument("--tmax", type=float, required=True)
    ap.add_argument("-j", "--nproc", type=int, default=8)
    ap.add_argument("-o", "--out")
    args = ap.parse_args()
    gts._init_worker(None)

    phis = {}
    for sub in ("", "bin"):
        for f in glob(os.path.join(args.run_dir, sub, "*.grav_phi.*.bin")):
            phis[round(gts._header_time(f), 6)] = f
    pairs = []
    for sub in ("", "bin"):
        for f in glob(os.path.join(args.run_dir, sub, "*.hydro_w.*.bin")):
            t = round(gts._header_time(f), 6)
            if args.tmin - 1e-6 <= t <= args.tmax + 1e-6 and t in phis:
                pairs.append((f, phis[t]))
    pairs.sort(key=lambda p: gts._header_time(p[0]))
    if len(pairs) < 3:
        raise SystemExit("fewer than 3 paired dumps in the window")

    with Pool(args.nproc, initializer=gts._init_worker, initargs=(None,)) as pool:
        res = pool.map(one, pairs)
    d = {k: np.array([r[k] for r in res]) for k in res[0]}
    t = d["time"]
    Etot = d["K"] + d["U"] + d["Etid"] + d["Esg"]
    slope = lambda y: np.polyfit(t, y, 1)[0]
    n = len(t)
    sem = lambda y: y.std() / np.sqrt(max(1.0, (t[-1] - t[0]) / 10.0))

    S = QSH * OMEGA * (d["wrey"] + d["wgrv"])
    C = d["U"] / args.beta
    Fz = d["eflux_top"] + d["eflux_bot"]
    Fm = d["mflux_top"] + d["mflux_bot"]
    dE = slope(Etot)
    R = dE - (S.mean() - C.mean() - Fz.mean())

    print(f"{args.run_dir}: {n} paired dumps, t = {t[0]:.1f}-{t[-1]:.1f}, beta = {args.beta:g}")
    print(f"  mean M = {d['mass'].mean():.1f};  dM/dt (fit) = {slope(d['mass']):+.3f};  "
          f"outflow through z faces = {Fm.mean():.3f} +- {sem(Fm):.3f}")
    print(f"  energy reservoirs (means): K = {d['K'].mean():.1f}  U = {d['U'].mean():.1f}  "
          f"E_tid = {d['Etid'].mean():.1f}  E_sg = {d['Esg'].mean():.1f}")
    print(f"  slopes: dK/dt {slope(d['K']):+.3f}  dU/dt {slope(d['U']):+.3f}  "
          f"dEtid/dt {slope(d['Etid']):+.3f}  dEsg/dt {slope(d['Esg']):+.3f}")
    c = C.mean()
    print(f"  rates (and as fractions of cooling C = {c:.2f}):")
    for lab, v, e in (("stress work S", S.mean(), sem(S)), ("cooling C", c, sem(C)),
                      ("z-face energy loss F_z", Fz.mean(), sem(Fz)),
                      ("  of which kinetic", (d['eflux_top_kin'] + d['eflux_bot_kin']).mean(), None),
                      ("  of which enthalpy", (d['eflux_top_enth'] + d['eflux_bot_enth']).mean(), None),
                      ("  of which tidal pot.", (d['eflux_top_tid'] + d['eflux_bot_tid']).mean(), None),
                      ("  of which self-grav. Phi", (d['eflux_top_phi'] + d['eflux_bot_phi']).mean(), None),
                      ("dE_tot/dt (fit)", dE, None),
                      ("residual R = dE/dt - (S - C - F_z)", R, None)):
        es = f" +- {e:8.2f}" if e is not None else " " * 12
        print(f"    {lab:36s} {v:+10.2f}{es}   ({v / c:+.3f} C)")
    print(f"  S / C = {S.mean() / c:.3f};  (S - F_z) / C = {(S.mean() - Fz.mean()) / c:.3f};  "
          f"(C + F_z + dE/dt) / S = {(c + Fz.mean() + dE) / S.mean():.3f}")
    if args.out:
        np.savez_compressed(args.out, **d, Etot=Etot, beta=args.beta)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
