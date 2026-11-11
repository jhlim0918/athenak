#!/usr/bin/env python3
"""Phase 1 of the GI dust-growth plan: relative velocities of dust particles in gravito-turbulence.

From one particle VTK dump (prtcl_all: positions, pspecies, pvel) and, optionally, the hydro_w
dump at the same time, measured in the midplane slab |z| < zcut (whole MeshBlock rows):

  particle-particle, by species pair (a, b) and averaging scale s = 1, 2, 4 grid cells:
      collision-weighted mean square  <|v_i - v_j|^2>  over all pairs in the same s^3 cell,
      exact from per-cell moments:  a != b:  |u_a - u_b|^2 + sig_a^2 + sig_b^2,  weight n_a n_b
                                    a == b:  2 n/(n-1) sig^2,                   weight n(n-1)/2
      and PDFs of |dv| from sampled pairs (each particle of a with one random partner of b in
      its cell, weight = number of possible partners), split by gas environment
      (rho_g / <rho_g>(z) < 0.7 inter-arm, 0.7-1.5, > 1.5 arm) when the gas dump is given;
  particle-gas slip |v_p - u_g(x_p)| per species (nearest-grid-point gas velocity);
  the fraction of pairs with |dv| > v_frag for v_frag / c_s in VFRAG, using the LOCAL c_s.

Velocities are the stored shear-frame (FARGO) velocities: the background shear q Omega x is
removed for gas and particles alike, so within-cell shear (q Omega dx ~ 0.08 at 40 cells/H) is
not counted as a collision velocity; the s-dependence shows how much resolved structure
inside the averaging cell adds.

Usage (repo root):
  python3 scripts/analysis/gt_collision_velocities.py --vtk RUN/gtb40.prtcl_all.00010.part.vtk \
      --gas RUN/gtb40.hydro_w.00020.bin --taus 0.01,0.1,1 -o RUN/collision_velocities.npz
  (without --gas: particle-particle only, cells from --nx/--L)
"""
import argparse
import mmap
import re
import sys

import numpy as np

VFRAG = np.array([0.005, 0.01, 0.015, 0.025, 0.05, 0.1])     # v_frag / c_s (local)
DVBINS = np.logspace(-5, 1, 121)                               # |dv| / c_s histogram edges
ENV_EDGES = [0.0, 0.7, 1.5, np.inf]                            # rho_g / <rho_g>(z)
ENV_NAMES = ["interarm", "mean", "arm"]


def read_particles(path):
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    t = float(re.search(r"time\s*=\s*([\d.eE+-]+)", mm[:400].decode("latin1")).group(1))
    i = mm.find(b"POINTS"); j = mm.find(b"\n", i); N = int(mm[i:j].split()[1]); p0 = j + 1
    xyz = np.frombuffer(mm, ">f4", 3 * N, p0).reshape(N, 3)
    i = mm.find(b"SCALARS pspecies", p0 + 12 * N); j = mm.find(b"\n", mm.find(b"LOOKUP_TABLE", i)) + 1
    spec = np.frombuffer(mm, ">f4", N, j)
    i = mm.find(b"VECTORS pvel", j + 4 * N); j = mm.find(b"\n", i) + 1
    vel = np.frombuffer(mm, ">f4", 3 * N, j).reshape(N, 3)
    return t, N, xyz, spec, vel


def gas_slab(path, zcut, want=("dens", "velx", "vely", "velz", "eint")):
    """Memory-map an AthenaK uniform-grid .bin and assemble the requested variables on the
    MeshBlock rows that overlap |z| < zcut, as float32 arrays (k, j, i)."""
    fp = open(path, "rb")
    assert fp.readline().split()[0] == b"Athena"
    nph = int(fp.readline().split(b"=")[-1]); ph = {}
    for _ in range(nph - 1):
        k, v = [x.strip() for x in fp.readline().decode().split("=")]; ph[k] = v
    loc, var = int(ph["size of location"]), int(ph["size of variable"])
    nvars = int(fp.readline().split(b"=")[-1]); names = [v.decode() for v in fp.readline().split()[1:]]
    hsz = int(fp.readline().split(b"=")[-1]); hdr = fp.read(hsz).decode("latin1")
    off0 = fp.tell(); fp.close()

    def hget(block, key):
        cur = None
        for line in hdr.split("\n"):
            line = line.split("#")[0].strip()
            if line.startswith("<"):
                cur = line; continue
            if cur == block and line.partition("=")[0].strip() == key:
                return line.partition("=")[2].strip()
    N1, N2, N3 = (int(hget("<mesh>", k)) for k in ("nx1", "nx2", "nx3"))
    m1, m2, m3 = (int(hget("<meshblock>", k)) for k in ("nx1", "nx2", "nx3"))
    x1min, x1max = float(hget("<mesh>", "x1min")), float(hget("<mesh>", "x1max"))
    x3min, x3max = float(hget("<mesh>", "x3min")), float(hget("<mesh>", "x3max"))
    gamma = float(hget("<hydro>", "gamma"))
    dz = (x3max - x3min) / N3
    rec = 24 + 16 + 6 * loc + nvars * m1 * m2 * m3 * var
    dt = np.float32 if var == 4 else np.float64
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    nmb = (mm.size - off0) // rec
    nb3 = N3 // m3
    kb = [b for b in range(nb3) if x3min + (b + 1) * m3 * dz > -zcut and x3min + b * m3 * dz < zcut]
    kb0 = kb[0]; nzs = len(kb) * m3
    out = {v: np.empty((nzs, N2, N1), np.float32) for v in want}
    vidx = {v: names.index(v) for v in want}
    for m in range(nmb):
        o = off0 + m * rec
        lx = np.frombuffer(mm[o + 24:o + 40].tobytes(), np.int32)
        if lx[2] not in kb:
            continue
        data = np.frombuffer(mm[o + 40 + 6 * loc:o + rec].tobytes(), dt).reshape(nvars, m3, m2, m1)
        k0 = (lx[2] - kb0) * m3
        for v in want:
            out[v][k0:k0 + m3, lx[1] * m2:(lx[1] + 1) * m2, lx[0] * m1:(lx[0] + 1) * m1] = data[vidx[v]]
    zlo = x3min + kb0 * m3 * dz
    return out, dict(N1=N1, N2=N2, nzs=nzs, dx=(x1max - x1min) / N1, x1min=x1min,
                     x2min=float(hget("<mesh>", "x2min")), zlo=zlo, gamma=gamma, time=float(ph["time"]))


def cell_moments(cid, spec, vel, ns):
    """Per occupied cell: n_s, mean velocity (3), population variance (sum over components)."""
    uniq, inv = np.unique(cid, return_inverse=True)
    U = len(uniq); mom = []
    for s in range(ns):
        sel = spec == s
        n = np.bincount(inv[sel], minlength=U).astype(np.float64)
        mean = np.zeros((U, 3), np.float32); var = np.zeros(U, np.float64)
        nz = n > 0
        for c in range(3):
            v = vel[sel, c].astype(np.float64)
            s1 = np.bincount(inv[sel], weights=v, minlength=U)
            s2 = np.bincount(inv[sel], weights=v * v, minlength=U)
            mu = np.where(nz, s1 / np.maximum(n, 1), 0.0)
            var += np.where(nz, s2 / np.maximum(n, 1) - mu * mu, 0.0)
            mean[:, c] = mu
        mom.append((n, mean, np.maximum(var, 0.0)))
    return mom


def pair_ms(mom, a, b):
    na, ua, va = mom[a]; nb, ub, vb = mom[b]
    if a != b:
        w = na * nb
        dv2 = ((ua - ub).astype(np.float64) ** 2).sum(1) + va + vb
    else:
        w = 0.5 * na * (na - 1)
        dv2 = np.where(na > 1, 2 * na / np.maximum(na - 1, 1) * va, 0.0)
    W = w.sum()
    return (np.sqrt((w * dv2).sum() / W) if W > 0 else np.nan), W


def sample_pairs(cid, spec, vel, a, b, rng):
    """Each particle of species a gets one random partner of species b in its cell.
    Returns (index_a, index_b, weight = number of possible partners)."""
    ia = np.where(spec == a)[0]; ib = np.where(spec == b)[0]
    cb = cid[ib]; ob = np.argsort(cb, kind="stable"); cb = cb[ob]; ib = ib[ob]
    ca = cid[ia]
    lo = np.searchsorted(cb, ca, "left"); hi = np.searchsorted(cb, ca, "right")
    cnt = hi - lo
    if a == b:
        ok = cnt >= 2
        ia, lo, cnt = ia[ok], lo[ok], cnt[ok]
        # pick among the other cnt-1 particles: draw in [0, cnt-1), skip self
        r = (rng.random(len(ia)) * (cnt - 1)).astype(np.int64)
        pos_self = np.empty(len(ia), np.int64)
        # position of self within its cell block
        order_in_b = np.empty(len(spec), np.int64); order_in_b[ib] = np.arange(len(ib))
        pos_self = order_in_b[ia] - lo
        r = r + (r >= pos_self)
        jb = ib[lo + r]
        w = (cnt - 1).astype(np.float64)
    else:
        ok = cnt >= 1
        ia, lo, cnt = ia[ok], lo[ok], cnt[ok]
        r = (rng.random(len(ia)) * cnt).astype(np.int64)
        jb = ib[lo + r]
        w = cnt.astype(np.float64)
    return ia, jb, w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vtk", required=True)
    ap.add_argument("--gas")
    ap.add_argument("--taus", required=True, help="comma list of species stopping times (St)")
    ap.add_argument("--Hg", type=float, default=2.16878)
    ap.add_argument("--zcut", type=float, default=2.0, help="slab half-height in units of Hg")
    ap.add_argument("--nx", type=int, help="cells per x (no --gas)")
    ap.add_argument("--L", type=float, default=55.2302, help="box length in x, y (no --gas)")
    ap.add_argument("--cs", type=float, default=2.16878, help="reference c_s when no --gas")
    ap.add_argument("--scales", default="1,2,4")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ttol", type=float, default=1e-2, help="max |t_gas - t_particles| (one step is ~1e-3)")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    taus = [float(x) for x in a.taus.split(",")]; ns = len(taus)
    scales = [int(x) for x in a.scales.split(",")]
    rng = np.random.default_rng(a.seed)
    zcut = a.zcut * a.Hg

    t, N, xyz, spec_all, vel_all = read_particles(a.vtk)
    print(f"{a.vtk}: t = {t:g}, {N} particles", flush=True)

    if a.gas:
        g, gm = gas_slab(a.gas, zcut)
        assert abs(gm["time"] - t) < a.ttol, f"gas t = {gm['time']} vs particles t = {t}"
        dx, x1min, x2min, zlo, nzs = gm["dx"], gm["x1min"], gm["x2min"], gm["zlo"], gm["nzs"]
        N1, N2 = gm["N1"], gm["N2"]
        zhi = zlo + nzs * dx
        cs2 = (gm["gamma"] * (gm["gamma"] - 1)) * g["eint"] / g["dens"]
        rho_mean_z = g["dens"].mean(axis=(1, 2))
        print(f"gas slab z = [{zlo:.2f}, {zhi:.2f}] ({nzs} cells), dx = {dx:.4f}", flush=True)
    else:
        N1 = N2 = a.nx; dx = a.L / a.nx; x1min = x2min = -a.L / 2
        nzs = 2 * int(np.ceil(zcut / dx)); zlo = -nzs * dx / 2; zhi = -zlo

    Z = np.asarray(xyz[:, 2], np.float32)
    sel = np.where((Z > zlo) & (Z < zhi))[0]
    print(f"slab holds {len(sel)} particles ({100 * len(sel) / N:.1f}%)", flush=True)
    X = np.asarray(xyz[sel, 0], np.float32); Y = np.asarray(xyz[sel, 1], np.float32); Z = Z[sel]
    spec = np.asarray(spec_all[sel]).astype(np.int8); vel = np.asarray(vel_all[sel], np.float32)
    ii = ((X - x1min) / dx).astype(np.int64) % N1
    jj = ((Y - x2min) / dx).astype(np.int64) % N2
    kk = ((Z - zlo) / dx).astype(np.int64).clip(0, nzs - 1)
    del X, Y
    counts = np.bincount(spec, minlength=ns)
    print("particles per species in slab:", counts.tolist(), flush=True)

    res = dict(time=t, taus=np.array(taus), scales=np.array(scales), vfrag=VFRAG, dvbins=DVBINS,
               env_names=np.array(ENV_NAMES), counts=counts, dx=dx, Hg=a.Hg)

    if a.gas:
        fine = (kk * N2 + jj) * N1 + ii
        cs_p = np.sqrt(cs2.reshape(-1)[fine]).astype(np.float32)
        rho_p = g["dens"].reshape(-1)[fine]
        env = np.digitize(rho_p / rho_mean_z[kk], ENV_EDGES[1:-1])
        ug = np.column_stack([g[c].reshape(-1)[fine] for c in ("velx", "vely", "velz")])
        slip = np.sqrt(((vel - ug) ** 2).sum(1))
        res["slip_rms"] = np.array([np.sqrt((slip[spec == s] ** 2).mean()) for s in range(ns)])
        res["slip_over_cs_median"] = np.array([np.median(slip[spec == s] / cs_p[spec == s]) for s in range(ns)])
        res["slip_vx_mean"] = np.array([(vel[spec == s, 0] - ug[spec == s, 0]).mean() for s in range(ns)])
        res["vx_mean"] = np.array([vel[spec == s, 0].mean() for s in range(ns)])
        res["slip_hist"] = np.array([np.histogram(slip[spec == s] / cs_p[spec == s], DVBINS)[0] for s in range(ns)])
        res["env_frac"] = np.array([[np.mean(env[spec == s] == e) for e in range(3)] for s in range(ns)])
        res["cs_p_median"] = np.array([np.median(cs_p[spec == s]) for s in range(ns)])
        del ug, fine
        for s in range(ns):
            print(f"  St={taus[s]:g}: slip rms {res['slip_rms'][s]:.4f}, median slip/c_s "
                  f"{res['slip_over_cs_median'][s]:.4f}, <v_x - u_x> {res['slip_vx_mean'][s]:+.4f}, "
                  f"<v_x> {res['vx_mean'][s]:+.4f}, median c_s {res['cs_p_median'][s]:.3f}, "
                  f"env (inter/mean/arm) {np.round(res['env_frac'][s], 3).tolist()}", flush=True)
    else:
        cs_p = np.full(len(sel), a.cs, np.float32); env = np.ones(len(sel), np.int64)

    pairs = [(p, q) for p in range(ns) for q in range(p, ns)]
    res["pairs"] = np.array(pairs)
    rms = np.full((len(scales), len(pairs)), np.nan); Wt = np.zeros_like(rms)
    hist = np.zeros((len(scales), len(pairs), 3, len(DVBINS) - 1))
    ffrag = np.full((len(scales), len(pairs), 3, len(VFRAG)), np.nan)
    wenv = np.zeros((len(scales), len(pairs), 3))
    med = np.full((len(scales), len(pairs), 3), np.nan)
    for si, sc in enumerate(scales):
        n1c, n2c = N1 // sc, N2 // sc; n3c = (nzs + sc - 1) // sc
        cid = ((kk // sc) * n2c + (jj // sc)) * n1c + (ii // sc)
        mom = cell_moments(cid, spec, vel, ns)
        for pi, (p, q) in enumerate(pairs):
            rms[si, pi], Wt[si, pi] = pair_ms(mom, p, q)
            ia, jb, w = sample_pairs(cid, spec, vel, p, q, rng)
            dv = np.sqrt(((vel[ia] - vel[jb]).astype(np.float64) ** 2).sum(1))
            csl = 0.5 * (cs_p[ia] + cs_p[jb])
            e = env[ia]
            for ei in range(3):
                m = e == ei
                if m.sum() == 0:
                    continue
                x = dv[m] / csl[m]; ww = w[m]
                hist[si, pi, ei] = np.histogram(x, DVBINS, weights=ww)[0]
                wenv[si, pi, ei] = ww.sum()
                ffrag[si, pi, ei] = [(ww * (x > vf)).sum() / ww.sum() for vf in VFRAG]
                o = np.argsort(x); cw = np.cumsum(ww[o]); med[si, pi, ei] = x[o][np.searchsorted(cw, 0.5 * cw[-1])]
            print(f"scale {sc}: St {taus[p]:g}-{taus[q]:g}: rms dv {rms[si, pi]:.4f} "
                  f"(pair weight {Wt[si, pi]:.3g}); median dv/c_s (inter/mean/arm) "
                  f"{np.round(med[si, pi], 4).tolist()}", flush=True)
        del cid, mom
    res.update(rms=rms, pair_weight=Wt, hist=hist, ffrag=ffrag, wenv=wenv, median=med)
    np.savez_compressed(a.out, **res)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
