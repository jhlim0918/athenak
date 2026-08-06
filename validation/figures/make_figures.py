#!/usr/bin/env python3
"""Generate validation figures for the FFT self-gravity solver documentation.

Reads AthenaK .bin outputs from ../run/bin and the convergence table from
../run/shwave_convergence.dat; writes PNG figures into this directory.
Standalone: needs only numpy + matplotlib (no h5py).
"""
import struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "run"

# validated categorical palette (dataviz reference, light mode, fixed order)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#1a1a19", "#52514e"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight",
})


def read_bin(filename):
    """Minimal AthenaK .bin reader (format version 1.1, uniform grid)."""
    fp = open(filename, "rb")
    fp.seek(0, 2); filesize = fp.tell(); fp.seek(0, 0)
    assert fp.readline().split()[0] == b"Athena"
    pheader_count = int(fp.readline().split(b"=")[-1])
    ph = {}
    for _ in range(pheader_count - 1):
        key, val = [x.strip() for x in fp.readline().decode().split("=")]
        ph[key] = val
    locsize = int(ph["size of location"]); varsize = int(ph["size of variable"])
    nvars = int(fp.readline().split(b"=")[-1])
    var_list = [v.decode() for v in fp.readline().split()[1:]]
    header_size = int(fp.readline().split(b"=")[-1])
    header = [ln.decode().split("#")[0].strip()
              for ln in fp.read(header_size).split(b"\n")]
    header = [ln for ln in header if ln]

    def get(block, key):
        blk = "<none>"
        for ln in header:
            if ln.startswith("<"):
                blk = ln; continue
            k, v = ln.split("=")
            if blk == f"<{block}>" and k.strip() == key:
                return v.strip()
        raise KeyError(f"{block}/{key}")

    nghost = int(get("mesh", "nghost"))
    mesh = {k: float(get("mesh", k)) for k in
            ("x1min", "x1max", "x2min", "x2max", "x3min", "x3max")}
    Nx = [int(get("mesh", f"nx{d}")) for d in (1, 2, 3)]
    locfmt = np.float64 if locsize == 8 else np.float32
    varfmt = np.float64 if varsize == 8 else np.float32
    blocks = []
    while fp.tell() < filesize:
        idx = np.frombuffer(fp.read(24), dtype=np.int32) - nghost
        n1 = idx[1]-idx[0]+1; n2 = idx[3]-idx[2]+1; n3 = idx[5]-idx[4]+1
        logical = np.frombuffer(fp.read(16), dtype=np.int32)
        geom = np.frombuffer(fp.read(6*locsize), dtype=locfmt)
        data = np.fromfile(fp, dtype=varfmt, count=n1*n2*n3*nvars)
        blocks.append((logical, geom, data.reshape(nvars, n3, n2, n1)))
    fp.close()
    # assemble global arrays (uniform grid: place by logical location)
    n1b, n2b, n3b = blocks[0][2].shape[3], blocks[0][2].shape[2], blocks[0][2].shape[1]
    out = {v: np.empty((Nx[2], Nx[1], Nx[0])) for v in var_list}
    for logical, geom, data in blocks:
        i0, j0, k0 = logical[0]*n1b, logical[1]*n2b, logical[2]*n3b
        for vi, v in enumerate(var_list):
            out[v][k0:k0+n3b, j0:j0+n2b, i0:i0+n1b] = data[vi]
    coords = [np.linspace(mesh[f"x{d}min"], mesh[f"x{d}max"], Nx[d-1]+1)[:-1]
              + 0.5*(mesh[f"x{d}max"]-mesh[f"x{d}min"])/Nx[d-1] for d in (1, 2, 3)]
    return out, coords, mesh


# ---------------------------------------------------------------- fig 1: convergence
tab = {}
for line in open(RUN/"shwave_convergence.dat"):
    if line.startswith("#"): continue
    rm, n, mx, l2 = line.split()
    tab.setdefault(rm, []).append((int(n), float(mx), float(l2)))

fig, ax = plt.subplots(figsize=(4.6, 3.6))
for rm, color, label in (("ppmx", C1, "ppmx (3rd order)"),
                         ("plm", C2, "plm (2nd order)"),
                         ("dc", C3, "dc (2nd order)")):
    d = np.array(tab[rm])
    ax.loglog(d[:, 0], d[:, 1], "-o", color=color, lw=2, ms=6, label=label)
ns = np.array([32.0, 128.0])
for p, y0 in ((2, 1.4e-2), (3, 3.5e-4)):
    ax.loglog(ns, y0*(ns/32.0)**(-p), "--", color=MUTED, lw=1)
    ax.annotate(f"$N^{{-{p}}}$", (140, y0*(128/32.0)**(-p)), color=MUTED, fontsize=9)
ax.set_xlabel("resolution  $N$  (cells per side)")
ax.set_ylabel(r"max $|\Phi - \Phi_{\rm analytic}|\, /\, |\Phi_{\rm amp}|$")
ax.set_title("Shearing-wave solve: pointwise potential error", fontsize=10)
ax.set_xticks([32, 64, 128]); ax.set_xticklabels(["32", "64", "128"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xticks([], minor=True)
ax.legend(frameon=False, fontsize=9)
fig.savefig(HERE/"convergence.png")
plt.close(fig)

# ---------------------------------------------------------------- fig 1b: phase sweep
ps = np.loadtxt(RUN/"phase_sweep.dat")
fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.semilogy(ps[:, 0], np.maximum(ps[:, 1], 1e-16), "-o", color=C1, lw=2, ms=6)
ax.set_xlabel(r"$t_0 / T_{\rm sh}$   (shear phase)")
ax.set_ylabel("max Laplacian residual")
ax.set_title("Exactness at shear-periodic instants", fontsize=10)
ax.set_ylim(1e-15, 1e-1)
fig.savefig(HERE/"phase_sweep.png")
plt.close(fig)

# ---------------------------------------------------------------- fig 2: slab profile
phi_d, coords, mesh = read_bin(RUN/"bin"/"FFTGravityOpen.grav_phi.00000.bin")
z = coords[2]
phi_col = phi_d["grav_phi"][:, 0, 0]
H, rho0, fpg = 0.05, 1.0, 1.0
zc = 0.5*(mesh["x3min"]+mesh["x3max"])
ana = fpg*rho0*H**2*np.log(np.cosh((z-zc)/H))
phi_c = phi_col - phi_col.mean()
ana_c = ana - ana.mean()

fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.6, 4.4), sharex=True,
                             height_ratios=[3, 1.3])
a1.plot(z, ana_c, "-", color=C2, lw=2, label="analytic (infinite slab)")
a1.plot(z[::2], phi_c[::2], "o", color=C1, ms=4.5, mfc="none", label="FFT solver")
a1.set_ylabel(r"$\Phi(z) - \langle\Phi\rangle$")
a1.set_title(r"Open vertical BC: sech$^2$ slab, $H = L_z/20$, $64^3$", fontsize=10)
a1.legend(frameon=False, fontsize=9)
a2.semilogy(z, np.abs(phi_c-ana_c)/np.abs(ana_c).max(), "-", color=C1, lw=1.5)
a2.set_ylabel("rel. error"); a2.set_xlabel(r"$z$")
fig.savefig(HERE/"slab_profile.png")
plt.close(fig)

# ---------------------------------------------------------------- fig 3: shwave slice
w_d, coords, mesh = read_bin(RUN/"bin"/"FFTGravityShwave.hydro_w.00000.bin")
phi_d, _, _ = read_bin(RUN/"bin"/"FFTGravityShwave.grav_phi.00000.bin")
x, y = coords[0], coords[1]
kmid = len(coords[2])//2
drho = w_d["dens"][kmid] - 1.0
phi_s = phi_d["grav_phi"][kmid] - phi_d["grav_phi"][kmid].mean()

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
for ax, f, ttl in ((axes[0], drho, r"$\rho - \rho_0$"),
                   (axes[1], phi_s, r"$\Phi - \langle\Phi\rangle$")):
    lim = np.abs(f).max()
    im = ax.pcolormesh(x, y, f, cmap="RdBu_r", vmin=-lim, vmax=lim, rasterized=True)
    ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_title(ttl, fontsize=10)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.9)
axes[0].set_ylabel("$y$")
fig.suptitle("Shearing wave (rolled-frame mode $n=(1,1,1)$) at $q\\Omega t_s=0.555$"
             ": density and solved potential, $z=0$ slice", fontsize=10)
fig.savefig(HERE/"shwave_slice.png")
plt.close(fig)

# ---------------------------------------------------------------- fig 4: swing amplification
def read_hst(path):
    rows = [np.array(ln.split(), dtype=float)
            for ln in open(path) if not ln.strip().startswith("#")]
    return np.array(rows)


def swing_ode(fpg, t1=8.0, nsteps=200000, kx0=-6.0, ky=1.0, q=1.5, Om=1.0,
              cs=1.0, rho0=1.0, dx=2*np.pi/128, amp=1e-4):
    def rhs(u, t):
        d, wx, wy = u
        kx = kx0 + q*Om*ky*t
        D = (2*np.cos(kx*dx)-2)/dx**2 + (2*np.cos(ky*dx)-2)/dx**2
        S = d*(cs**2 + fpg*rho0/D)
        return np.array([-(kx*wx + ky*wy), 2*Om*wy + kx*S, -(2-q)*Om*wx + ky*S])
    h = t1/nsteps
    u = np.array([amp, 0.0, 0.0])
    ts, out = [0.0], [u.copy()]
    for n in range(nsteps):
        t = n*h
        k1 = rhs(u, t); k2 = rhs(u+h/2*k1, t+h/2)
        k3 = rhs(u+h/2*k2, t+h/2); k4 = rhs(u+h*k3, t+h)
        u = u + h/6*(k1+2*k2+2*k3+k4)
        ts.append(t+h); out.append(u.copy())
    return np.array(ts), np.array(out)

amp0 = 1e-4
hg = read_hst(RUN/"SwingSG.user.hst")
tsw = 6.0/1.5
ts_g, u_g = swing_ode(1.9)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.4, 5.2), sharex=True,
                             height_ratios=[3, 2])
a1.axvline(tsw, color="#999", ls="--", lw=1)
a1.plot(ts_g, u_g[:, 0]/amp0, "-", color=C2, lw=2.5, label="linear theory")
a1.plot(hg[::8, 0], hg[::8, 2]/amp0, "o", color=C1, ms=4, mfc="none",
        label="AthenaK + FFT self-gravity")
a1.set_ylabel(r"$\delta / \delta_0$")
a1.set_title("Swing amplification of a leading shearing wave", fontsize=10)
a1.legend(frameon=False, fontsize=9, loc="upper left")
a1.annotate(r"$k_x=0$", (tsw, a1.get_ylim()[0]*0.85), fontsize=9, color="#666")

try:
    hn = read_hst(RUN/"SwingNoG128.user.hst")
    ts_n, u_n = swing_ode(0.0)
    a2.axvline(tsw, color="#999", ls="--", lw=1)
    a2.plot(ts_n, u_n[:, 0]/amp0, "-", color=C2, lw=2.5, label="linear theory, no gravity")
    a2.plot(hn[::8, 0], hn[::8, 2]/amp0, "o", color=C3, ms=4, mfc="none",
            label="control: self-gravity off")
    a2.set_ylabel(r"$\delta / \delta_0$")
    a2.legend(frameon=False, fontsize=9, loc="upper left")
except FileNotFoundError:
    pass
a2.set_xlabel(r"$t$   $(\Omega^{-1})$")
fig.savefig(HERE/"swing.png")
plt.close(fig)

print("wrote:", [p.name for p in sorted(HERE.glob("*.png"))])
