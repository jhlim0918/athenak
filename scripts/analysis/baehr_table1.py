#!/usr/bin/env python3
"""Table-1 diagnostics of Baehr, Zhu & Yang (2022) for an AthenaK dust run (the
two-stage BR/noBR setup of scripts/cluster/gt_baehr_*.slurm).

Reads, from a run directory:
  <base>.user.hst   the gravito_turb history (SC14 columns + the dust columns)
  <base>.phst       the particle history (sig_z = H_d, velocity dispersions)
  pvtk/*.vtk        particle snapshots (tag-matched displacements -> diffusion)
  bin/*.dust_dpm.*  the module's dust density (Roche-density clump census)
and prints one Table-1 row averaged over [t0, t1] (their 50 <= t <= 80):
  Q      = <cs>_rho Omega / (pi G Sigma),  <cs>_rho = h[rho_cs]/h[mass], Sigma = mass/(Lx Ly)
  alphaR = (2/3) <rho vx dvy> / <rho cs^2>          (their eq. 21; hst wrey / rho_cs2)
  alphaG = (2/3) <gx gy> / (4 pi G <rho cs^2>)      (their eq. 22; hst wgrv / rho_cs2)
  H_d    = sig_z of the phst (in H_g)
  delta_x, delta_z = D / (cs H_g), D = d<(x(t)-x(t0))^2>/dt / 2 from tag-matched pvtk
           snapshots (x wrapped over Lx; z not wrapped; y is sheared and not used)
  Sc     = (alphaR + alphaG) / delta_z
  sigma_x, sigma_z = phst velocity dispersions in units of <cs>_rho
Units: SC14 code units; H_g = cs0/Omega (problem/cs0 of the stage-1 input), G from
<gravity>/four_pi_G.  Usage:
  python3 scripts/analysis/baehr_table1.py <rundir> [--t0 50 --t1 80 --hg 2.16878
          --four_pi_G 4.25231 --lx 55.2302 --ly 55.2302 --base gtb --roche]
"""
import argparse, glob, os, re, sys
import numpy as np

def read_hst(fn):
    cols = None
    with open(fn) as f:
        for l in f:
            if l.startswith('#') and '[1]=' in l:
                cols = [m.group(1) for m in re.finditer(r'\[\d+\]=(\S+)', l)]
                break
    d = np.loadtxt(fn)
    return cols, d

def read_pvtk(fn):
    """Legacy binary VTK point file written by AthenaK's ParticleVTKOutput: POINTS (float,
    big-endian), SCALARS gid/ptag/pspecies (float), VECTORS pvel (float)."""
    with open(fn, 'rb') as f:
        data = f.read()
    def find(key):
        i = data.find(key.encode())
        return i
    i = find('POINTS'); j = data.find(b'\n', i)
    npts = int(data[i:j].split()[1]); pos0 = j + 1
    xyz = np.frombuffer(data, dtype='>f4', count=3*npts, offset=pos0).reshape(npts, 3).astype(float)
    out = {'xyz': xyz}
    for key in ('gid', 'ptag', 'pspecies'):
        i = find('SCALARS ' + key)
        if i < 0: continue
        j = data.find(b'LOOKUP_TABLE', i); j = data.find(b'\n', j) + 1
        out[key] = np.frombuffer(data, dtype='>f4', count=npts, offset=j).astype(float)
    i = find('VECTORS pvel')
    if i >= 0:
        j = data.find(b'\n', i) + 1
        out['vel'] = np.frombuffer(data, dtype='>f4', count=3*npts, offset=j).reshape(npts, 3).astype(float)
    return out

def pvtk_time(fn):
    with open(fn, 'rb') as f:
        head = f.read(400).decode('latin1')
    m = re.search(r'time\s*=\s*([\d.eE+-]+)', head)
    return float(m.group(1)) if m else np.nan

def diffusion(rundir, t0, t1, lx):
    files = sorted(glob.glob(os.path.join(rundir, 'pvtk', '*.vtk')))
    if len(files) < 3:
        return np.nan, np.nan, 0
    snaps = [(pvtk_time(f), f) for f in files]
    snaps = [s for s in snaps if t0 <= s[0] <= t1]
    if len(snaps) < 3:
        return np.nan, np.nan, len(snaps)
    ref = read_pvtk(snaps[0][1]); tref = snaps[0][0]
    order = np.argsort(ref['ptag']); tags_ref = ref['ptag'][order]; x_ref = ref['xyz'][order]
    ts, dx2, dz2 = [], [], []
    for t, f in snaps[1:]:
        s = read_pvtk(f); o = np.argsort(s['ptag']); tags = s['ptag'][o]; x = s['xyz'][o]
        common, ia, ib = np.intersect1d(tags_ref, tags, return_indices=True)
        if len(common) < 10: continue
        d = x[ib] - x_ref[ia]
        d[:, 0] -= lx*np.round(d[:, 0]/lx)           # periodic wrap in x
        ts.append(t - tref); dx2.append(np.mean(d[:, 0]**2)); dz2.append(np.mean(d[:, 2]**2))
    ts, dx2, dz2 = map(np.array, (ts, dx2, dz2))
    if len(ts) < 2:
        return np.nan, np.nan, len(ts)
    # D = (1/2) d<dx^2>/dt from a linear fit through the origin
    Dx = 0.5*np.sum(ts*dx2)/np.sum(ts*ts); Dz = 0.5*np.sum(ts*dz2)/np.sum(ts*ts)
    return Dx, Dz, len(ts)

def roche_census(rundir, t0, t1, rho_roche):
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'vis', 'python'))
        import bin_convert as bc
        from scipy import ndimage
    except Exception as e:
        return None
    files = sorted(glob.glob(os.path.join(rundir, 'bin', '*.dust_dpm.*.bin')))
    res = []
    for f in files:
        d = bc.read_binary(f)
        if not (t0 <= d['time'] <= t1): continue
        N1, N2, N3 = d['Nx1'], d['Nx2'], d['Nx3']; A = np.zeros((N3, N2, N1))
        dx = (d['x1max'] - d['x1min'])/N1
        for m in range(d['n_mbs']):
            g = d['mb_geometry'][m]; blk = d['mb_data']['dustdpm'][m]
            i0 = int(round((g[0]-d['x1min'])/dx)); j0 = int(round((g[2]-d['x2min'])/dx)); k0 = int(round((g[4]-d['x3min'])/dx))
            A[k0:k0+blk.shape[0], j0:j0+blk.shape[1], i0:i0+blk.shape[2]] = blk
        lab, n = ndimage.label(A >= rho_roche)
        vol = dx**3
        masses = ndimage.sum(A, lab, index=range(1, n+1))*vol if n > 0 else np.array([])
        res.append((d['time'], n, masses.max() if n > 0 else 0.0, A.max()))
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rundir'); ap.add_argument('--base', default='gtb')
    ap.add_argument('--t0', type=float, default=50.0); ap.add_argument('--t1', type=float, default=80.0)
    ap.add_argument('--hg', type=float, default=2.16878); ap.add_argument('--omega', type=float, default=1.0)
    ap.add_argument('--four_pi_G', type=float, default=4.25231)
    ap.add_argument('--lx', type=float, default=55.2302); ap.add_argument('--ly', type=float, default=55.2302)
    ap.add_argument('--roche', action='store_true', help='Roche-density clump census from the dust_dpm dumps')
    a = ap.parse_args()
    G = a.four_pi_G/(4*np.pi)
    cols, h = read_hst(os.path.join(a.rundir, a.base + '.user.hst'))
    ci = {n: i for i, n in enumerate(cols)}
    w = (h[:, 0] >= a.t0) & (h[:, 0] <= a.t1)
    if w.sum() == 0:
        sys.exit(f"no history rows in [{a.t0},{a.t1}] (history spans {h[0,0]:.1f}-{h[-1,0]:.1f})")
    hw = h[w]
    mass = hw[:, ci['mass']]; cs_rho = hw[:, ci['rho_cs']]/mass; sigma = mass/(a.lx*a.ly)
    Q = np.mean(cs_rho*a.omega/(np.pi*G*sigma))
    rho_cs2 = hw[:, ci['rho_cs2']]
    alphaR = np.mean((2.0/3.0)*hw[:, ci['wrey']]/rho_cs2)
    alphaG = np.mean((2.0/3.0)*hw[:, ci['wgrv']]/rho_cs2)
    csm = np.mean(cs_rho)
    print(f"# {a.rundir}: t in [{a.t0},{a.t1}], {w.sum()} history rows")
    print(f"Q        = {Q:.3f}")
    print(f"alpha_R  = {alphaR:.3e}")
    print(f"alpha_G  = {alphaG:.3e}")
    # dust: phst
    pf = os.path.join(a.rundir, a.base + '.phst')
    Hd = sx = sz = np.nan
    if os.path.exists(pf):
        pc, p = read_hst(pf); pi_ = {n: i for i, n in enumerate(pc)}
        pw = (p[:, 0] >= a.t0) & (p[:, 0] <= a.t1)
        if pw.sum() > 0:
            Hd = np.mean(p[pw, pi_['sig_z_1']])/a.hg
            sx = np.mean(p[pw, pi_['sig_vx_1']])/csm; sz = np.mean(p[pw, pi_['sig_vz_1']])/csm
            print(f"H_d      = {Hd:.3e} H_g   (sig_z averaged; H_g = {a.hg})")
            print(f"sigma_dx = {sx:.3f} cs   sigma_dz = {sz:.3f} cs   (cs = <cs>_rho = {csm:.3f})")
            print(f"m_escaped = {p[pw, pi_['m_escaped']][-1]:.3e}   dpm_max = {np.mean(p[pw, pi_['dpm_max']]):.3e}")
    if 'd_mass' in ci:
        print(f"dust mass = {np.mean(hw[:, ci['d_mass']]):.4e} (Z = {np.mean(hw[:, ci['d_mass']]/mass):.4f})")
    Dx, Dz, nsn = diffusion(a.rundir, a.t0, a.t1, a.lx)
    if np.isfinite(Dx):
        dxn = Dx/(csm*a.hg); dzn = Dz/(csm*a.hg)
        print(f"delta_dx = {dxn:.3e}   delta_dz = {dzn:.3e}   ({nsn} pvtk snapshots)")
        if np.isfinite(dzn) and dzn > 0:
            print(f"Sc       = {(alphaR+alphaG)/dzn:.1f}")
    else:
        print(f"diffusion: not enough pvtk snapshots in the window ({nsn})")
    if a.roche:
        rho_R = 3.5*a.omega**2/G      # Chandrasekhar 1963 (their eq. 3)
        res = roche_census(a.rundir, a.t0, a.t1, rho_R)
        if res:
            for t, n, mmax, dmax in res:
                print(f"  t={t:7.2f}: clumps above rho_R={rho_R:.2f}: {n:4d}  max clump mass {mmax:.3e}  max rho_d {dmax:.3e}")
        else:
            print("Roche census: no dust_dpm dumps in the window (or scipy missing)")

if __name__ == '__main__':
    main()
