#!/usr/bin/env python3
"""The dusty Jeans instability of Krapp et al. (2024, ApJS 271, 7, sec. 3.5) for the
superparticle dust of the dust track: the linear eigenproblem, the initial condition
of the fastest-growing mode as command-line overrides for pgen dust_jeans, and the
growth-rate fit of a run's history.

Units: cs = 1, rho0 = 1, and four_pi_G chosen so that the dusty Jeans wavenumber
kJ = sqrt(4 pi G rho0 (1 + eps0))/cs = 1; k is then k/kJ and Re(s) is sigma in units
of cs kJ.  Perturbations ~ exp(i k x + s t).

  dust_jeans.py eig   --k 0.6 --eps 0.01 --ts 1 [--nx1 128 --ny 32]   # IC overrides + sigma
  dust_jeans.py eig   --k 0.6 --nspec 128 --input in.athinput --out run.athinput
                      # Krapp fig. 6: 128 species, eps0 = 0.01, ts linear in [1e-4, 10];
                      # writes a full input file (the command line cannot add absent keys)
  dust_jeans.py curve --eps 0.01 --ts 1 [--kmin --kmax]    # analytic sigma(k) table
  dust_jeans.py fit   run/jeans.user.hst [--tmin --tmax]   # measured sigma per column
"""
import sys, re, argparse, numpy as np

def matrix(k, eps, ts, cs=1.0, rho0=1.0, fpg=None, br=True):
    eps = np.asarray(eps, float); ts = np.asarray(ts, float); N = len(eps)
    if fpg is None: fpg = cs*cs/(1.0 + eps.sum())          # kJ = 1
    n = 2 + 2*N; M = np.zeros((n, n), complex)
    g = 1j*fpg/k                                           # -ik*phi = (i 4piG/k) * sum(drho)
    M[0, 1] = -1j*k*rho0
    M[1, 0] = -1j*k*cs*cs/rho0 + g
    for m in range(N):
        if br:
            M[1, 1] += -eps[m]/ts[m]
            M[1, 3+2*m] += eps[m]/ts[m]
        M[1, 2+2*m] += g
    for i in range(N):
        M[2+2*i, 3+2*i] = -1j*k*eps[i]*rho0
        M[3+2*i, 3+2*i] = -1.0/ts[i]
        M[3+2*i, 1] = 1.0/ts[i]
        M[3+2*i, 0] += g
        for m in range(N):
            M[3+2*i, 2+2*m] += g
    return M, fpg

def fastest(k, eps, ts, br=True, **kw):
    M, fpg = matrix(k, eps, ts, br=br, **kw)
    s, V = np.linalg.eig(M)
    j = np.argmax(s.real)
    v = V[:, j]/V[0, j]                                    # drho_g real and positive
    return s[j], v, fpg

def apply_overrides(base, out, ov):
    """Write a copy of athinput `base` to `out` with the block/key=value overrides applied
    (existing keys replaced in place, absent keys appended to their block; the command
    line cannot add absent keys, which matters for many-species runs)."""
    lines = open(base).read().split('\n'); blocks = {}; cur = None
    for i, l in enumerate(lines):
        m = re.match(r'\s*<(\w+)>', l)
        if m: cur = m.group(1); blocks.setdefault(cur, [i, i])
        elif cur is not None and l.strip() and not l.lstrip().startswith('#'): blocks[cur][1] = i
    add = {}
    for o in ov:
        bk, val = o.split('=', 1); b, key = bk.split('/')
        done = False
        if b in blocks:
            for i in range(blocks[b][0]+1, blocks[b][1]+1):
                if re.match(r'\s*'+re.escape(key)+r'\s*=', lines[i]):
                    lines[i] = f"{key} = {val}"; done = True; break
        if not done: add.setdefault(b, []).append(f"{key} = {val}")
    for b in sorted(add, key=lambda b: -blocks.get(b, [len(lines)]*2)[1]):
        if b in blocks: lines[blocks[b][1]+1:blocks[b][1]+1] = add[b]
        else: lines += ['', f'<{b}>'] + add[b]
    open(out, 'w').write('\n'.join(lines))

def cmd_eig(a):
    species(a); s, v, fpg = fastest(a.k, a.eps, a.ts, br=not a.nobr)
    A = a.amp; N = len(a.eps)
    L = 2.0*np.pi/a.k; Lt = a.ny*L/a.nx1                   # cubic cells, ny = nz cells across
    out = [f"gravity/four_pi_G={fpg:.16g}", f"problem/kmode=1",
           f"mesh/nx1={a.nx1}", f"mesh/nx2={a.ny}", f"mesh/nx3={a.ny}",
           f"mesh/x1min={-L/2:.16g}", f"mesh/x1max={L/2:.16g}",
           f"mesh/x2min={-Lt/2:.16g}", f"mesh/x2max={Lt/2:.16g}",
           f"mesh/x3min={-Lt/2:.16g}", f"mesh/x3max={Lt/2:.16g}",
           f"problem/rho_amp={A:.16g}",
           f"problem/vg_amp={A*abs(v[1]):.16g}", f"problem/vg_phase={np.angle(v[1]):.16g}"]
    for i in range(N):
        out += [f"problem/rhod_amp_{i+1}={A*abs(v[2+2*i]):.16g}",
                f"problem/rhod_phase_{i+1}={np.angle(v[2+2*i]):.16g}",
                f"problem/vd_amp_{i+1}={A*abs(v[3+2*i]):.16g}",
                f"problem/vd_phase_{i+1}={np.angle(v[3+2*i]):.16g}",
                f"problem/eps_{i+1}={a.eps[i]:.16g}", f"dust/taus_{i+1}={a.ts[i]:.16g}"]
    out += [f"dust/nspecies={N}", f"particles/ppc={N}", f"time/tlim={a.tgrow/max(s.real,1e-3):.6g}"]
    print("# sigma =", s.real, " Im(s) =", s.imag, " (gas-only sigma =", np.sqrt(max(1.0/(1+sum(a.eps)) - a.k**2, 0)), ")", file=sys.stderr)
    if a.out:
        apply_overrides(a.input, a.out, out); print(a.out)
    else:
        print(" ".join(out))

def species(a):
    """--eps/--ts lists, or --nspec N: N species, eps0 split evenly, ts spaced linearly (or
    logarithmically with --ts-log) between --ts-min and --ts-max.  Krapp et al. (2024) fig. 6
    says "linear distribution from 1e-4 to 10", but its plotted curve (sigma = 0.2 at the
    gas cutoff, 0.02 at k/kJ = 1.4) is the log-uniform mixture; the linear one has a tail
    three times stronger (0.28, 0.07).  Their sigma also carries a factor sqrt(1 + eps0)."""
    if a.nspec:
        a.eps = [a.eps0/a.nspec]*a.nspec
        a.ts = list(np.logspace(np.log10(a.ts_min), np.log10(a.ts_max), a.nspec)) if a.ts_log \
               else list(np.linspace(a.ts_min, a.ts_max, a.nspec))
    return a

def cmd_curve(a):
    species(a); ks = np.linspace(a.kmin, a.kmax, a.n)
    print("# k/kJ  sigma_dust  sigma_gasonly(no BR)  sigma_gas(eps=0)")
    for k in ks:
        s, _, _ = fastest(k, a.eps, a.ts); s0, _, _ = fastest(k, a.eps, a.ts, br=False)
        g = np.sqrt(max(1.0/(1+sum(a.eps)) - k*k, 0.0))
        print(f"{k:.4f}  {s.real:.6f}  {s0.real:.6f}  {g:.6f}")

def cmd_fit(a):
    import re
    lines = [l for l in open(a.hst) if l.startswith('#') and '[1]=' in l]
    cols = [m.group(1) for m in re.finditer(r'\[\d+\]=(\S+)', lines[0])]
    d = np.loadtxt(a.hst); t = d[:, 0]
    sel = (t >= a.tmin) & (t <= (a.tmax if a.tmax > 0 else t[-1]))
    names = [c[:-2] for c in cols if c.endswith('_c')]
    for nm in names:
        ic = cols.index(nm+'_c'); isn = cols.index(nm+'_s')
        amp = np.sqrt(d[:, ic]**2 + d[:, isn]**2)
        ok = sel & (amp > 0)
        p = np.polyfit(t[ok], np.log(amp[ok]), 1)
        print(f"{nm:10s} sigma = {p[0]:.6f}   (amp {amp[ok][0]:.3e} -> {amp[ok][-1]:.3e} over t {t[ok][0]:.2f}-{t[ok][-1]:.2f})")

p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='cmd', required=True)
def spec_args(q):
    q.add_argument('--eps', type=float, nargs='+', default=[0.01]); q.add_argument('--ts', type=float, nargs='+', default=[1.0])
    q.add_argument('--nspec', type=int, default=0, help='N species: eps0 split evenly, ts linear in [ts-min, ts-max]')
    q.add_argument('--eps0', type=float, default=0.01); q.add_argument('--ts-min', type=float, default=1e-4); q.add_argument('--ts-max', type=float, default=10.0); q.add_argument('--ts-log', action='store_true')
q = sub.add_parser('eig'); q.add_argument('--k', type=float, required=True); spec_args(q); q.add_argument('--amp', type=float, default=1e-4); q.add_argument('--tgrow', type=float, default=6.0, help='run for tgrow e-foldings'); q.add_argument('--nobr', action='store_true')
q.add_argument('--nx1', type=int, default=128, help='cells per wavelength'); q.add_argument('--ny', type=int, default=32, help='transverse cells (cubic cells)')
q.add_argument('--input', default=None, help='base athinput'); q.add_argument('--out', default=None, help='write the overrides into a copy of --input (prints its path)'); q.set_defaults(f=cmd_eig)
q = sub.add_parser('curve'); spec_args(q); q.add_argument('--kmin', type=float, default=0.05); q.add_argument('--kmax', type=float, default=1.3); q.add_argument('--n', type=int, default=51); q.set_defaults(f=cmd_curve)
q = sub.add_parser('fit'); q.add_argument('hst'); q.add_argument('--tmin', type=float, default=0.0); q.add_argument('--tmax', type=float, default=-1.0); q.set_defaults(f=cmd_fit)
a = p.parse_args(); a.f(a)
