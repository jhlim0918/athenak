"""KOS02 tier 1 analysis: omega^2 of the axisymmetric mode in each run, and the marginal
Q_c for each (s0, beta) from omega^2 against Q^2.

Each run starts from a density perturbation at rest. That projects onto the oscillating
or growing mode and onto a static one (the geostrophic mode of the rotating disk, and with
a field a change of the mass-to-flux ratio), so the surface-density amplitude follows
a(t) = c0 + c1 C(w2, t) + c2 S(w2, t), C, S = cos, sin (w2 > 0) or cosh, sinh (w2 < 0),
fitted over the linear phase (|a/a0| < 300) after the start-up transient. omega^2 depends on the rotation only through
kappa^2 = Q^2 (code units), so omega^2(Q^2) is a line; Q_c is its root.

Usage: python3 fit.py  -> fits.json, qc.txt (printed)
"""
import json, os, math
import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))

def CS(w2, t):
    """C, S with C'' = -w2 C: cos/sin for w2 > 0, cosh/sinh for w2 < 0, 1/t at 0."""
    if w2 > 1e-12:
        w = math.sqrt(w2); return np.cos(w*t), np.sin(w*t)/w
    if w2 < -1e-12:
        s = math.sqrt(-w2); return np.cosh(s*t), np.sinh(s*t)/s
    return np.ones_like(t), t

def fit_run(path, qc):
    """omega^2 from a(t) = c0 + c1 C(w2,t) + c2 S(w2,t) over the linear phase after the
    start-up transient (t > 2/qc): c1, c2 by linear least squares at each w2, w2 by a
    grid search refined with a bounded scalar minimisation."""
    from scipy.optimize import minimize_scalar
    h = np.loadtxt(path)
    t, a = h[:, 0], h[:, 2]
    y = a/a[0]
    m = np.abs(y) < 300.0
    if not m.all():
        m[np.argmax(~m):] = False
    m &= t > 2.0/qc
    t, y = t[m], y[m]
    t0 = t[0]; tt = t - t0
    def resid(w2):
        C, S = CS(w2, tt)
        A = np.vstack([np.ones_like(tt), C, S]).T   # static (geostrophic) offset + mode
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        return np.sqrt(np.mean((A @ c - y)**2))/np.max(np.abs(y))
    grid = np.linspace(-1.0, 1.0, 401)*qc**2
    r = np.array([resid(g) for g in grid])
    k = int(np.argmin(r))
    lo, hi = grid[max(k-1, 0)], grid[min(k+1, len(grid)-1)]
    best = minimize_scalar(resid, bounds=(lo, hi), method='bounded', options={'xatol': 1e-9})
    w2 = float(best.x)
    return dict(kind='unstable' if w2 < 0 else 'stable', omega2=w2, resid=float(best.fun),
                tfit=float(t[-1]), amp_end=float(y[-1]))

def main():
    cases = json.load(open(os.path.join(HERE, 'cases.json')))
    out = []
    for c in cases:
        p = os.path.join(HERE, 'runs', c['name'], 'kos.user.hst')
        if not os.path.exists(p) or not os.path.exists(os.path.join(HERE, 'runs', c['name'], 'done')):
            continue
        r = fit_run(p, c['qc_eq18'])
        r['qc_run'] = math.sqrt(c['Q']**2 - r['omega2']) if c['Q']**2 > r['omega2'] else float('nan')
        r.update(c)
        out.append(r)
        print('%-22s Q=%.4f  %-8s omega^2=%+.5f  Qc(run)=%.4f  resid=%.1e  t_fit=%.1f' %
              (c['name'], c['Q'], r['kind'], r['omega2'], r['qc_run'], r['resid'], r['tfit']))
    json.dump(out, open(os.path.join(HERE, 'fits.json'), 'w'), indent=1)
    # Q_c per (s0, beta)
    print()
    print('%-6s %-5s %10s %10s %8s %6s  %s' % ('s0', 'beta', 'Qc(sim)', 'Qc(eq18)', 'slope', 'ratio', 'Qc per run'))
    rows = []
    groups = {}
    for r in out:
        groups.setdefault((r['s0'], r['beta']), []).append(r)
    for (s0, b), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        if len(rs) < 2:
            continue
        Q2 = np.array([r['Q']**2 for r in rs]); w2 = np.array([r['omega2'] for r in rs])
        slope, icpt = np.polyfit(Q2, w2, 1)
        qc = math.sqrt(-icpt/slope) if -icpt/slope > 0 else float('nan')
        q18 = rs[0]['qc_eq18']
        per = [r['qc_run'] for r in sorted(rs, key=lambda r: r['f'])]
        rows.append(dict(s0=s0, beta=b, qc_sim=qc, qc_eq18=q18, slope=slope, n=len(rs),
                         qc_runs=per))
        print('%-6g %-5s %10.4f %10.4f %8.3f %6.3f  %s' % (s0, b, qc, q18, slope, qc/q18,
              ' '.join('%.4f' % x for x in per)))
    json.dump(rows, open(os.path.join(HERE, 'qc.json'), 'w'), indent=1)

if __name__ == '__main__':
    main()
