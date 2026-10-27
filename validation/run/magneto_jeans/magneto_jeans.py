"""Magneto-Jeans test suite (ideal MHD + multigrid self-gravity), 2026-10-26.

Runs inputs/tests/magneto_jeans.athinput over field angle, branch, lambda/lambda_J and
resolution, fits the growth rate or frequency of the projected density amplitude a_sin(t)
from the user history, and compares with the dispersion relation
    (w^2 - W_J^2)(w^2 - k^2 vA^2 cos^2 th) = k^2 vA^2 sin^2 th w^2,  W_J^2 = k^2 cs^2 - 4 pi G rho0
(slow = smaller root, fast = larger; at th = 0 and 90 the single compressive root).
Units: cs = rho0 = 1, lambda = 1 (k = 2 pi), so n = lambda/lambda_J sets 4 pi G = k^2 n^2.

Usage:  python3 magneto_jeans.py run [NJOBS]  (ATHENA=<binary>)  -> runs/<case>/ + results.json
        python3 magneto_jeans.py fit                         -> results.json from runs/
"""
import json, os, subprocess, sys
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
INPUT = os.path.join(REPO, 'inputs', 'tests', 'magneto_jeans.athinput')
ATHENA = os.environ.get('ATHENA', os.path.join(REPO, 'build', 'src', 'athena'))
K = 2*np.pi


def omega2(n, vA, th_deg, mode):
    """Analytic w^2 (cs = rho0 = 1, k = 2 pi)."""
    th = np.radians(th_deg)
    wj2 = K**2*(1 - n**2)
    if abs(np.sin(th)) < 1e-12:
        return wj2
    if abs(np.cos(th)) < 1e-12:
        return wj2 + K**2*vA**2
    s = wj2 + K**2*vA**2
    p = wj2*K**2*vA**2*np.cos(th)**2
    d = np.sqrt(max(s*s - 4*p, 0.0))
    return 0.5*(s + d) if mode == 'fast' else 0.5*(s - d)


def cases():
    out = []
    # dispersion scan at vA = cs, 64 cells per wavelength
    for th in (0, 30, 60, 90):
        modes = ('slow', 'fast') if th in (30, 60) else ('fast',)
        for mode in modes:
            for n in (0.5, 0.8, 1.2, 1.5, 2.0):
                out.append(dict(th=th, mode=mode, n=n, vA=1.0, nx=64, dir='x1'))
    # field strength at th = 90: the field stabilizes lambda/lambda_J = 1.5 once vA > 1.12
    for vA in (0.5, 1.5, 2.0):
        out.append(dict(th=90, mode='fast', n=1.5, vA=vA, nx=64, dir='x1'))
    # resolution: an unstable slow and a stable fast mode at th = 60
    for nx in (16, 32, 128):
        out.append(dict(th=60, mode='slow', n=2.0, vA=1.0, nx=nx, dir='x1'))
        out.append(dict(th=60, mode='fast', n=0.5, vA=1.0, nx=nx, dir='x1'))
    # 3-D oblique wave (box diagonal), multi-dimensional CT + multigrid
    for (th, mode, n) in ((45, 'slow', 2.0), (45, 'fast', 0.8), (90, 'fast', 1.5)):
        out.append(dict(th=th, mode=mode, n=n, vA=1.0, nx=64, dir='oblique'))
    for c in out:
        c['name'] = (f"{c['dir']}_th{c['th']:02d}_{c['mode']}_n{c['n']:.1f}"
                     f"_vA{c['vA']:.1f}_nx{c['nx']}")
    return out


def run(c):
    # rates scale with k: the diagonal wave of the 1 x 1/2 x 1/2 box has k = 3 x 2 pi
    # (lambda = 1/3), so its rates are 3x those of omega2() at the same lambda/lambda_J
    scale = 3.0 if c['dir'] == 'oblique' else 1.0
    w2 = omega2(c['n'], c['vA'], c['th'], c['mode'])
    w = np.sqrt(abs(w2))*scale
    # 3 e-folds when growing; 2 periods when oscillating (at least t = 0.05); a stable fast
    # mode whose slow branch is unstable is cut at 6 slow e-folds, before the growing
    # slow mode seeded by the discretization error turns nonlinear
    tlim = 3.0/w if w2 < 0 else 2*2*np.pi/max(w, 1e-3)
    ws2 = omega2(c['n'], c['vA'], c['th'], 'slow')
    if c['mode'] == 'fast' and w2 > 0 and ws2 < 0 and c['th'] not in (0, 90):
        tlim = min(tlim, 6.0/(np.sqrt(-ws2)*scale))
    tlim = float(min(max(tlim, 0.05), 3.0))
    d = os.path.join(HERE, 'runs', c['name'])
    os.makedirs(d, exist_ok=True)
    ovr = [f"problem/n_jeans={c['n']}", f"problem/b0={c['vA']}", f"problem/b_angle={c['th']}",
           f"problem/mode={c['mode']}", f"problem/wave_dir={c['dir']}", f"time/tlim={tlim}",
           f"output1/dt={tlim/200}"]
    if c['dir'] == 'x1':
        nx = c['nx']
        ovr += [f"mesh/nx1={nx}", "mesh/nx2=8", "mesh/nx3=8", f"mesh/x2max={8/nx}",
                f"mesh/x3max={8/nx}"]
    else:     # the original diagonal box: 1 x 1/2 x 1/2, nx x nx/2 x nx/2
        nx = c['nx']
        ovr += [f"mesh/nx1={nx}", f"mesh/nx2={nx//2}", f"mesh/nx3={nx//2}", "mesh/x2max=0.5",
                "mesh/x3max=0.5"]
    with open(os.path.join(d, 'run.log'), 'w') as f:
        subprocess.run([ATHENA, '-i', INPUT, '-d', d] + ovr, stdout=f, stderr=subprocess.STDOUT,
                       check=False)


def _run_named(c):
    run(c)
    return c['name']


def fit(c, d=None):
    d = d or os.path.join(HERE, 'runs', c['name'])
    h = np.loadtxt(os.path.join(d, 'MagJeans.user.hst'))
    t, a, bp = h[:, 0], h[:, 2], h[:, 4]
    w2 = omega2(c['n'], c['vA'], c['th'], c['mode'])
    if c['dir'] == 'oblique':
        # the diagonal wave's wavelength is set by the box; recompute k from the run log
        log = open(os.path.join(d, 'run.log')).read()
        kw = float(log.split('k_wave = ')[1].split()[0].rstrip(','))
        kj = float(log.split('k_Jeans = ')[1].split()[0].rstrip(','))
        w2 = omega2_k(kw, kj, c['vA'], c['th'], c['mode'])
    else:
        kw, kj = K, K*c['n']
    c['kw'], c['kj'] = float(kw), float(kj)
    a0 = a[0]
    if c['dir'] == 'oblique':
        ws2 = omega2_k(kw, kj, c['vA'], c['th'], 'slow')
    else:
        ws2 = omega2(c['n'], c['vA'], c['th'], 'slow')
    mixed = c['mode'] == 'fast' and w2 > 0 and ws2 < 0 and c['th'] not in (0, 90)
    if w2 < 0:
        k = t > 0.1*t[-1]
        meas = -np.polyfit(t[k], np.log(np.abs(a[k])), 1)[0]**2      # -s^2
    elif mixed:
        # the fast standing wave plus the slow growing/decaying pair seeded by the
        # discretization error: a/a0 = A cos(w t) + B exp(s t) + C exp(-s t)
        s0 = np.sqrt(-ws2)
        f = lambda p: a/a0 - (p[2]*np.cos(p[0]*t) + p[3]*np.exp(p[1]*t) + p[4]*np.exp(-p[1]*t))
        r = least_squares(f, [np.sqrt(w2), s0, 1.0, 1e-4, 0.0], x_scale=[np.sqrt(w2), s0, 1, 1e-3, 1e-3])
        meas = r.x[0]**2
        c['s_slow_fit'] = float(r.x[1])
        c['s_slow_ana'] = float(s0)
        c['slow_seed'] = float(r.x[3])
    else:
        w0 = np.sqrt(w2)
        r = least_squares(lambda p: a/a0 - np.cos(p[0]*t)*np.exp(-p[1]*t), [w0, 0.0])
        meas = r.x[0]**2
        c['damp'] = float(r.x[1])
    c['w2_ana'] = float(w2)
    c['w2_meas'] = float(meas)
    c['rel_err'] = float(abs(np.sqrt(abs(meas)) - np.sqrt(abs(w2)))/np.sqrt(abs(w2)))
    # dB_perp / drho along the run, against C = B_perp w^2/(rho0 (w^2 - k^2 vA_par^2))
    log = open(os.path.join(d, 'run.log')).read()
    c['C_ana'] = float(log.split('dB_perp/drho = ')[1].split()[0])
    k = np.abs(a) > 0.3*np.abs(a0)
    if mixed:           # before the slow mode has grown
        k &= t < 0.25*2*np.pi/np.sqrt(w2)
    c['C_meas'] = float(np.median(bp[k]/a[k]))
    return c


def omega2_k(kw, kj, vA, th_deg, mode):
    """Analytic w^2 for wavenumber kw and Jeans wavenumber kj (cs = rho0 = 1)."""
    th = np.radians(th_deg)
    wj2 = kw**2 - kj**2
    if abs(np.sin(th)) < 1e-12:
        return wj2
    if abs(np.cos(th)) < 1e-12:
        return wj2 + kw**2*vA**2
    s = wj2 + kw**2*vA**2
    p = wj2*kw**2*vA**2*np.cos(th)**2
    dd = np.sqrt(max(s*s - 4*p, 0.0))
    return 0.5*(s + dd) if mode == 'fast' else 0.5*(s - dd)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'fit'
    cs_ = cases()
    if what == 'run':
        from multiprocessing import Pool
        njobs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        with Pool(njobs) as pool:
            for name in pool.imap_unordered(_run_named, cs_):
                print('done', name, flush=True)
    res = [fit(dict(c)) for c in cs_]
    json.dump(res, open(os.path.join(HERE, 'results.json'), 'w'), indent=1)
    # flat table for the Julia figure (validation/figures/plot_magneto_jeans.jl)
    cols = ['name', 'dir', 'th', 'mode', 'n', 'vA', 'nx', 'kw', 'kj', 'w2_ana', 'w2_meas', 'rel_err',
            'C_ana', 'C_meas']
    with open(os.path.join(HERE, 'results.tsv'), 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for c in res:
            f.write('\t'.join(str(c[k]) for k in cols) + '\n')
    print(f"{'case':44s} {'w2_ana':>11s} {'w2_meas':>11s} {'rel_err':>9s} {'C_ana':>8s} {'C_meas':>8s}")
    for c in res:
        print(f"{c['name']:44s} {c['w2_ana']:11.4f} {c['w2_meas']:11.4f} {c['rel_err']:9.2e} "
              f"{c['C_ana']:8.4f} {c['C_meas']:8.4f}")
