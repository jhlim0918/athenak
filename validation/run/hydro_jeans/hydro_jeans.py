"""Hydrodynamic Jeans test suite (hydro + multigrid self-gravity), 2026-11-03.

The hydro counterpart of ../magneto_jeans/magneto_jeans.py and ../magneto_jeans_amr/:
inputs/tests/hydro_jeans{,_smr,_amr}.athinput over lambda/lambda_J, resolution, EOS and mesh.
The projected density amplitude a_sin(t) is fitted for the growth rate s (lambda > lambda_J)
or the frequency w (lambda < lambda_J) and compared with
    continuum:  w^2 = k^2 cs^2 - 4 pi G rho0
    discrete:   w^2 = k^2 cs^2 - 4 pi G rho0 * (sum_d k_d sin(k_d dx_d)/dx_d) / D
with D = sum_d (2/dx_d)^2 sin^2(k_d dx_d/2) the 7-point Laplacian eigenvalue and
sin(k dx)/dx the centred gradient of SourceTerms::SelfGravity: the solver's gravity is
weaker by ~(k dx)^2/12, which matters only near marginal stability.
Units: cs = rho0 = 1, lambda = 1 (k = 2 pi) along x1, so n = lambda/lambda_J sets
4 pi G = k^2 n^2; the oblique (box-diagonal) wave of the 1 x 1/2 x 1/2 box has lambda = 1/3.

Cases (names in results.tsv):
  disp_*   : n = 0.25 ... 3, 64 cells per wavelength, adiabatic (gamma = 5/3)
  res_*    : n = 0.5 (oscillating) and 2 (growing) at 16, 32, 128 cells per wavelength
  iso_*    : isothermal EOS (hllc), n = 0.5, 0.95, 1.5, 2
  obl_*    : 3-D wave along the box diagonal, 64 x 32 x 32 (21 cells per wavelength)
  mesh_*   : n = 0.5 and 2 on uni64, uni128, smr1, smr2, amr1, amr2 (as the MHD suite);
             3-D: uni64, smr1 (a central block), amr1

Usage:  python3 hydro_jeans.py run [NJOBS]   (ATHENA=<binary>)  -> runs/<case>/ + results
        python3 hydro_jeans.py fit                              -> results from runs/
"""
import json, os, re, subprocess, sys
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
INP = {k: os.path.join(REPO, 'inputs', 'tests', f'hydro_jeans{s}.athinput')
       for k, s in (('uni', ''), ('smr', '_smr'), ('amr', '_amr'))}
ATHENA = os.environ.get('ATHENA', os.path.join(REPO, 'build', 'src', 'athena'))
K = 2*np.pi


def cases():
    out = []
    for n in (0.25, 0.5, 0.75, 0.9, 0.95, 1.05, 1.1, 1.25, 1.5, 2.0, 3.0):
        out.append(dict(tag='disp', n=n, nx=64, dir='x1', eos='ideal', mesh='uni'))
    for n in (0.5, 2.0):
        for nx in (16, 32, 128):
            out.append(dict(tag='res', n=n, nx=nx, dir='x1', eos='ideal', mesh='uni'))
    for n in (0.5, 0.95, 1.5, 2.0):
        out.append(dict(tag='iso', n=n, nx=64, dir='x1', eos='isothermal', mesh='uni'))
    for n in (0.5, 2.0):
        out.append(dict(tag='obl', n=n, nx=64, dir='oblique', eos='ideal', mesh='uni'))
    for n in (0.5, 2.0):
        for mesh in ('smr1', 'smr2', 'amr1', 'amr2'):
            out.append(dict(tag='mesh', n=n, nx=64, dir='x1', eos='ideal', mesh=mesh))
        for mesh in ('smr1', 'amr1'):
            out.append(dict(tag='mesh', n=n, nx=64, dir='oblique', eos='ideal', mesh=mesh))
    for c in out:
        c['name'] = f"{c['tag']}_{c['dir']}_n{c['n']:.2f}_nx{c['nx']}_{c['eos'][:3]}_{c['mesh']}"
    return out


def geometry(c):
    """Wave vector components and cell sizes (root grid) of a case."""
    nx = c['nx']
    if c['dir'] == 'x1':
        return np.array([K, 0.0, 0.0]), np.array([1.0/nx, 1.0/nx, 1.0/nx])
    # the diagonal wave of the 1 x 1/2 x 1/2 box: one wavelength per box length along each axis
    return 2*np.pi*np.array([1.0, 2.0, 2.0]), np.array([1.0/nx, 0.5/(nx//2), 0.5/(nx//2)])


def theory(c):
    """(w^2 continuum, w^2 discrete, k^2) with cs = rho0 = 1 and k_J = n k."""
    kv, dx = geometry(c)
    k2 = float(np.sum(kv**2))
    g4 = c['n']**2*k2                                   # 4 pi G rho0 = k_J^2 cs^2
    D = np.sum((2/dx)**2*np.sin(kv*dx/2)**2)
    kg = np.sum(kv*np.sin(kv*dx)/dx)
    return k2 - g4, k2 - g4*kg/D, k2


def tlim_of(c):
    w2, _, _ = theory(c)
    w = np.sqrt(abs(w2))
    t = 3.0/w if w2 < 0 else 2*2*np.pi/w            # 3 e-folds, or 2 periods
    return float(min(max(t, 0.05), 8.0))


def input_text(c):
    kind = c['mesh'][:3]
    s = open(INP[kind]).read()
    if c['mesh'] == 'smr2':
        s = s.replace('num_levels = 2', 'num_levels = 3')
        s += ('\n<refined_region2>\nlevel     = 2\nx1min     = 0.3125\nx1max     = 0.4375\n'
              'x2min     = 0.0\nx2max     = 0.125\nx3min     = 0.0\nx3max     = 0.125\n')
    if c['mesh'] == 'amr2':
        s = s.replace('num_levels = 2', 'num_levels = 3')
    if c['dir'] == 'oblique' and kind == 'smr':
        # a central block of the 1 x 1/2 x 1/2 box: interfaces oblique to k on all sides
        for key, old, new in (('x1max', '0.5', '0.75'), ('x2min', '0.0', '0.125'),
                              ('x2max', '0.125', '0.375'), ('x3min', '0.0', '0.125'),
                              ('x3max', '0.125', '0.375')):
            s = re.sub(rf'(<refined_region1>.*?{key}\s*=\s*){re.escape(old)}\b', rf'\g<1>{new}',
                       s, flags=re.S)
    return s


def run(c):
    tlim = tlim_of(c)
    d = os.path.join(HERE, 'runs', c['name'])
    os.makedirs(d, exist_ok=True)
    inp = os.path.join(d, 'input.athinput')
    open(inp, 'w').write(input_text(c))
    nx = c['nx']
    ovr = [f"problem/n_jeans={c['n']}", f"problem/wave_dir={c['dir']}", f"time/tlim={tlim}",
           f"output1/dt={tlim/200}", f"hydro/eos={c['eos']}"]
    if c['dir'] == 'x1':
        ovr += [f"mesh/nx1={nx}", "mesh/nx2=8", "mesh/nx3=8", f"mesh/x2max={8/nx}",
                f"mesh/x3max={8/nx}"]
    else:
        ovr += [f"mesh/nx1={nx}", f"mesh/nx2={nx//2}", f"mesh/nx3={nx//2}", "mesh/x2max=0.5",
                "mesh/x3max=0.5"]
    if c['mesh'].startswith('amr'):
        ovr += [f"problem/win_speed={2.0/tlim}"]       # two crossings of the unit box
    with open(os.path.join(d, 'run.log'), 'w') as f:
        subprocess.run([ATHENA, '-i', inp, '-d', d] + ovr, stdout=f, stderr=subprocess.STDOUT,
                       check=False)


def _run_named(c):
    run(c)
    return c['name']


def fit(c):
    d = os.path.join(HERE, 'runs', c['name'])
    h = np.loadtxt(os.path.join(d, 'HydroJeans.user.hst'))
    t, a, ac, nmb = h[:, 0], h[:, 2], h[:, 3], h[:, 9]
    w2c, w2d, k2 = theory(c)
    a0 = a[0]
    if w2c < 0:
        k = t > 0.1*t[-1]
        meas = -np.polyfit(t[k], np.log(np.abs(a[k])), 1)[0]**2       # -s^2
        c['damp'] = 0.0
    else:
        w0 = np.sqrt(w2c)
        r = least_squares(lambda p: a/a0 - np.cos(p[0]*t)*np.exp(-p[1]*t), [w0, 0.0])
        meas = r.x[0]**2
        c['damp'] = float(r.x[1])
    c['k2'] = float(k2)
    c['w2_cont'], c['w2_disc'], c['w2_meas'] = float(w2c), float(w2d), float(meas)
    rt = lambda x: np.sqrt(abs(x))
    c['err_cont'] = float(abs(rt(meas) - rt(w2c))/rt(w2c))
    c['err_disc'] = float(abs(rt(meas) - rt(w2d))/rt(w2d))
    c['dw2_k2'] = float(abs(meas - w2c)/k2)                   # |dw^2|/k^2 (marginal-safe)
    c['acos_rel_max'] = float(np.max(np.abs(ac))/np.max(np.abs(a)))
    c['nmb_min'], c['nmb_max'] = int(nmb.min()), int(nmb.max())
    log = open(os.path.join(d, 'run.log')).read()
    m = re.search(r'(\d+) MeshBlocks created, (\d+) deleted', log)
    c['created'], c['deleted'] = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    c['tlim'] = tlim_of(c)
    return c


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'fit'
    cs_ = cases()
    if what == 'run':
        from multiprocessing import Pool
        njobs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        cost = lambda c: (tlim_of(c)*(c['nx']/64)**2*(16 if c['dir'] == 'oblique' else 1)
                          * (4 if c['mesh'][-1] == '2' else 1))
        with Pool(njobs) as pool:
            for name in pool.imap_unordered(_run_named, sorted(cs_, key=lambda c: -cost(c))):
                print('done', name, flush=True)
    res = [fit(dict(c)) for c in cs_]
    json.dump(res, open(os.path.join(HERE, 'results.json'), 'w'), indent=1)
    cols = ['name', 'tag', 'dir', 'n', 'nx', 'eos', 'mesh', 'k2', 'w2_cont', 'w2_disc', 'w2_meas',
            'err_cont', 'err_disc', 'dw2_k2', 'damp', 'acos_rel_max', 'nmb_min', 'nmb_max',
            'created', 'deleted', 'tlim']
    with open(os.path.join(HERE, 'results.tsv'), 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for c in res:
            f.write('\t'.join(str(c[k]) for k in cols) + '\n')
    print(f"{'case':38s} {'w2_cont':>10s} {'w2_disc':>10s} {'w2_meas':>10s} {'err_c':>8s} "
          f"{'err_d':>8s} {'dw2/k2':>8s} {'damp':>8s} {'acos':>8s} {'nmb':>8s} {'cr/del':>9s}")
    for c in res:
        print(f"{c['name']:38s} {c['w2_cont']:10.3f} {c['w2_disc']:10.3f} {c['w2_meas']:10.3f} "
              f"{c['err_cont']:8.1e} {c['err_disc']:8.1e} {c['dw2_k2']:8.1e} {c['damp']:8.1e} "
              f"{c['acos_rel_max']:8.1e} {c['nmb_min']:3d}-{c['nmb_max']:<4d} "
              f"{c['created']:4d}/{c['deleted']:<4d}")
