"""Magneto-Jeans test on refined meshes (ideal MHD + multigrid self-gravity), 2026-10-26.

The uniform-grid suite (../magneto_jeans/magneto_jeans.py) checks the dispersion relation
    (w^2 - W_J^2)(w^2 - k^2 vA^2 cos^2 th) = k^2 vA^2 sin^2 th w^2,  W_J^2 = k^2 cs^2 - 4 pi G rho0
on a uniform grid.  Here the same eigenmodes run on
    uni64, uni128 : uniform, 64 / 128 cells per wavelength (the bracket)
    smr1          : 64 at the root, a static slab x1 in [0.25, 0.5) at level 1 (128)
    smr2          : as smr1 plus x1 in [0.3125, 0.4375) at level 2 (256)
    amr1          : 64 at the root, a moving slab of half-width 0.125 at level 1 (128),
                    swept through the box twice per run (<problem>/amr_mode = window)
    amr2          : the same slab refined to level 2 (2:1 balance adds a level-1 rim)
for four quasi-1-D modes, and the 3-D wave along the box diagonal on uni64 / smr1 (a
central block) / amr1.  Measured: the frequency or growth rate (fit as in the uniform
suite), dB_perp/drho, the relative divergence divb/babs, the spurious cos(k xi) density
component |a_cos|/|a_sin| (reflection and phase error at the level interfaces), and the
MeshBlocks created/deleted.

Usage:  python3 magneto_jeans_amr.py run [NJOBS]  (ATHENA=<binary>; NJOBS serial runs at once)
                                                      -> runs/<case>/ + results.{json,tsv}
        python3 magneto_jeans_amr.py fit                      -> results from runs/
"""
import json, os, re, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'magneto_jeans'))
from magneto_jeans import omega2, fit as fit_uniform   # noqa: E402

ATHENA = os.environ.get('ATHENA', os.path.join(REPO, 'build', 'src', 'athena'))
INP = {'uni': os.path.join(REPO, 'inputs', 'tests', 'magneto_jeans.athinput'),
       'smr': os.path.join(REPO, 'inputs', 'tests', 'magneto_jeans_smr.athinput'),
       'amr': os.path.join(REPO, 'inputs', 'tests', 'magneto_jeans_amr.athinput')}

# quasi-1-D modes: (theta, branch, lambda/lambda_J, vA)
MODES_1D = [(60, 'slow', 2.0, 1.0),    # growing slow mode
            (60, 'fast', 0.5, 1.0),    # oscillating fast mode
            (90, 'fast', 1.5, 1.5),    # hydro-unstable wavelength held up by the field
            (30, 'fast', 2.0, 1.0)]    # fast mode carried by the field (C = 7.6); slow unstable
MESH_1D = ['uni64', 'uni128', 'smr1', 'smr2', 'amr1', 'amr2']
MODES_3D = [(45, 'slow', 2.0, 1.0), (45, 'fast', 0.8, 1.0)]
MESH_3D = ['uni64', 'smr1', 'amr1']


def cases():
    out = []
    for (th, mode, n, vA) in MODES_1D:
        for mesh in MESH_1D:
            out.append(dict(th=th, mode=mode, n=n, vA=vA, dir='x1', mesh=mesh,
                            nx=128 if mesh == 'uni128' else 64))
    for (th, mode, n, vA) in MODES_3D:
        for mesh in MESH_3D:
            out.append(dict(th=th, mode=mode, n=n, vA=vA, dir='oblique', mesh=mesh, nx=64))
    for c in out:
        c['name'] = f"{c['dir']}_th{c['th']:02d}_{c['mode']}_n{c['n']:.1f}_vA{c['vA']:.1f}_{c['mesh']}"
    return out


def tlim_of(c):
    """Run length as in the uniform suite: 3 e-folds if growing, 2 periods if not, a stable
    fast mode with an unstable slow branch cut at 6 slow e-folds."""
    if c['dir'] == 'x1':
        w2 = omega2(c['n'], c['vA'], c['th'], c['mode'])
        ws2 = omega2(c['n'], c['vA'], c['th'], 'slow')
        scale = 1.0
    else:   # the diagonal wave has k = 3 x 2 pi (lambda = 1/3): rates scale by 3
        w2 = omega2(c['n'], c['vA'], c['th'], c['mode'])
        ws2 = omega2(c['n'], c['vA'], c['th'], 'slow')
        scale = 3.0
    w = np.sqrt(abs(w2))*scale
    t = 3.0/w if w2 < 0 else 2*2*np.pi/max(w, 1e-3)
    if c['mode'] == 'fast' and w2 > 0 and ws2 < 0 and c['th'] not in (0, 90):
        t = min(t, 6.0/(np.sqrt(-ws2)*scale))
    return float(min(max(t, 0.05), 3.0))


def input_text(c):
    """The input file of a case: the uniform/SMR/AMR base, patched where an override cannot
    add a block (a second refined region; the 3-D regions)."""
    kind = c['mesh'][:3]
    s = open(INP['uni' if kind == 'uni' else kind]).read()
    if c['mesh'] == 'smr2':
        s = s.replace('num_levels = 2', 'num_levels = 3')
        s += ('\n<refined_region2>\nlevel     = 2\nx1min     = 0.3125\nx1max     = 0.4375\n'
              'x2min     = 0.0\nx2max     = 0.125\nx3min     = 0.0\nx3max     = 0.125\n')
    if c['mesh'] == 'amr2':
        s = s.replace('num_levels = 2', 'num_levels = 3')
    if c['dir'] == 'oblique' and kind == 'smr':
        # a central block of the 1 x 1/2 x 1/2 box: interfaces oblique to k on all sides
        s = re.sub(r'(<refined_region1>.*?x1min\s*=\s*)0\.25', r'\g<1>0.25', s, flags=re.S)
        s = re.sub(r'(<refined_region1>.*?x1max\s*=\s*)0\.5\b', r'\g<1>0.75', s, flags=re.S)
        s = re.sub(r'(<refined_region1>.*?x2min\s*=\s*)0\.0', r'\g<1>0.125', s, flags=re.S)
        s = re.sub(r'(<refined_region1>.*?x2max\s*=\s*)0\.125', r'\g<1>0.375', s, flags=re.S)
        s = re.sub(r'(<refined_region1>.*?x3min\s*=\s*)0\.0', r'\g<1>0.125', s, flags=re.S)
        s = re.sub(r'(<refined_region1>.*?x3max\s*=\s*)0\.125', r'\g<1>0.375', s, flags=re.S)
    return s


def run(c):
    tlim = tlim_of(c)
    d = os.path.join(HERE, 'runs', c['name'])
    os.makedirs(d, exist_ok=True)
    inp = os.path.join(d, 'input.athinput')
    open(inp, 'w').write(input_text(c))
    ovr = [f"problem/n_jeans={c['n']}", f"problem/b0={c['vA']}", f"problem/b_angle={c['th']}",
           f"problem/mode={c['mode']}", f"problem/wave_dir={c['dir']}", f"time/tlim={tlim}",
           f"output1/dt={tlim/200}"]
    nx = c['nx']
    if c['dir'] == 'x1':
        ovr += [f"mesh/nx1={nx}", "mesh/nx2=8", "mesh/nx3=8", f"mesh/x2max={8/nx}",
                f"mesh/x3max={8/nx}"]
    else:
        ovr += [f"mesh/nx1={nx}", f"mesh/nx2={nx//2}", f"mesh/nx3={nx//2}", "mesh/x2max=0.5",
                "mesh/x3max=0.5"]
    if c['mesh'].startswith('amr'):
        ovr += [f"problem/win_speed={2.0/tlim}"]   # two crossings of the unit box
    with open(os.path.join(d, 'run.log'), 'w') as f:
        subprocess.run([ATHENA, '-i', inp, '-d', d] + ovr, stdout=f, stderr=subprocess.STDOUT,
                       check=False)


def _run_named(c):
    run(c)
    return c['name']


def fit(c):
    d = os.path.join(HERE, 'runs', c['name'])
    c = fit_uniform(c, d)
    h = np.loadtxt(os.path.join(d, 'MagJeans.user.hst'))
    a, ac, divb, babs, nmb = h[:, 2], h[:, 3], h[:, 7], h[:, 8], h[:, 9]
    c['divb_rel_max'] = float(np.max(divb/babs))
    c['acos_rel_max'] = float(np.max(np.abs(ac))/np.max(np.abs(a)))
    c['nmb_min'], c['nmb_max'] = int(nmb.min()), int(nmb.max())
    log = open(os.path.join(d, 'run.log')).read()
    m = re.search(r'(\d+) MeshBlocks created, (\d+) deleted', log)
    c['created'], c['deleted'] = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    return c


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'fit'
    cs_ = cases()
    if what == 'run':
        from multiprocessing import Pool
        njobs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        # longest first, so the pool is not left waiting on one slow case at the end
        order = sorted(cs_, key=lambda c: -tlim_of(c)*(4 if c['mesh'][-1] == '2' else 1))
        with Pool(njobs) as pool:
            for name in pool.imap_unordered(_run_named, order):
                print('done', name, flush=True)
    res = [fit(dict(c)) for c in cs_]
    json.dump(res, open(os.path.join(HERE, 'results.json'), 'w'), indent=1)
    cols = ['name', 'dir', 'th', 'mode', 'n', 'vA', 'mesh', 'kw', 'kj', 'w2_ana', 'w2_meas',
            'rel_err', 'C_ana', 'C_meas', 'divb_rel_max', 'acos_rel_max', 'nmb_min', 'nmb_max',
            'created', 'deleted']
    with open(os.path.join(HERE, 'results.tsv'), 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for c in res:
            f.write('\t'.join(str(c[k]) for k in cols) + '\n')
    print(f"{'case':42s} {'w2_ana':>10s} {'w2_meas':>10s} {'rel_err':>9s} {'C_ana':>7s} "
          f"{'C_meas':>7s} {'divB/B':>8s} {'acos':>8s} {'nmb':>8s} {'cr/del':>9s}")
    for c in res:
        print(f"{c['name']:42s} {c['w2_ana']:10.3f} {c['w2_meas']:10.3f} {c['rel_err']:9.2e} "
              f"{c['C_ana']:7.4f} {c['C_meas']:7.4f} {c['divb_rel_max']:8.1e} "
              f"{c['acos_rel_max']:8.1e} {c['nmb_min']:3d}-{c['nmb_max']:<4d} "
              f"{c['created']:4d}/{c['deleted']:<4d}")
