"""Minimal reproducer of the refined-mesh field error of the magnetized swing test, 2026-10-27.

inputs/tests/linear_wave_mhd_tflow_smr.athinput: a 2-D MHD linear wave with k oblique to the
grid (k ~ (1, 2)/sqrt(5)) crossing a static level-1 slab x1 in [0.5, 1); background flow V
and field B = 1 along x2 (TANGENTIAL to the slab faces), or the flow along x1 (normal) as a
contrast. No shearing box, no gravity. After two wave periods the exact solution is the
initial state, and the pgen writes volume-weighted L1 errors (LinWave-errs.dat).
For each wave (4 slow, 5 Alfven, 6 fast) and flow, the slab run is compared with uniform
runs at the root (64 cells per unit length) and the refined resolution (128): a correct
interface puts the slab between them.

Usage:  python3 tflow_smr.py run [NJOBS]  (ATHENA=<binary>)  -> runs/ + results.tsv
        python3 tflow_smr.py fit
"""
import math, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
ATHENA = os.environ.get('ATHENA', os.path.join(REPO, 'build', 'src', 'athena'))
INPUT = os.path.join(REPO, 'inputs', 'tests', 'linear_wave_mhd_tflow_smr.athinput')
SA, CA = math.sin(math.atan(2.0)), math.cos(math.atan(2.0))
WAVES = {4: 'slow', 5: 'alfven', 6: 'fast'}
FLOWS = [('t0', 0.0, 'tan'), ('t1', 1.0, 'tan'), ('t2', 2.0, 'tan'), ('n2', 2.0, 'nor')]
MESHES = ['uni64', 'uni128', 'smr']
B0 = 1.0


def cases():
    return [dict(name=f"{WAVES[w]}_{f[0]}_{m}", wave=w, flow=f[0], V=f[1], dir=f[2], mesh=m)
            for w in WAVES for f in FLOWS for m in MESHES]


def run(c):
    d = os.path.join(HERE, 'runs', c['name'])
    os.makedirs(d, exist_ok=True)
    V = c['V']
    if c['dir'] == 'tan':      # grid flow (0, V): wave-frame (V sin a3, V cos a3)
        vx, vy = V*SA, V*CA
    else:                      # grid flow (V, 0): wave-frame (V cos a3, -V sin a3)
        vx, vy = V*CA, -V*SA
    ovr = [f"problem/wave_flag={c['wave']}", f"problem/vx0={vx}", f"problem/vy0={vy}",
           f"problem/bx0={B0*SA}", f"problem/by0={B0*CA}"]
    if c['mesh'] != 'smr':
        ovr += ["mesh_refinement/refinement=none"]
    if c['mesh'] == 'uni128':
        ovr += ["mesh/nx1=256", "mesh/nx2=128"]
    with open(os.path.join(d, 'run.log'), 'w') as f:
        subprocess.run([ATHENA, '-i', INPUT, '-d', d] + ovr, stdout=f,
                       stderr=subprocess.STDOUT, check=False)
    return c['name']


def errs(c):
    d = np.loadtxt(os.path.join(HERE, 'runs', c['name'], 'LinWave-errs.dat'), ndmin=2)[-1]
    # Nx1 Nx2 Nx3 Ncycle RMS-L1 Linf d M1 M2 M3 E B1 B2 B3
    return dict(rms=d[4], d=d[6], M1=d[7], M2=d[8], E=d[10], B1=d[11], B2=d[12])


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'fit'
    cs_ = cases()
    if what == 'run':
        from multiprocessing import Pool
        with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 1) as p:
            for n in p.imap_unordered(run, cs_):
                print('done', n, flush=True)
    E = {c['name']: errs(c) for c in cs_}
    rows = []
    print(f"{'wave':7s} {'flow':5s} {'var':4s} {'uni64':>10s} {'uni128':>10s} {'smr':>10s} "
          f"{'smr/u64':>8s} {'smr/u128':>8s}")
    for w in WAVES.values():
        for f in FLOWS:
            for var in ('rms', 'B1', 'B2'):
                u64, u128, s = (E[f"{w}_{f[0]}_{m}"][var] for m in MESHES)
                rows.append((w, f[0], var, u64, u128, s))
                print(f"{w:7s} {f[0]:5s} {var:4s} {u64:10.3e} {u128:10.3e} {s:10.3e} "
                      f"{s/u64:8.2f} {s/u128:8.2f}")
    with open(os.path.join(HERE, 'results.tsv'), 'w') as fo:
        fo.write('wave\tflow\tvar\tuni64\tuni128\tsmr\n')
        for r in rows:
            fo.write('\t'.join(str(x) for x in r) + '\n')
