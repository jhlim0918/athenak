"""Minimal reproducer of the refined-mesh field error (magnetized swing test), 2026-10-27.

inputs/tests/swing_mhd_repro.athinput: thin 3-D MHD shearing box without orbital advection
or gravity, B0 = 1 along y, a leading shearing wave, a static level-1 ring |x| < pi/4. Each
variant is run as a pair -- the ring and the same input with refinement off -- and the
refined run is scored by its deviation from the uniform twin over t <= 6:
    dev(X) = max_t |X_ring - X_uniform| / max_t |X_uniform|,  X = d_cos (density), by_cos (c_y)
A correct refinement gives a few per cent (B0 = 0: ~2%); the defect gives ~20% in c_y.

Usage:  python3 repro.py [NJOBS] [variant ...]    (ATHENA=<binary>; default: all variants)
"""
import os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
ATHENA = os.environ.get('ATHENA', os.path.join(REPO, 'build', 'src', 'athena'))
INPUT = os.path.join(REPO, 'inputs', 'tests', 'swing_mhd_repro.athinput')

# variant name -> command-line overrides (applied to both members of the pair)
VARIANTS = {
    'base':       [],
    'b0':         ['problem/b0=0.0'],
    'b0.5':       ['problem/b0=0.5'],
    'llf':        ['mhd/rsolver=llf'],
    'dc':         ['mhd/reconstruct=dc'],
    'ppm4':       ['mhd/reconstruct=ppm4'],
    'rk3':        ['time/integrator=rk3'],
    'cfl0.1':     ['time/cfl_number=0.1'],
    'q0.5':       ['shearing_box/qshear=0.5'],
    'q0':         ['shearing_box/qshear=0.0'],
    # plane-parallel shear flow, (almost) no rotation: q Omega = 1.5 kept, Coriolis and tidal
    # terms 150x weaker (vy = -1.5 x is an equilibrium without them)
    'shear_only': ['shearing_box/omega0=0.01', 'shearing_box/qshear=150.0'],
    'shear_only_b0': ['shearing_box/omega0=0.01', 'shearing_box/qshear=150.0', 'problem/b0=0.0'],
    # wave amplitude: an error from the refined background equilibrium does not scale with it
    'amp1e-3':    ['problem/amp=1.0e-3'],
    'amp1e-5':    ['problem/amp=1.0e-5'],
    'amp0':       ['problem/amp=0.0'],
    # energy prolongation: primitives instead of conserved variables; no energy at all
    'prim':       ['mesh_refinement/prolong_primitives=true'],
    'iso_hlle':   ['mhd/eos=isothermal', 'mhd/rsolver=hlle'],
    'hlle':       ['mhd/rsolver=hlle'],
}


def run(args):
    name, ovr, mesh = args
    d = os.path.join(HERE, 'runs', f'{name}_{mesh}')
    os.makedirs(d, exist_ok=True)
    extra = [] if mesh == 'ring' else ['mesh_refinement/refinement=none']
    with open(os.path.join(d, 'run.log'), 'w') as f:
        subprocess.run([ATHENA, '-i', INPUT, '-d', d] + ovr + extra, stdout=f,
                       stderr=subprocess.STDOUT, check=False)
    return f'{name}_{mesh}'


def score(name):
    h = [np.loadtxt(os.path.join(HERE, 'runs', f'{name}_{m}', 'SwingMHD.user.hst'))
         for m in ('ring', 'uni')]
    n = min(len(h[0]), len(h[1]))
    out = {}
    for col, nm in ((2, 'delta'), (8, 'cy')):
        ref = np.max(np.abs(h[1][:n, col]))
        out[nm] = (np.max(np.abs(h[0][:n, col] - h[1][:n, col]))/ref) if ref > 0 else np.nan
    out['divb'] = float(np.max(h[0][1:n, 9]/np.maximum(h[0][1:n, 10], 1e-300)))
    return out


if __name__ == '__main__':
    njobs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    names = sys.argv[2:] or list(VARIANTS)
    jobs = [(n, VARIANTS[n], m) for n in names for m in ('ring', 'uni')]
    from multiprocessing import Pool
    with Pool(njobs) as p:
        for j in p.imap_unordered(run, jobs):
            print('done', j, flush=True)
    print(f"{'variant':10s} {'dev_delta':>10s} {'dev_cy':>10s} {'divB/B':>9s}")
    for n in names:
        s = score(n)
        print(f"{n:10s} {s['delta']:10.3f} {s['cy']:10.3f} {s['divb']:9.1e}")
