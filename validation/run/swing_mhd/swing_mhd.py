"""Magnetized swing amplification (ideal MHD + multigrid self-gravity + shear), 2026-10-26.

inputs/tests/swing_mhd.athinput: a leading shearing wave (kx0 = -6, ky = 1) in a uniform
azimuthal field B0 = b0 y_hat, adiabatic (cs = 1), 4 pi G rho0 = 1.9, Lx = Ly = 2 pi. The
user history projects the solution on the instantaneous wavevector (d_cos, vx_sin, vy_sin,
bx_cos, by_cos) and is compared with the linear system of swing_ode.py, integrated with the
multigrid's 7-point Laplacian eigenvalue at the root resolution.

Cases
    uF_b0.5, uF_b1.0, uF_b0.0  : uniform 128^2 x 16, orbital advection (FARGO)
    uF_b0.5_nograv             : the same, self-gravity off (4 pi G = 0 in the ODE)
    uF_b0.5_256                : uniform 256^2 x 16, FARGO (convergence)
    uN_b0.5, uN_b1.0           : uniform, no orbital advection (the full shear flow)
    smr_b0.5, smr_b1.0         : static level-1 ring over the middle two of eight x1 block
                                 columns (|x| < pi/4), no orbital advection. (The hydro
                                 Phase-2a ring of six columns puts refined cells where the
                                 shear flow is 3.5 cs; without orbital advection that sets a
                                 1.6x smaller step on 6x the cells.)
    amr_b0.5, amr_b1.0         : moving level-1 ring, xc = (pi/4) sin t, half-width pi/16
                                 (the hydro Phase-2c criterion), no orbital advection
MHD with orbital advection cannot refine, so every refined run is non-FARGO and is compared
with its uniform non-FARGO twin as well as with linear theory.

Usage:  python3 swing_mhd.py run [NJOBS] [NRANKS]  (ATHENA=<MPI binary>; AMR cases use 2 ranks)
                                                   -> runs/ + results
        python3 swing_mhd.py fit
"""
import json, os, re, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
from swing_ode import solve   # noqa: E402

ATHENA = os.environ.get('ATHENA', os.path.join(REPO, 'build-mpi', 'src', 'athena'))
INPUT = os.path.join(REPO, 'inputs', 'tests', 'swing_mhd.athinput')
PI = np.pi
SMR = """
<mesh_refinement>
refinement = static
max_nmb_per_rank = 1024

<refined_region1>
level = 1
x1min = -0.78
x1max =  0.78
x2min = -3.14159265358979
x2max =  3.14159265358979
x3min = -0.39269908169872
x3max =  0.39269908169872
"""
AMR = """
<mesh_refinement>
refinement = adaptive
num_levels = 2
refinement_interval = 5
max_nmb_per_rank = 1024

<amr_criterion1>
method = user
"""
AMR_PROBLEM = ("amr_moving_ring = true\nring_xc0     = 0.0\nring_xamp    = 0.78539816339745\n"
               "ring_freq    = 1.0\nring_hwidth  = 0.19634954084936\n")


def cases():
    out = [dict(name='uF_b0.5', b0=0.5, mesh='uni', fargo=True, grav=True, nx=128),
           dict(name='uF_b1.0', b0=1.0, mesh='uni', fargo=True, grav=True, nx=128),
           dict(name='uF_b0.0', b0=0.0, mesh='uni', fargo=True, grav=True, nx=128),
           dict(name='uF_b0.5_nograv', b0=0.5, mesh='uni', fargo=True, grav=False, nx=128),
           dict(name='uF_b0.5_256', b0=0.5, mesh='uni', fargo=True, grav=True, nx=256),
           dict(name='uN_b0.5', b0=0.5, mesh='uni', fargo=False, grav=True, nx=128),
           dict(name='uN_b1.0', b0=1.0, mesh='uni', fargo=False, grav=True, nx=128),
           dict(name='smr_b0.5', b0=0.5, mesh='smr', fargo=False, grav=True, nx=128),
           dict(name='smr_b1.0', b0=1.0, mesh='smr', fargo=False, grav=True, nx=128),
           dict(name='amr_b0.5', b0=0.5, mesh='amr', fargo=False, grav=True, nx=128),
           dict(name='amr_b1.0', b0=1.0, mesh='amr', fargo=False, grav=True, nx=128)]
    # diagnostics: which ingredient carries the refined runs' extra deviation (to t = 8)
    for mesh in ('uni', 'smr', 'amr'):
        tag = 'uN' if mesh == 'uni' else mesh
        out.append(dict(name=f'{tag}_b1.0_nograv', b0=1.0, mesh=mesh, fargo=False, grav=False,
                        nx=128, tlim=8.0))
        out.append(dict(name=f'{tag}_b0.0', b0=0.0, mesh=mesh, fargo=False, grav=True, nx=128,
                        tlim=8.0))
    # resolution of the refined runs' field error: the same pair at a 64-cell root (8^3
    # blocks keep the ring at |x| < pi/4 on two of eight block columns)
    for mesh in ('uni', 'smr'):
        tag = 'uN' if mesh == 'uni' else mesh
        out.append(dict(name=f'{tag}_b1.0_nograv_64', b0=1.0, mesh=mesh, fargo=False,
                        grav=False, nx=64, tlim=8.0))
    return out


def cost(c):
    """rough relative cost, for ordering the pool"""
    f = (c['nx']/128)**3
    if not c['fargo']:
        f *= 5
    return f*{'uni': 1, 'smr': 3, 'amr': 6}[c['mesh']]


def run(c, nranks):
    d = os.path.join(HERE, 'runs', c['name'])
    os.makedirs(d, exist_ok=True)
    s = open(INPUT).read()
    if c['mesh'] == 'smr':
        s += SMR
    elif c['mesh'] == 'amr':
        s += AMR
        s = s.replace('pgen_name = swing\n', 'pgen_name = swing\n' + AMR_PROBLEM)
    inp = os.path.join(d, 'input.athinput')
    open(inp, 'w').write(s)
    ovr = [f"problem/b0={c['b0']}", f"shearing_box/orbital_advection={str(c['fargo']).lower()}",
           f"mhd_srcterms/self_gravity={str(c['grav']).lower()}"]
    if c['nx'] != 128:
        ovr += [f"mesh/nx1={c['nx']}", f"mesh/nx2={c['nx']}"]
    if c['nx'] == 64:
        ovr += ["meshblock/nx1=8", "meshblock/nx2=8", "meshblock/nx3=8"]
    if 'tlim' in c:
        ovr += [f"time/tlim={c['tlim']}"]
    with open(os.path.join(d, 'run.log'), 'w') as f:
        subprocess.run(['mpirun', '--bind-to', 'none', '-np', str(nranks), ATHENA, '-i', inp, '-d', d] + ovr,
                       stdout=f, stderr=subprocess.STDOUT, check=False)


def _run(args):
    c, nranks = args
    # the moving-ring runs do not speed up past 2 ranks (remesh + load balance)
    run(c, 2 if c['mesh'] == 'amr' else nranks)
    return c['name']


def fit(c):
    d = os.path.join(HERE, 'runs', c['name'])
    h = np.loadtxt(os.path.join(d, 'SwingMHD.user.hst'))
    t = h[:, 0]
    amp = 1.0e-4
    dx = 2*PI/c['nx']        # theory at the root resolution (the refined runs' coarse grid)
    s = solve(-6.0, 1.0, c['b0'], 1.9 if c['grav'] else 0.0, dx=dx, dy=dx, tmax=t[-1],
              t_eval=t, amp=amp)
    th = s.y/amp                          # delta, wx, wy, cx, cy per unit amplitude
    np.savetxt(os.path.join(d, 'theory.tsv'), np.column_stack([t, th.T]),
               header='t delta wx wy cx cy (per unit amplitude; swing_ode.py)')
    meas = np.array([h[:, 2], h[:, 4], h[:, 5], h[:, 7], h[:, 8]])/amp
    i = np.argmax(np.abs(th[0]))
    j = np.argmax(np.abs(meas[0]))
    c['peak_theory'], c['t_peak_theory'] = float(th[0][i]), float(t[i])
    c['peak_meas'], c['t_peak_meas'] = float(meas[0][j]), float(t[j])
    c['peak_rel_err'] = float(abs(meas[0][j]/th[0][i] - 1))
    # normalized trajectory deviation of each variable: max|meas - theory| / max|theory|
    for k, nm in enumerate(['delta', 'wx', 'wy', 'cx', 'cy']):
        if np.max(np.abs(th[k])) > 0:
            c['dev_' + nm] = float(np.max(np.abs(meas[k] - th[k]))/np.max(np.abs(th[k])))
        else:
            c['dev_' + nm] = float(np.max(np.abs(meas[k])))
    c['dsin_rel_max'] = float(np.max(np.abs(h[:, 3]))/amp)
    c['divb_rel_max'] = float(np.max(h[:, 9]/h[:, 10])) if c['b0'] > 0 else 0.0
    k8 = t <= 8.0 + 1e-9                  # the hydro swing tests' window
    c['dev8_delta'] = float(np.max(np.abs(meas[0][k8] - th[0][k8]))/np.max(np.abs(th[0][k8])))
    c['dev8_cy'] = (float(np.max(np.abs(meas[4][k8] - th[4][k8]))/np.max(np.abs(th[4][k8])))
                    if np.max(np.abs(th[4][k8])) > 0 else 0.0)
    c['dsin8_max'] = float(np.max(np.abs(h[k8, 3]))/amp)
    c['nmb_min'], c['nmb_max'] = int(h[:, 11].min()), int(h[:, 11].max())
    log = open(os.path.join(d, 'run.log')).read()
    m = re.search(r'(\d+) MeshBlocks created, (\d+) deleted', log)
    c['created'], c['deleted'] = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    m = re.findall(r'cpu time used\s*=\s*([0-9.eE+-]+)', log)
    c['cpu_s'] = float(m[-1]) if m else float('nan')
    return c


def compare_twin(res):
    """refined runs against their uniform non-FARGO twin: max|d - d_twin|/max|d_twin|"""
    by = {c['name']: c for c in res}
    for c in res:
        if c['mesh'] in ('smr', 'amr'):
            twin = (f"uN_b{c['b0']:.1f}" + ('' if c['grav'] else '_nograv') +
                    ('_64' if c['nx'] == 64 else ''))
            h1 = np.loadtxt(os.path.join(HERE, 'runs', c['name'], 'SwingMHD.user.hst'))
            h0 = np.loadtxt(os.path.join(HERE, 'runs', twin, 'SwingMHD.user.hst'))
            n = min(len(h0), len(h1), int(np.searchsorted(h0[:, 0], 8.0 + 1e-9)))
            for k, nm in ((2, 'delta'), (8, 'cy')):
                c['twin_dev_' + nm] = float(np.max(np.abs(h1[:n, k] - h0[:n, k])) /
                                            np.max(np.abs(h0[:n, k])))
    return res


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'fit'
    cs_ = cases()
    if what == 'run':
        from multiprocessing.pool import ThreadPool
        njobs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        nranks = int(sys.argv[3]) if len(sys.argv) > 3 else 2
        order = sorted(cs_, key=lambda c: -cost(c))
        with ThreadPool(njobs) as pool:
            for name in pool.imap_unordered(_run, [(c, nranks) for c in order
                                               if not os.path.isfile(os.path.join(
                                                   HERE, 'runs', c['name'],
                                                   'SwingMHD.user.hst'))]):
                print('done', name, flush=True)
    res = compare_twin([fit(dict(c)) for c in cs_])
    json.dump(res, open(os.path.join(HERE, 'results.json'), 'w'), indent=1)
    cols = ['name', 'b0', 'mesh', 'fargo', 'grav', 'nx', 'peak_theory', 't_peak_theory',
            'peak_meas', 't_peak_meas', 'peak_rel_err', 'dev_delta', 'dev_wx', 'dev_wy',
            'dev_cx', 'dev_cy', 'dev8_delta', 'dev8_cy', 'dsin8_max', 'twin_dev_delta',
            'twin_dev_cy', 'dsin_rel_max', 'divb_rel_max', 'nmb_min', 'nmb_max',
            'created', 'deleted', 'cpu_s']
    with open(os.path.join(HERE, 'results.tsv'), 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for c in res:
            f.write('\t'.join(str(c.get(k, 'nan')) for k in cols) + '\n')
    print(f"{'case':18s} {'dev8_d':>8s} {'dev8_cy':>8s} {'dev_d':>8s} {'dev_cy':>8s} "
          f"{'twin_d':>8s} {'dsin':>8s} {'divB/B':>8s} {'nmb':>9s} {'cr/del':>9s}")
    for c in res:
        print(f"{c['name']:18s} {c['dev8_delta']:8.3f} {c['dev8_cy']:8.3f} "
              f"{c['dev_delta']:8.3f} {c['dev_cy']:8.3f} "
              f"{c.get('twin_dev_delta', float('nan')):8.1e} {c['dsin_rel_max']:8.1e} "
              f"{c['divb_rel_max']:8.1e} {c['nmb_min']:4d}-{c['nmb_max']:<4d} "
              f"{c['created']:4d}/{c['deleted']:<4d}")
