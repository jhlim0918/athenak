"""KOS02 tier 2: reduce the density dumps of one model to what the KOS02 comparisons need.

For every bin dump (density only, every 0.05 orbit): the time, the maximum density and
where it is, the surface density map Sigma(x, y) = Int rho dz (float32), the differential
mass dM/dz = Int rho dx dy (KOS02 eq. 19) and the total mass in the box.

Usage (on the cluster, from the run directory's parent):
    python3 tier2_reduce.py MODEL      -> MODEL/reduced.npz
"""
import sys, os, glob
import numpy as np
sys.path.insert(0, '/data/limjay/athenak_fargoamr_gpu/vis/python')
import bin_convert as bc

m = sys.argv[1]
files = sorted(glob.glob(os.path.join(m, 'bin', 'kos%s.*.bin' % m)))
T, RMAX, LOC, SIG, DMDZ, MASS = [], [], [], [], [], []
for f in files:
    b = bc.read_binary(f)
    var = b['var_names'][0]
    geo = np.array(b['mb_geometry'])
    nz, ny, nx = b['mb_data'][var][0].shape
    dx = (geo[0, 1] - geo[0, 0])/nx; dy = (geo[0, 3] - geo[0, 2])/ny; dz = (geo[0, 5] - geo[0, 4])/nz
    NX = int(round((b['x1max'] - b['x1min'])/dx)); NY = int(round((b['x2max'] - b['x2min'])/dy))
    NZ = int(round((b['x3max'] - b['x3min'])/dz))
    rho = np.empty((NZ, NY, NX), dtype=np.float32)
    for g, a in zip(geo, b['mb_data'][var]):
        i0 = int(round((g[0] - b['x1min'])/dx)); j0 = int(round((g[2] - b['x2min'])/dy))
        k0 = int(round((g[4] - b['x3min'])/dz))
        rho[k0:k0 + nz, j0:j0 + ny, i0:i0 + nx] = a
    k = int(np.argmax(rho)); kk, jj, ii = np.unravel_index(k, rho.shape)
    T.append(b['time']); RMAX.append(float(rho.flat[k]))
    LOC.append((b['x1min'] + (ii + 0.5)*dx, b['x2min'] + (jj + 0.5)*dy, b['x3min'] + (kk + 0.5)*dz))
    SIG.append(rho.sum(axis=0, dtype=np.float64).astype(np.float32)*dz)
    DMDZ.append(rho.sum(axis=(1, 2), dtype=np.float64)*dx*dy)
    MASS.append(DMDZ[-1].sum()*dz)
    print('%s t=%8.3f rhomax=%10.4g sigmax=%10.4g mass=%.6f' % (os.path.basename(f), T[-1], RMAX[-1],
          SIG[-1].max(), MASS[-1]), flush=True)
z = b['x3min'] + (np.arange(NZ) + 0.5)*dz
np.savez_compressed(os.path.join(m, 'reduced.npz'), t=np.array(T), rhomax=np.array(RMAX),
                    loc=np.array(LOC), sigma=np.array(SIG), dmdz=np.array(DMDZ), mass=np.array(MASS),
                    z=z, ext=np.array([b['x1min'], b['x1max'], b['x2min'], b['x2max'], b['x3min'], b['x3max']]))
print('wrote', os.path.join(m, 'reduced.npz'))
