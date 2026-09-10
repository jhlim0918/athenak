"""Period, amplitude and (x,y) spread of the test-particle oscillation in the
self-gravitating sech^2 slab of the dust_grav_orbit generator, from the user history
(columns time, dt, zsum, vzsum, z2sum, np, gas_rz2), against the exact anharmonic
period of phi(z) = 2 cs^2 ln cosh(z/h) at release amplitude z0.

  python3 grav_orbit_period.py [--z0 Z] [--cs C] [--four-pi-G G] [--rho0 R] file.user.hst ...

Defaults match inputs/dust/dust_grav_orbit.athinput except z0, which must be given
whenever the input's <problem>/z0 is not 0.03 (the refined-midplane twin uses 0.30).
"""
import argparse
import numpy as np
from scipy.integrate import quad

ap = argparse.ArgumentParser()
ap.add_argument("--z0", type=float, default=0.03)
ap.add_argument("--cs", type=float, default=0.1)
ap.add_argument("--four-pi-G", type=float, default=1.0, dest="four_pi_G")
ap.add_argument("--rho0", type=float, default=1.0)
ap.add_argument("files", nargs="+")
a = ap.parse_args()

h = a.cs/np.sqrt(0.5*a.four_pi_G*a.rho0)
phi = lambda z: 2*a.cs*a.cs*np.log(np.cosh(z/h))
# T = 4 int_0^z0 dz / sqrt(2 (phi(z0) - phi(z))), with z = z0 sin(theta) to tame the
# endpoint singularity
f = lambda th: a.z0*np.cos(th)/np.sqrt(2*(phi(a.z0) - phi(a.z0*np.sin(th))))
T_exact = 4*quad(f, 0, np.pi/2, limit=200)[0]
T_harm = 2*np.pi/np.sqrt(a.four_pi_G*a.rho0)
print(f"h={h:.6f}  z0={a.z0:g}  z0/h={a.z0/h:.4f}  T_harmonic={T_harm:.6f}  "
      f"T_exact={T_exact:.6f}  (ratio {T_exact/T_harm:.5f})")


def periods(fn):
    d = np.loadtxt(fn)
    t, z, vz = d[:, 0], d[:, 2]/d[:, 5], d[:, 3]/d[:, 5]
    s = np.sign(vz)
    idx = np.where(s[1:]*s[:-1] < 0)[0]
    tc = np.array([t[i] - vz[i]*(t[i+1] - t[i])/(vz[i+1] - vz[i]) for i in idx])
    if len(tc) < 3:
        return np.array([]), d, z
    p = np.concatenate([np.diff(tc[::2]), np.diff(tc[1::2])])
    return p[~np.isnan(p)], d, z


for fn in a.files:
    p, d, z = periods(fn)
    spread = np.sqrt(np.maximum(d[:, 4]/d[:, 5] - (d[:, 2]/d[:, 5])**2, 0)).max()
    gas = (d[-1, 6] - d[0, 6])/d[0, 6]
    if len(p) == 0:
        print(f"{fn:34s} dt={d[1,1]:.5f} t_end={d[-1,0]:.3f} (fewer than two half "
              f"periods)  |z|max={np.abs(z).max():.6f} xy-spread={spread:.1e}")
        continue
    zl = np.abs(z[-max(len(z)//5, 1):]).max()
    print(f"{fn:34s} dt={d[1,1]:.5f} n_per={len(p)} T={p.mean():.6f} (+-{p.std():.1e})"
          f"  T/T_exact-1={p.mean()/T_exact-1:+.3e}  |z|max={np.abs(z).max():.6f}"
          f"  late|z|max={zl:.6f}  amp_drift={(zl-a.z0)/a.z0:+.3e}"
          f"  xy-spread={spread:.2e}  gas<rz2>drift={gas:+.3e}")
    print(f"{'':34s}   per-period T: " + " ".join(f"{x:.5f}" for x in p))
