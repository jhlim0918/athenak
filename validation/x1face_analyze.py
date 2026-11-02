#!/usr/bin/env python3
"""x1-face port validation: shearing-wave history comparison and mass drift.

usage: python3 x1face_analyze.py REF RUN [RUN ...]     (directories with shwave3.{user,hydro}.hst)

rms_lin    = sqrt(mean over the reference's output times t_r of
                  (dVyc_run(t_r) - dVyc_ref(t_r))^2),
             dVyc = user.hst column 3, the run linearly interpolated to t_r (the runs have
             different time steps, so their output times differ by up to one step).
             This reproduces R. Li's numbers (uniform 128^2 vs 256^2: 6.8e-5).
rms_spline = the same on t_k = 0.1 k (k = 0..80), both histories by cubic splines.
mass drift = (M(t_end) - M(0))/M(0) and max_t |M(t) - M(0)|/M(0)  (hydro.hst column 3)
"""
import sys
import numpy as np
from scipy.interpolate import CubicSpline

TK = 0.1*np.arange(81)

def hist(run):
    u = np.loadtxt(run + '/shwave3.user.hst')
    t, i = np.unique(u[:, 0], return_index=True)
    return t, u[i, 2]

def mass(run):
    h = np.loadtxt(run + '/shwave3.hydro.hst')
    m0 = h[0, 2]
    return (h[-1, 2] - m0)/m0, np.max(np.abs(h[:, 2] - m0))/m0, h[-1, 0]

ref = sys.argv[1]
tr, vr = hist(ref)
sr = CubicSpline(tr, vr)(TK)
print(f"# reference {ref}: max|dVyc| = {np.max(np.abs(vr)):.6e}")
print(f"{'run':28s} {'rms_lin':>10s} {'rms_spline':>10s} {'max_lin':>10s} "
      f"{'dM/M(end)':>11s} {'max|dM|/M':>11s} {'t_end':>6s}")
for run in sys.argv[2:]:
    t, v = hist(run)
    msk = tr <= t[-1]
    dl = np.interp(tr[msk], t, v) - vr[msk]
    ks = TK <= t[-1]
    ds = CubicSpline(t, v)(TK[ks]) - sr[ks]
    dm, dmax, tend = mass(run)
    print(f"{run:28s} {np.sqrt(np.mean(dl**2)):10.3e} {np.sqrt(np.mean(ds**2)):10.3e} "
          f"{np.max(np.abs(dl)):10.3e} {dm:11.3e} {dmax:11.3e} {tend:6.2f}")
