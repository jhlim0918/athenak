"""Noise check summary: total kinetic energy (KE1+KE2+KE3) against time; amplitude growth rate s
from the slope of ln KE over t in [1.5, 3] (s = slope/2); KE(3)/KE(0)."""
import glob, numpy as np
print('%-22s %10s %10s %10s  %s' % ('run', 'KE(3)/KE0', 'min KE/KE0', 's (ampl.)', 'verdict'))
for d in sorted(glob.glob('*/MagJeans.mhd.hst')):
    h = np.loadtxt(d); t = h[:, 0]; ke = h[:, 6:9].sum(1)
    m = t >= 1.5; s = 0.5*np.polyfit(t[m], np.log(ke[m]), 1)[0]
    r = ke[-1]/ke[0]
    print('%-22s %10.3g %10.3g %+10.2f  %s' % (d.split('/')[0], r, ke.min()/ke[0], s, 'GROWS' if s > 0.5 and r > 1e2 else 'stable'))
