"""Data for the KOS02 Fig. 3b replica: Q_c(s0) of eq. 18 on a fine s0 grid for beta = inf, 10, 1,
and the simulated Q_c of every (s0, beta) case (qc.json from fit.py).

Usage: python3 fig3b_data.py  -> fig3b_curves.txt, fig3b_points.txt (read by
validation/figures/plot_kos02_fig3b.jl)
"""
import json, os
import numpy as np
from setup import equilibrium, eq18

HERE = os.path.dirname(os.path.abspath(__file__))
BETAS = [(None, 'inf'), (10.0, '10'), (1.0, '1')]

with open(os.path.join(HERE, 'fig3b_curves.txt'), 'w') as f:
    f.write('# s0  Qc_eq18(beta=inf)  Qc_eq18(beta=10)  Qc_eq18(beta=1)  [KOS02 eq. 18 with the eq. 16 equilibrium]\n')
    for s0 in np.logspace(-3, 2, 101):
        q = [eq18(*equilibrium(s0, b), b)[1] for b, _ in BETAS]
        f.write('%.6e %.6f %.6f %.6f\n' % (s0, *q))
    q = [eq18(*equilibrium(None, b), b)[1] for b, _ in BETAS]      # s0 -> infinity
    f.write('# s0=inf %.6f %.6f %.6f\n' % tuple(q))

rows = json.load(open(os.path.join(HERE, 'qc.json')))
with open(os.path.join(HERE, 'fig3b_points.txt'), 'w') as f:
    f.write('# s0  beta  Qc_sim  Qc_eq18  slope  Qc_run_min  Qc_run_max  partial\n')
    for r in rows:
        b = 'inf' if r['beta'] in (None, 'inf') else '%g' % float(r['beta'])
        part = int(r['s0'] == 0.01 and b == '1')    # two of its three runs stopped at t = 16 (job limit)
        f.write('%g %s %.5f %.5f %.4f %.5f %.5f %d\n' % (r['s0'], b, r['qc_sim'], r['qc_eq18'],
                r['slope'], min(r['qc_runs']), max(r['qc_runs']), part))
print(open(os.path.join(HERE, 'fig3b_points.txt')).read())
print(open(os.path.join(HERE, 'fig3b_curves.txt')).read().splitlines()[-1])
