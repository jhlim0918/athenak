"""Per run: t, max rho, MeshBlock count and blocks per level from the density dumps.
Usage: python3 rhomax.py RUN_DIR [...]   (writes RUN_DIR/rhomax.txt)"""
import sys, glob, os
import numpy as np
sys.path.insert(0, '/data/limjay/athenak_fargoamr_gpu/vis/python')
import bin_convert as bc
for d in sys.argv[1:]:
    files = sorted(glob.glob(os.path.join(d, 'bin', '*_u_d.*.bin')))
    rows = []
    root = None
    for f in files:
        b = bc.read_binary(f)
        var = b['var_names'][0]
        rmax = max(float(np.max(x)) for x in b['mb_data'][var])
        lev = np.array([l[3] for l in b['mb_logical']])
        if root is None:
            root = int(lev.min())   # the first dump is the root grid
        lev = lev - root
        nl = [int(np.sum(lev == L)) for L in range(4)]
        rows.append([b['time'], rmax, len(lev)] + nl)
    np.savetxt(os.path.join(d, 'rhomax.txt'), np.array(rows),
               header='t max_rho nmb nmb_L0 nmb_L1 nmb_L2 nmb_L3', fmt='%.6e')
    print(d, len(rows), 'dumps; final t=%.2f max rho %.3g nmb %d' % tuple(rows[-1][:3]))
