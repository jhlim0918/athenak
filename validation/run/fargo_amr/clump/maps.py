"""Midplane density slices resampled to the finest level, plus MeshBlock outlines.
Usage: python3 maps.py RUN_DIR DUMP [DUMP ...]
Writes RUN_DIR/maps/map_DUMP.f32 (N x N float32, row-major, [j][i] = (y, x)) and
map_DUMP.txt (header: t N x1min x1max x2min x2max; rows x0 x1 y0 y1 r for each block
that cuts z = 0, r = fine pixels per cell of that block)."""
import sys, glob, os
import numpy as np
sys.path.insert(0, os.environ.get('BCPATH', '/data/limjay/athenak_fargoamr_gpu/vis/python'))
import bin_convert as bc
d = sys.argv[1]
os.makedirs(os.path.join(d, 'maps'), exist_ok=True)
for dump in sys.argv[2:]:
    f = glob.glob(os.path.join(d, 'bin', '*_u_d.%05d.bin' % int(dump)))[0]
    b = bc.read_binary(f)
    var = b['var_names'][0]
    x1min, x1max, x2min, x2max = b['x1min'], b['x1max'], b['x2min'], b['x2max']
    geo = np.array(b['mb_geometry'])
    nxb = b['mb_data'][var][0].shape[2]
    dxmin = ((geo[:, 1] - geo[:, 0]) / nxb).min()     # geometry rows: x0 x1 y0 y1 z0 z1
    N = int(round((x1max - x1min) / dxmin))
    img = np.zeros((N, N), dtype=np.float32)
    rows = []
    for m, g in enumerate(geo):
        a = b['mb_data'][var][m]               # (nz, ny, nx)
        nz, ny, nx = a.shape
        x0, x1, y0, y1, z0, z1 = g
        if not (z0 <= 0.0 < z1):
            continue
        dx = (x1 - x0) / nx; dy = (y1 - y0) / ny; dz = (z1 - z0) / nz
        k = int((0.0 - z0) / dz)
        r = int(round(dx / dxmin))
        i0 = int(round((x0 - x1min) / dxmin)); j0 = int(round((y0 - x2min) / dxmin))
        img[j0:j0 + ny * r, i0:i0 + nx * r] = np.kron(a[k], np.ones((r, r)))
        rows.append([x0, x0 + nx * dx, y0, y0 + ny * dy, r])
    img.tofile(os.path.join(d, 'maps', 'map_%05d.f32' % int(dump)))
    np.savetxt(os.path.join(d, 'maps', 'map_%05d.txt' % int(dump)), np.array(rows),
               header='%.9e %d %.9e %.9e %.9e %.9e' % (b['time'], N, x1min, x1max, x2min, x2max))
    print(d, dump, 't=%.3f N=%d max=%.4g min=%.3g blocks_mid=%d' % (b['time'], N, img.max(), img.min(), len(rows)))
