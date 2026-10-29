"""Midplane slice of the first variable of an AthenaK bin dump, resampled to the finest
level, plus the MeshBlock outlines cutting z = 0.
Usage: python3 slice.py DUMP.bin OUT_PREFIX  ->  OUT_PREFIX.f32 (N x N float32, [j][i]),
OUT_PREFIX.txt (header: t N x1min x1max x2min x2max; rows x0 x1 y0 y1 r, r = pixels
per cell of the block)."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'vis', 'python'))
import bin_convert as bc
b = bc.read_binary(sys.argv[1]); out = sys.argv[2]
var = b['var_names'][0]
x1min, x1max, x2min, x2max = b['x1min'], b['x1max'], b['x2min'], b['x2max']
geo = np.array(b['mb_geometry'])
nxb = b['mb_data'][var][0].shape[2]
dxmin = ((geo[:, 1] - geo[:, 0]) / nxb).min()
N = int(round((x1max - x1min) / dxmin))
img = np.zeros((N, N), dtype=np.float32); rows = []
for m, g in enumerate(geo):
    a = b['mb_data'][var][m]; nz, ny, nx = a.shape
    x0, x1, y0, y1, z0, z1 = g
    if not (z0 <= 0.0 < z1): continue
    dz = (z1 - z0) / nz; k = int((0.0 - z0) / dz)
    r = int(round((x1 - x0) / nx / dxmin))
    i0 = int(round((x0 - x1min) / dxmin)); j0 = int(round((y0 - x2min) / dxmin))
    img[j0:j0 + ny * r, i0:i0 + nx * r] = np.kron(a[k], np.ones((r, r)))
    rows.append([x0, x1, y0, y1, r])
img.tofile(out + '.f32')
np.savetxt(out + '.txt', np.array(rows), header='%.9e %d %.9e %.9e %.9e %.9e' % (b['time'], N, x1min, x1max, x2min, x2max))
print(out, 't=%.3f N=%d blocks=%d' % (b['time'], N, len(rows)))
