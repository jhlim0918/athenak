"""Slices of an AMR be_collapse dump for the Tomida & Stone (2023) Fig. 12 panels.

At each point of a uniform 2D grid the value of the finest MeshBlock containing it is
taken (piecewise constant).  Writes an npz with, per panel, the coordinates in au and the
fields in physical units: rho [g/cc], velocity [km/s], plasma beta, radial (spherical)
and rotational velocity [km/s], in-plane velocity / field components for the vectors.

Usage: python3 be_collapse_slices.py <mhd_w.bin> <mhd_bcc.bin> <out.npz> [N]
Units (be_collapse pgen, M = 1 Msun, T = 10 K, f = 1.2): l0 = 1302.7 au,
rho0 = 1.1336e-18 g/cc, v0 = 0.19 km/s; B in units sqrt(4 pi rho0 v0^2), P_B = B^2/2.
"""
import sys
import numpy as np
sys.path.insert(0, '/data/limjay/athenak/vis/python')
import bin_convert as bc

L0_AU, RHO0, V0_KMS, GAMMA = 1302.7, 1.1336e-18, 0.19, 5.0/3.0


def sampler(d, db):
    lv = np.array(d['mb_logical'])[:, 3]
    g = np.array(d['mb_geometry'])
    order = np.argsort(lv)                      # coarse first, finer blocks overwrite
    def sample(x, y, z):
        out = {k: np.full(x.shape, np.nan) for k in
               ('dens', 'velx', 'vely', 'velz', 'eint', 'bcc1', 'bcc2', 'bcc3')}
        for m in order:
            x0, x1, y0, y1, z0, z1 = g[m][:6]
            sel = (x >= x0) & (x < x1) & (y >= y0) & (y < y1) & (z >= z0) & (z < z1)
            if not sel.any():
                continue
            n3, n2, n1 = d['mb_data']['dens'][m].shape
            i = np.minimum(((x[sel] - x0)/(x1 - x0)*n1).astype(int), n1 - 1)
            j = np.minimum(((y[sel] - y0)/(y1 - y0)*n2).astype(int), n2 - 1)
            k = np.minimum(((z[sel] - z0)/(z1 - z0)*n3).astype(int), n3 - 1)
            for key in ('dens', 'velx', 'vely', 'velz', 'eint'):
                out[key][sel] = d['mb_data'][key][m][k, j, i]
            for key in ('bcc1', 'bcc2', 'bcc3'):
                out[key][sel] = db['mb_data'][key][m][k, j, i]
        return out
    return sample


def panel(sample, plane, half_au, n):
    s = np.linspace(-half_au, half_au, n)/L0_AU
    a, b = np.meshgrid(s, s, indexing='xy')          # a: horizontal, b: vertical
    eps = 1e-7                                       # just off the plane (cell faces at 0)
    if plane == 'xz':
        x, y, z = a, np.full_like(a, eps), b
    else:                                            # 'xy' equatorial
        x, y, z = a, b, np.full_like(a, eps)
    f = sample(x, y, z)
    r = np.sqrt(x**2 + y**2 + z**2)
    rc = np.sqrt(x**2 + y**2)
    P = (GAMMA - 1.0)*f['eint']
    b2 = f['bcc1']**2 + f['bcc2']**2 + f['bcc3']**2
    vr = (x*f['velx'] + y*f['vely'] + z*f['velz'])/r
    vphi = (x*f['vely'] - y*f['velx'])/rc
    hv, vv = ('velx', 'velz') if plane == 'xz' else ('velx', 'vely')
    hb, vb = ('bcc1', 'bcc3') if plane == 'xz' else ('bcc1', 'bcc2')
    return dict(coord=s*L0_AU, rho=f['dens']*RHO0, beta=P/(0.5*b2),
                vr=vr*V0_KMS, vphi=vphi*V0_KMS,
                vh=f[hv]*V0_KMS, vv=f[vv]*V0_KMS, bh=f[hb], bv=f[vb])


if __name__ == '__main__':
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 400
    d = bc.read_binary(sys.argv[1])
    db = bc.read_binary(sys.argv[2])
    sample = sampler(d, db)
    out = {'time': d['time'], 'time_yr': d['time']*32503.0}
    for name, plane, half in (('large', 'xz', 250.0), ('eq', 'xy', 130.0),
                              ('core', 'xz', 45.0)):
        p = panel(sample, plane, half, n)
        for k, v in p.items():
            out[name + '_' + k] = v
        print('%s: rho %.2e..%.2e g/cc' % (name, np.nanmin(p['rho']), np.nanmax(p['rho'])))
    np.savez_compressed(sys.argv[3], **out)
    print('t = %.4f (%.0f yr) ->' % (d['time'], out['time_yr']), sys.argv[3])
