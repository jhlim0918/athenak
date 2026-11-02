"""Random density perturbation of KOS02 sec. 2.2: a Gaussian random field delta with
<|delta_k|^2> ~ k^(-11/3) for 1 <= k L_x/2 pi <= 32 (zero outside), normalized to an rms of
1% in real space, written as raw float64 [nz][ny][nx] for <problem> pert_file of kos_disk.

Usage: python3 gen_noise.py NX NY NZ LX LY LZ SEED OUT
"""
import sys
import numpy as np

nx, ny, nz = (int(a) for a in sys.argv[1:4])
lx, ly, lz = (float(a) for a in sys.argv[4:7])
seed, out = int(sys.argv[7]), sys.argv[8]
rng = np.random.default_rng(seed)
kx = 2*np.pi*np.fft.fftfreq(nx, d=lx/nx)
ky = 2*np.pi*np.fft.fftfreq(ny, d=ly/ny)
kz = 2*np.pi*np.fft.rfftfreq(nz, d=lz/nz)
KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing='ij')      # shape (nz//2+1, ny, nx)
k = np.sqrt(KX**2 + KY**2 + KZ**2)
kn = k*lx/(2*np.pi)
amp = np.where((kn >= 1.0) & (kn <= 32.0), np.maximum(k, 1e-30)**(-11.0/6.0), 0.0)
noise = (rng.standard_normal(k.shape) + 1j*rng.standard_normal(k.shape))*amp
# irfftn over axes (z, y, x) with z the half axis: transpose so the last axis is z
d = np.fft.irfftn(np.moveaxis(noise, 0, -1), s=(ny, nx, nz), axes=(0, 1, 2))
d = np.moveaxis(d, -1, 0)                                  # back to [nz][ny][nx]
d -= d.mean()
d *= 0.01/d.std()
d.astype(np.float64).tofile(out)
print(out, d.shape, 'rms %.4f min %.4f max %.4f' % (d.std(), d.min(), d.max()))
