"""Linear theory of a magnetized swinging shearing wave (k_z = 0, uniform B0 along y),
real amplitudes delta cos(k.x), w sin(k.x), c cos(k.x) of drho/rho0, dv, dB:
    delta' = -(kx wx + ky wy)
    wx'    = 2 Om wy + kx Psi delta - (B0/rho0)(ky cx - kx cy)
    wy'    = -(2-q) Om wx + ky Psi delta
    cx'    = ky B0 wx
    cy'    = -kx B0 wx - q Om cx
kx(t) = kx0 + q Om ky t, Psi = cs^2 + 4 pi G rho0 / D, D = -k^2 (continuum) or the 7-point
Laplacian eigenvalue -(2/dx)^2 [sin^2(kx dx/2) + sin^2(ky dy/2)] (multigrid).
"""
import numpy as np
from scipy.integrate import solve_ivp


def solve(kx0, ky, b0, fpg, cs=1.0, om=1.0, q=1.5, rho0=1.0, dx=None, dy=None, tmax=8.0,
          t_eval=None, amp=1.0):
    def D(kx):
        if dx is None:
            return -(kx*kx + ky*ky)
        return -((2/dx*np.sin(kx*dx/2))**2 + (2/dy*np.sin(ky*dy/2))**2)

    def rhs(t, y):
        d, wx, wy, cx, cy = y
        kx = kx0 + q*om*ky*t
        psi = cs*cs + fpg*rho0/D(kx)
        return [-(kx*wx + ky*wy),
                2*om*wy + kx*psi*d - (b0/rho0)*(ky*cx - kx*cy),
                -(2-q)*om*wx + ky*psi*d,
                ky*b0*wx,
                -kx*b0*wx - q*om*cx]
    return solve_ivp(rhs, (0, tmax), [amp, 0, 0, 0, 0], t_eval=t_eval, rtol=1e-11, atol=1e-14,
                     method='DOP853')


if __name__ == '__main__':
    kx0, ky = -6.0, 1.0
    t = np.linspace(0, 8, 4001)
    for fpg in (1.9, 0.0):
        for b0 in (0.0, 0.25, 0.5, 0.75, 1.0):
            s = solve(kx0, ky, b0, fpg, t_eval=t)
            d = s.y[0]
            i = np.argmax(np.abs(d))
            print(f"4piG={fpg:.1f} vA={b0:.2f}: peak |delta| {abs(d[i]):8.3f} at t={t[i]:.2f}, "
                  f"delta(8)={d[-1]:+8.3f}, max|cy| {np.max(np.abs(s.y[4])):.3f}")
