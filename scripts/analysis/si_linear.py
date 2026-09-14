#!/usr/bin/env python3
"""Linear streaming instability of Youdin & Goodman (2005) with an isothermal gas: the
dispersion relation and the eigenvector in the convention of the streaming_linear
problem generator (inputs/dust/streaming_linA_smr.athinput, streaming_modelAs.athinput).

Model: axisymmetric shearing sheet, isothermal compressible gas, pressureless dust fluid,
mutual drag, NSH background.  Code units Omega = c_s = rho_g = 1, eta v_K = 0.05.
Perturbations ~ exp(i kx x + i kz z - i omega t); the growth rate is s = Im(omega).  The
state vector is (drho_g/rho_g, u_x, u_y, u_z, drho_p/rho_p, v_x, v_y, v_z).

The generator seeds the standing-in-z form of YJ07's Table 1: densities and horizontal
velocities ~ Re[A exp(i kx x)] cos(kz z), the vertical velocities ~ Re[i A_z exp(i kx x)]
sin(kz z), all amplitudes absolute in code units, the dust density real; that is the
exp(i kz z) eigenvector fed in directly (checked: the linA input file's fourteen numbers
are reproduced to every printed digit).

Validation (YJ07 Table 1, incompressible gas): linA (tau_s 0.1, eps 3, K 30) s = 0.4190204,
this solver 0.4190091; linB (tau_s 0.1, eps 0.2, K 6) s = 0.0154764, this solver 0.0154862.
The residual is the gas compressibility.

usage:  si_linear.py                      validation + a growth-rate map for the BA parameters
        si_linear.py TAUS EPS KX KZ [AMP]  omega and the <problem> block for that mode
                                          (AMP = relative dust-density amplitude, default 1e-6)
"""
import sys
import numpy as np

ETAVK = 0.05


def nsh(eps, taus, etavk=ETAVK):
    """NSH drift velocities (gas x, y; dust x, y) relative to the Keplerian shear."""
    D = (1 + eps)**2 + taus**2
    return (2*eps*taus/D*etavk, -(1 + eps + taus**2)/D*etavk,
            -2*taus/D*etavk, -(1 + eps)/D*etavk)


def matrix(kx, kz, eps, taus, q=1.5):
    Ux, Uy, Vx, Vy = nsh(eps, taus)
    M = np.zeros((8, 8), dtype=complex)
    M[0, 0] = -1j*kx*Ux; M[0, 1] = -1j*kx; M[0, 3] = -1j*kz
    M[1, 1] = -1j*kx*Ux - eps/taus; M[1, 2] = 2; M[1, 0] = -1j*kx - eps/taus*(Vx - Ux)
    M[1, 5] = eps/taus; M[1, 4] = eps/taus*(Vx - Ux)
    M[2, 2] = -1j*kx*Ux - eps/taus; M[2, 1] = -(2 - q); M[2, 0] = -eps/taus*(Vy - Uy)
    M[2, 6] = eps/taus; M[2, 4] = eps/taus*(Vy - Uy)
    M[3, 3] = -1j*kx*Ux - eps/taus; M[3, 0] = -1j*kz; M[3, 7] = eps/taus
    M[4, 4] = -1j*kx*Vx; M[4, 5] = -1j*kx; M[4, 7] = -1j*kz
    M[5, 5] = -1j*kx*Vx - 1/taus; M[5, 6] = 2; M[5, 1] = 1/taus
    M[6, 6] = -1j*kx*Vx - 1/taus; M[6, 5] = -(2 - q); M[6, 2] = 1/taus
    M[7, 7] = -1j*kx*Vx - 1/taus; M[7, 3] = 1/taus
    return M


def fastest(kx, kz, eps, taus):
    """omega and eigenvector of the fastest-growing mode: -i omega y = M y."""
    w, V = np.linalg.eig(1j*matrix(kx, kz, eps, taus))
    k = np.argmax(w.imag)
    return w[k], V[:, k]


def problem_block(taus, eps, Kx, Kz, amp=1e-6):
    """The <problem> keys of streaming_linear for the fastest mode at (Kx, Kz) = k eta r."""
    kx, kz = Kx/ETAVK, Kz/ETAVK
    w, y = fastest(kx, kz, eps, taus)
    A = y*(amp/y[4])                       # dust density relative amplitude = amp, real
    keys = [("kx", kx), ("kz", kz), ("omega_re", w.real),
            ("drhog_re", A[0].real), ("drhog_im", A[0].imag),
            ("dugx_re", A[1].real), ("dugx_im", A[1].imag),
            ("dugp_re", A[2].real), ("dugp_im", A[2].imag),
            ("dugz_re", A[3].real), ("dugz_im", A[3].imag),
            ("drhop_re", A[4].real*eps),
            ("dvpx_re", A[5].real), ("dvpx_im", A[5].imag),
            ("dvpp_re", A[6].real), ("dvpp_im", A[6].imag),
            ("dvpz_re", A[7].real), ("dvpz_im", A[7].imag)]
    lines = [f"# tau_s = {taus}, eps = {eps}, Kx = {Kx}, Kz = {Kz}: omega = {w.real:.7f} {w.imag:+.7f} i",
             f"# (growth rate s = {w.imag:.7f}; wavelength {2*np.pi/kx:.6g} H radial, {2*np.pi/kz:.6g} H vertical)",
             f"etavk      = {ETAVK}", f"eps        = {eps}"]
    lines += [f"{k:10s} = {v:.15g}" for k, v in keys]
    return w, "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) >= 5:
        taus, eps, Kx, Kz = map(float, sys.argv[1:5])
        amp = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-6
        print(problem_block(taus, eps, Kx, Kz, amp)[1])
        sys.exit(0)
    for name, taus, eps, K, s_ref in (("linA", 0.1, 3.0, 30.0, 0.4190204), ("linB", 0.1, 0.2, 6.0, 0.0154764)):
        w, _ = fastest(K/ETAVK, K/ETAVK, eps, taus)
        print(f"{name}: s = {w.imag:.7f} (YJ07 {s_ref}), Re omega = {w.real:+.7f}")
    taus, eps = 1.0, 0.2
    Ks = np.logspace(-1, 2, 61)
    smap = np.array([[fastest(kx/ETAVK, kz/ETAVK, eps, taus)[0].imag for kz in Ks] for kx in Ks])
    i, j = np.unravel_index(np.argmax(smap), smap.shape)
    print(f"BA (tau_s = 1, eps = 0.2): max s = {smap[i, j]:.5f} at Kx = {Ks[i]:.3g}, Kz = {Ks[j]:.3g}")
    Kd = np.linspace(0.5, 1.6, 111); sd = [fastest(K/ETAVK, K/ETAVK, eps, taus)[0].imag for K in Kd]
    print(f"  Kx = Kz: max s = {max(sd):.5f} at K = {Kd[int(np.argmax(sd))]:.3f}; growth confined to Kx < ~1.5")
