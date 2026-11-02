"""KOS02 tier 1: vertical equilibria and axisymmetric stability (Kim, Ostriker & Stone
2002, ApJ 581, 1080, sec. 3, Figs. 1 and 3b).

Code units: c_s = 1, 4 pi G = 1, Sigma0 = 4, so pi G Sigma0 = 1, H_inf = c_s^2/(pi G
Sigma0) = 1, rho_inf = pi G Sigma0^2/(2 c_s^2) = 2, 2 pi G rho_inf = 1, and
Q = kappa c_s/(pi G Sigma0) = kappa. External gravity g_*/H_* = 4 pi G rho_inf/s0 = 2/s0.
Rotation: q = 1 (flat rotation curve), kappa = sqrt(2) Omega, so Omega = Q/sqrt(2).

For each (s0, beta): solve the equilibrium (KOS02 eq. 16 in dimensional form) for rho00
and H = Sigma0/(2 rho00); take k_max and the marginal Q_c of KOS02 eq. 18,
    omega^2 = kappa^2 + (c_s^2 + v_A^2) k^2 - [k H/(1 + k H)] 4 pi G rho00;
write inputs at Q = f Q_c(eq. 18), f = 0.85, 1.0, 1.15, each seeding one mode at k_max
(L_x = 2 pi/k_max). Cells are cubic (the multigrid solver needs it): dz = L_x/n_x with
about 8 cells per H, n_z = 8 H/dz rounded up to a multiple of 8, n_y = 8.

Usage: python3 setup.py   -> cases.json, inputs/<name>.athinput
"""
import json, os, math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
CS2, FPG, SIG0 = 1.0, 1.0, 4.0
PIG = FPG/4.0
HINF = CS2/(PIG*SIG0)
RHOINF = PIG*SIG0**2/(2.0*CS2)

def equilibrium(s0, beta):
    """rho00, H for the given s0 (None: no external gravity), beta (None: no field)."""
    fmag = 1.0 + (0.5/beta if beta else 0.0)
    g = FPG*RHOINF/s0 if s0 else 0.0
    zinf = 40.0*HINF
    def colmass(rho00):
        f = lambda z, y: [-(g*z + FPG*y[1])/(CS2*fmag), math.exp(y[0])]
        sol = solve_ivp(f, (0, zinf), [math.log(rho00), 0.0], rtol=1e-11, atol=1e-14)
        return 2.0*sol.y[1, -1]
    lo, hi = 1e-4*RHOINF, 1e6*RHOINF
    rho00 = brentq(lambda r: colmass(r) - SIG0, lo, hi, xtol=1e-14, rtol=1e-13)
    return rho00, SIG0/(2.0*rho00)

def eq18(rho00, H, beta):
    va2 = CS2/beta if beta else 0.0
    F = lambda k: (CS2 + va2)*k*k - (k*H/(1.0 + k*H))*FPG*rho00
    r = minimize_scalar(F, bounds=(1e-4/H, 20.0/H), method='bounded',
                        options={'xatol': 1e-12})
    return r.x, math.sqrt(-r.fun)      # k_max, Q_c (= kappa_c in these units)

TEMPLATE = """<comment>
problem   = KOS02 axisymmetric stability: s0 = {s0}, beta = {beta}, Q = {Q:.6f}

<job>
basename  = kos

<mesh>
nghost    = 4
nx1       = {nx}
x1min     = {x1min:.12f}
x1max     = {x1max:.12f}
ix1_bc    = shear_periodic
ox1_bc    = shear_periodic
nx2       = 8
x2min     = {x2min:.12f}
x2max     = {x2max:.12f}
ix2_bc    = periodic
ox2_bc    = periodic
nx3       = {nz}
x3min     = {x3min:.12f}
x3max     = {x3max:.12f}
ix3_bc    = hse_outflow
ox3_bc    = hse_outflow

<meshblock>
nx1       = 8
nx2       = 8
nx3       = 8

<time>
evolution  = dynamic
integrator = rk2
cfl_number = 0.3
nlim       = -1
tlim       = {tlim:.4f}
ndiag      = 1000

<{fluid}>
eos             = isothermal
reconstruct     = plm
rsolver         = hlle
iso_sound_speed = 1.0
dfloor          = 1.0e-8

<shearing_box>
qshear = 1.0
omega0 = {omega0:.12f}
stratified = true
vertical_gravity_coef = {vg:.12f}   # g_*/H_* = 4 pi G rho_inf/s0
orbital_advection = true

<gravity>
solver    = multigrid
four_pi_G = 1.0
mg_bc     = slab
mg_remap  = ppmx
threshold = -1.0
niteration = 12
full_multigrid = true
fmg_ncycle = 1

<{fluid}_srcterms>
self_gravity = true

<problem>
pgen_name = kos_disk
sigma0    = 4.0
s0        = {s0v}
beta      = {betav}
amp       = 1.0e-4
nwx       = 1
dfloor_ic = 1.0e-6
user_hist = true

<output1>
file_type      = hst
dt             = {dthst:.5f}
data_format    = %.15e
user_hist_only = true
"""

def main():
    os.makedirs(os.path.join(HERE, 'inputs'), exist_ok=True)
    s0s = [0.01, 0.1, 1.0, 10.0, 25.0, 100.0]   # Fig. 3b range; 1, 25: KOS02 Table 1
    betas = [None, 10.0, 1.0]
    cases = []
    for s0 in s0s:
        for beta in betas:
            rho00, H = equilibrium(s0, beta)
            kmax, qc = eq18(rho00, H, beta)
            lx = 2.0*math.pi/kmax
            nx = max(64, 8*round(lx/(H/8.0)/8.0))
            dz = lx/nx
            nz = 8*math.ceil(8.0*H/dz/8.0)
            lz, ly = nz*dz, 8*dz
            for f in (0.85, 1.0, 1.15):
                Q = f*qc
                kappa = Q
                tlim = (14.0 if f < 1.0 else 30.0)/qc   # unstable: stop before collapse
                bt = 'inf' if beta is None else ('%g' % beta)
                name = 's0_%g_b%s_f%.2f' % (s0, bt, f)
                txt = TEMPLATE.format(
                    s0=s0, beta=bt, Q=Q, nx=nx, nz=nz, x1min=-lx/2, x1max=lx/2,
                    x2min=-ly/2, x2max=ly/2, x3min=-lz/2, x3max=lz/2, tlim=tlim,
                    fluid='hydro' if beta is None else 'mhd', omega0=kappa/math.sqrt(2.0),
                    vg=FPG*RHOINF/s0, s0v=s0, betav=(-1.0 if beta is None else beta),
                    dthst=tlim/600.0)
                open(os.path.join(HERE, 'inputs', name + '.athinput'), 'w').write(txt)
                cases.append(dict(name=name, s0=s0, beta=bt, f=f, Q=Q, qc_eq18=qc,
                                  kmax=kmax, rho00=rho00, H=H, H_over_Hinf=H/HINF,
                                  lx=lx, nx=nx, nz=nz, lz=lz, tlim=tlim))
            print('s0=%-6g beta=%-4s  H/Hinf=%.4f  rho00=%.4f  kmax*H=%.4f  Qc(eq18)=%.4f'
                  '  nx=%d nz=%d  Lz/H=%.2f' % (s0, bt, H/HINF, rho00, kmax*H, qc, nx, nz, lz/H))
    json.dump(cases, open(os.path.join(HERE, 'cases.json'), 'w'), indent=1)

if __name__ == '__main__':
    main()
