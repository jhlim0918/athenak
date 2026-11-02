"""KOS02 tier 2: the 3-D models A-J of Table 1 (Kim, Ostriker & Stone 2002).

Same code units as setup.py (c_s = 1, 4 pi G = 1, Sigma0 = 4: H_inf = 1, Q = kappa).
kappa^2 = (4 - 2q) Omega^2, so Omega = Q/sqrt(4 - 2q). Box L_x = L_y = 17 H (s0 = 25) or
25 H (s0 = 1) and about 8 H in z, H the model's own scale height (eq. 14); cubic cells
(multigrid): A-E 256 x 256 x 128 (dx = 17H/256, their 256 x 128 x 128 is finer in x
only), F-J 192 x 192 x 64 (dx = 0.13 H; their 64^3 has dz = 0.125 H, dx = 0.39 H). Models
without self-gravity keep the equilibrium's vertical self-gravity frozen (footnote 8).
Isothermal HLLE throughout (isothermal HLLD is unstable at beta <~ 1). Random density
noise (gen_noise.py, 1% rms, k^-11/3). Run lengths: 4 orbits (A-D), 2 (E), 5 (F-J).

Usage: python3 tier2.py  -> tier2/inputs/<model>.athinput, tier2/noise/*.bin, tier2/models.json
"""
import json, os, math, subprocess, sys
from setup import equilibrium, FPG, RHOINF

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'tier2')
MODELS = [  # name, Q, s0, beta (None = no field), q, self-gravity
    ('A', 0.7, 25.0, 1.0, 0.0, False), ('B', 0.7, 25.0, 1.0, 1.0, False),
    ('C', 0.7, 25.0, None, 1.0, True), ('D', 0.7, 25.0, 1.0, 1.0, True),
    ('E', 0.7, 25.0, 1.0, 0.0, True), ('F', 1.5, 1.0, 1.0, 1.0, False),
    ('G', 1.5, 1.0, 1.0, 1.0, True), ('H', 1.0, 1.0, 1.0, 1.0, False),
    ('I', 1.0, 1.0, 1.0, 1.0, True), ('J', 1.0, 1.0, None, 1.0, True)]

TEMPLATE = """<comment>
problem   = KOS02 model {name}: Q = {Q}, s0 = {s0}, beta = {bt}, q = {q}, self-gravity = {sg}

<job>
basename  = kos{name}

<mesh>
nghost    = 4
nx1       = {nx}
x1min     = {hx:.12f}
x1max     = {hxp:.12f}
ix1_bc    = shear_periodic
ox1_bc    = shear_periodic
nx2       = {nx}
x2min     = {hx:.12f}
x2max     = {hxp:.12f}
ix2_bc    = periodic
ox2_bc    = periodic
nx3       = {nz}
x3min     = {hz:.12f}
x3max     = {hzp:.12f}
ix3_bc    = hse_outflow
ox3_bc    = hse_outflow

<meshblock>
nx1       = 16
nx2       = 16
nx3       = 16

<time>
evolution  = dynamic
integrator = rk2
cfl_number = 0.3
nlim       = -1
tlim       = {tlim:.6f}     # {norb} orbits, t_orb = 2 pi/Omega = {torb:.6f}
ndiag      = 1000

<{fluid}>
eos             = isothermal
reconstruct     = plm
rsolver         = hlle
iso_sound_speed = 1.0
dfloor          = 1.0e-8

<shearing_box>
qshear = {q}
omega0 = {omega:.12f}
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
self_gravity = {sgflag}

<problem>
pgen_name = kos_disk
sigma0    = 4.0
s0        = {s0}
beta      = {betav}
amp       = 0.0
pert_file = {noise}
frozen_sg = {frozen}
dfloor_ic = 1.0e-6
user_hist = true

<output1>
file_type      = hst
dt             = {dthst:.6f}
data_format    = %.15e
user_hist_only = true

<output2>
file_type      = bin
variable       = {fluid}_w_d
dt             = {dtbin:.6f}     # 0.05 orbit

<output3>
file_type      = rst
dt             = {dtrst:.6f}     # 0.5 orbit
"""

def main(noise_dir_remote):
    os.makedirs(os.path.join(OUT, 'inputs'), exist_ok=True)
    os.makedirs(os.path.join(OUT, 'noise'), exist_ok=True)
    rows = []
    for (name, Q, s0, beta, q, sg) in MODELS:
        rho00, H = equilibrium(s0, beta)
        L = (17.0 if s0 == 25.0 else 25.0)*H
        nx, nzt = (256, 8.0) if s0 == 25.0 else (192, 8.0)
        dx = L/nx
        nz = 16*math.ceil(nzt*H/dx/16)
        lz = nz*dx
        omega = Q/math.sqrt(4.0 - 2.0*q)
        torb = 2.0*math.pi/omega
        norb = {'E': 2}.get(name, 4 if s0 == 25.0 else 5)
        nfile = 'noise_%dx%dx%d_s%d.bin' % (nx, nx, nz, 1)
        npath = os.path.join(OUT, 'noise', nfile)
        if not os.path.exists(npath):
            subprocess.run([sys.executable, os.path.join(HERE, 'gen_noise.py'), str(nx), str(nx),
                            str(nz), repr(L), repr(L), repr(lz), '1', npath], check=True)
        bt = 'inf' if beta is None else '%g' % beta
        txt = TEMPLATE.format(
            name=name, Q=Q, s0=s0, bt=bt, q=q, sg='yes' if sg else 'no (frozen)', nx=nx,
            hx=-L/2, hxp=L/2, nz=nz, hz=-lz/2, hzp=lz/2, tlim=norb*torb, norb=norb,
            torb=torb, fluid='hydro' if beta is None else 'mhd', omega=omega,
            vg=FPG*RHOINF/s0, sgflag='true' if sg else 'false', betav=-1.0 if beta is None else beta,
            noise=os.path.join(noise_dir_remote, nfile), frozen='false' if sg else 'true',
            dthst=torb/200.0, dtbin=0.05*torb, dtrst=0.5*torb)
        open(os.path.join(OUT, 'inputs', 'kos%s.athinput' % name), 'w').write(txt)
        rows.append(dict(name=name, Q=Q, s0=s0, beta=bt, q=q, sg=sg, H=H, rho00=rho00, L=L,
                         nx=nx, nz=nz, lz_over_H=lz/H, dx_over_H=dx/H, omega=omega,
                         torb=torb, norb=norb, cells=nx*nx*nz))
        print('%s Q=%.1f s0=%-4g beta=%-3s q=%g sg=%-5s H=%.4f L=%.3f grid=%dx%dx%d '
              'dx/H=%.4f Lz/H=%.2f t_orb=%.3f x %d' % (name, Q, s0, bt, q, sg, H, L, nx, nx, nz,
                                                     dx/H, lz/H, torb, norb))
    json.dump(rows, open(os.path.join(OUT, 'models.json'), 'w'), indent=1)

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, 'noise'))
