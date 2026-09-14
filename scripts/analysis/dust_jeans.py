#!/usr/bin/env python3
"""The dusty Jeans instability of Krapp et al. (2024, ApJS 271, 7, sec. 3.5) for the
superparticle dust of the dust track: the linear eigenproblem, the initial condition
of the fastest-growing mode as command-line overrides for pgen dust_jeans, and the
growth-rate fit of a run's history.

Units: cs = 1, rho0 = 1, and four_pi_G chosen so that the dusty Jeans wavenumber
kJ = sqrt(4 pi G rho0 (1 + eps0))/cs = 1; k is then k/kJ and Re(s) is sigma in units
of cs kJ.  Perturbations ~ exp(i k x + s t).

  dust_jeans.py eig   --k 0.6 --eps 0.01 --ts 1 [--nx1 128 --ny 32]   # IC overrides + sigma
  dust_jeans.py eig   --k 0.6 --nspec 128 --input in.athinput --out run.athinput
                      # Krapp fig. 6: 128 species, eps0 = 0.01, ts linear in [1e-4, 10];
                      # writes a full input file (the command line cannot add absent keys)
  dust_jeans.py curve --eps 0.01 --ts 1 [--kmin --kmax]    # analytic sigma(k) table
  dust_jeans.py fit   run/jeans.user.hst [--tmin --tmax]   # measured sigma (and Im s) per column

ROTATING (the shearing box, --omega > 0): the same problem with the Coriolis and shear
terms, each component gaining an azimuthal velocity (the (3N+3) system).  Units cs =
rho0 = 1 and Omega = 1 (kappa^2 = 2(2-q) Omega^2, = 1 for q = 3/2); four_pi_G is given
explicitly (--fpg).  Closed-form anchors, all checked by `check`:
  gas only            s^2 = 4 pi G rho0 - kappa^2 - cs^2 k^2      (Chandrasekhar 1961)
  one species, gas    s [(s + 1/ts)^2 + kappa^2] = 4 pi G rho_d (s + 1/ts):
  held fixed            ts -> inf: s^2 = 4 pi G rho_d - kappa^2  (dynamical);
                        strong drag: s = 4 pi G rho_d ts/(1 + kappa^2 ts^2) (the secular
                        GI of Ward 2000 / Youdin 2005, 2011), k-INDEPENDENT in 3D
  Omega -> 0          the Krapp matrix above
  G -> 0              gas s = +-i sqrt(kappa^2 + cs^2 k^2), dust s = -1/ts +- i kappa
  dust_jeans.py check
  dust_jeans.py eig --omega 1 --fpg 2 --k 0.6 --eps 1 --ts 1 [--input .. --out ..]
  dust_jeans.py curve --omega 1 --fpg 0.5 --eps 1 --ts 1 --kmin 0.2 --kmax 3
"""
import sys, re, argparse, numpy as np

def matrix(k, eps, ts, cs=1.0, rho0=1.0, fpg=None, br=True):
    eps = np.asarray(eps, float); ts = np.asarray(ts, float); N = len(eps)
    if fpg is None: fpg = cs*cs/(1.0 + eps.sum())          # kJ = 1
    n = 2 + 2*N; M = np.zeros((n, n), complex)
    g = 1j*fpg/k                                           # -ik*phi = (i 4piG/k) * sum(drho)
    M[0, 1] = -1j*k*rho0
    M[1, 0] = -1j*k*cs*cs/rho0 + g
    for m in range(N):
        if br:
            M[1, 1] += -eps[m]/ts[m]
            M[1, 3+2*m] += eps[m]/ts[m]
        M[1, 2+2*m] += g
    for i in range(N):
        M[2+2*i, 3+2*i] = -1j*k*eps[i]*rho0
        M[3+2*i, 3+2*i] = -1.0/ts[i]
        M[3+2*i, 1] = 1.0/ts[i]
        M[3+2*i, 0] += g
        for m in range(N):
            M[3+2*i, 2+2*m] += g
    return M, fpg

def fastest(k, eps, ts, br=True, **kw):
    M, fpg = matrix(k, eps, ts, br=br, **kw)
    s, V = np.linalg.eig(M)
    j = np.argmax(s.real)
    v = V[:, j]/V[0, j]                                    # drho_g real and positive
    return s[j], v, fpg

def matrix_rot(k, eps, ts, fpg, omega=1.0, q=1.5, cs=1.0, rho0=1.0, br=True, fixed_gas=False):
    """(3N+3) system in the shearing box: [rho_g, vgx, vgy, (rho_di, vdix, vdiy)...],
    velocities peculiar (the code's convention), Coriolis 2 Omega v_y in x and
    -(2-q) Omega v_x in y.  fixed_gas freezes the gas (rows 0-2 zero): the dust-only
    anchor.  Omega = 0 is the non-rotating problem in this layout."""
    eps = np.asarray(eps, float); ts = np.asarray(ts, float); N = len(eps)
    n = 3 + 3*N; M = np.zeros((n, n), complex)
    g = 1j*fpg/k; c2 = (2.0 - q)*omega
    ir = lambda i: 3 + 3*i; ix = lambda i: 4 + 3*i; iy = lambda i: 5 + 3*i
    M[0, 1] = -1j*k*rho0
    M[1, 0] = -1j*k*cs*cs/rho0 + g; M[1, 2] = 2.0*omega
    M[2, 1] = -c2
    for m in range(N):
        M[1, ir(m)] += g
        if br:
            M[1, 1] -= eps[m]/ts[m]; M[1, ix(m)] += eps[m]/ts[m]
            M[2, 2] -= eps[m]/ts[m]; M[2, iy(m)] += eps[m]/ts[m]
    for i in range(N):
        M[ir(i), ix(i)] = -1j*k*eps[i]*rho0
        M[ix(i), 0] += g
        for m in range(N): M[ix(i), ir(m)] += g
        M[ix(i), iy(i)] = 2.0*omega; M[ix(i), ix(i)] -= 1.0/ts[i]; M[ix(i), 1] += 1.0/ts[i]
        M[iy(i), ix(i)] = -c2; M[iy(i), iy(i)] -= 1.0/ts[i]; M[iy(i), 2] += 1.0/ts[i]
    if fixed_gas: M[0:3, :] = 0.0
    return M

def fastest_rot(k, eps, ts, fpg, omega=1.0, q=1.5, br=True, fixed_gas=False, **kw):
    M = matrix_rot(k, eps, ts, fpg, omega, q, br=br, fixed_gas=fixed_gas, **kw)
    s, V = np.linalg.eig(M)
    order = np.lexsort((-s.imag, -np.round(s.real, 6)))     # largest Re (ties within 1e-6), then +Im
    j = order[0]
    v = V[:, j]
    # normalize by the gas density perturbation (amp = drho_g, as in the non-rotating runs)
    # unless it is negligible in this mode (the frozen gas; the neutral mode at the
    # back-reaction threshold), then by the largest component so that amp is the largest
    ref = v[0] if abs(v[0]) > 0.05*np.max(np.abs(v)) else v[np.argmax(np.abs(v))]
    return s[j], v/ref

def components(v, N, rot):
    """(rhog, vgx, vgy, [(rhod, vdx, vdy)]) from an eigenvector in either layout."""
    if rot:
        return v[0], v[1], v[2], [(v[3+3*i], v[4+3*i], v[5+3*i]) for i in range(N)]
    return v[0], v[1], 0.0, [(v[2+2*i], v[3+2*i], 0.0) for i in range(N)]

def cmd_check(a):
    """The closed-form anchors of the rotating matrix."""
    om, q = 1.0, 1.5; kap2 = 2.0*(2.0 - q)*om*om; ok = True
    def rep(name, got, want, tol=1e-9):
        nonlocal ok; err = abs(got - want); ok &= err < tol
        print(f"  {name:58s} got {got:+.10f}  want {want:+.10f}  |diff| {err:.1e}")
    print("gas only, s^2 = 4piG - kappa^2 - k^2 (Chandrasekhar):")
    for fpg, k in ((2.0, 0.6), (2.0, 0.95), (5.0, 1.5)):
        s, _ = fastest_rot(k, [1e-30], [1.0], fpg, om, q)
        rep(f"fpg={fpg} k={k}", s.real, np.sqrt(max(fpg - kap2 - k*k, 0.0)))
    print("one species against frozen gas, the cubic s[(s+1/t)^2+kappa^2] = A (s+1/t):")
    for fpg, eps, t in ((0.5, 1.0, 1.0), (0.5, 1.0, 0.1), (0.5, 1.0, 10.0), (3.0, 1.0, 100.0), (2.0, 0.5, 0.3)):
        A = fpg*eps
        r = np.roots([1.0, 2.0/t, 1.0/t/t + kap2 - A, -A/t]); want = max(r.real)
        s, _ = fastest_rot(0.7, [eps], [t], fpg, om, q, fixed_gas=True)
        rep(f"fpg={fpg} eps={eps} ts={t}  (secular est. {A*t/(1+kap2*t*t):.4f})", s.real, want)
    print("  ... and it is k-independent:")
    ss = [fastest_rot(k, [1.0], [1.0], 0.5, om, q, fixed_gas=True)[0].real for k in (0.2, 0.7, 3.0, 20.0)]
    rep("spread of s over k = 0.2..20", max(ss) - min(ss), 0.0)
    print("Omega -> 0 reproduces the Krapp matrix (eigenvalues, sorted):")
    for eps, ts in (([1.0], [1.0]), ([0.4, 0.3, 0.3], [0.01, 1.0, 100.0])):
        M0, fpg = matrix(0.6, eps, ts); e0 = np.sort_complex(np.linalg.eigvals(M0))
        M1 = matrix_rot(0.6, eps, ts, fpg, 0.0, q); e1 = np.linalg.eigvals(M1)
        # the rotating layout adds the N+1 decoupled v_y drag modes; every Krapp eigenvalue
        # must appear among the rotating ones
        rep(f"N={len(eps)}: max over the 2N+2 Krapp modes of min |e_rot - e_krapp|",
            max(np.min(np.abs(e1 - z)) for z in e0), 0.0)
    print("G -> 0: epicycles")
    M = matrix_rot(0.6, [1.0], [2.0], 0.0, om, q, br=False); e = np.linalg.eigvals(M)
    rep("gas  Im s = sqrt(kappa^2 + k^2)", max(e.imag), np.sqrt(kap2 + 0.36))
    rep("dust Re s = -1/ts (damped epicycle)", min(e.real), -0.5)
    rep("dust Im s = kappa", np.sort(e.imag)[-2] if len(e) > 1 else 0.0, np.sqrt(kap2))
    print("ALL OK" if ok else "FAILED")

def apply_overrides(base, out, ov):
    """Write a copy of athinput `base` to `out` with the block/key=value overrides applied
    (existing keys replaced in place, absent keys appended to their block; the command
    line cannot add absent keys, which matters for many-species runs)."""
    lines = open(base).read().split('\n'); blocks = {}; cur = None
    for i, l in enumerate(lines):
        m = re.match(r'\s*<(\w+)>', l)
        if m: cur = m.group(1); blocks.setdefault(cur, [i, i])
        elif cur is not None and l.strip() and not l.lstrip().startswith('#'): blocks[cur][1] = i
    add = {}
    for o in ov:
        bk, val = o.split('=', 1); b, key = bk.split('/')
        done = False
        if b in blocks:
            for i in range(blocks[b][0]+1, blocks[b][1]+1):
                if re.match(r'\s*'+re.escape(key)+r'\s*=', lines[i]):
                    lines[i] = f"{key} = {val}"; done = True; break
        if not done: add.setdefault(b, []).append(f"{key} = {val}")
    for b in sorted(add, key=lambda b: -blocks.get(b, [len(lines)]*2)[1]):
        if b in blocks: lines[blocks[b][1]+1:blocks[b][1]+1] = add[b]
        else: lines += ['', f'<{b}>'] + add[b]
    open(out, 'w').write('\n'.join(lines))

def cmd_eig(a):
    species(a); N = len(a.eps); rot = a.omega > 0.0
    if rot:
        if a.fpg is None: sys.exit("--fpg is required with --omega (no kJ normalization in the rotating problem)")
        s, v = fastest_rot(a.k, a.eps, a.ts, a.fpg, a.omega, a.qshear, br=not a.nobr); fpg = a.fpg
    else:
        s, v, fpg = fastest(a.k, a.eps, a.ts, br=not a.nobr)
    rhog, vgx, vgy, dust = components(v, N, rot)
    A = a.amp
    L = 2.0*np.pi/a.k; Lt = a.ny*L/a.nx1                   # cubic cells, ny = nz cells across
    ap = lambda key, z: [f"problem/{key}_amp={A*abs(z):.16g}", f"problem/{key}_phase={np.angle(z):.16g}"]
    out = [f"gravity/four_pi_G={fpg:.16g}", f"problem/kmode=1",
           f"mesh/nx1={a.nx1}", f"mesh/nx2={a.ny}", f"mesh/nx3={a.ny}",
           f"mesh/x1min={-L/2:.16g}", f"mesh/x1max={L/2:.16g}",
           f"mesh/x2min={-Lt/2:.16g}", f"mesh/x2max={Lt/2:.16g}",
           f"mesh/x3min={-Lt/2:.16g}", f"mesh/x3max={Lt/2:.16g}",
           f"problem/rho_amp={A*abs(rhog):.16g}"] + ap("vg", vgx)
    if rot: out += ap("vgy", vgy) + [f"shearing_box/omega0={a.omega:.16g}", f"shearing_box/qshear={a.qshear:.16g}"]
    for i, (rd, vx, vy) in enumerate(dust):
        out += [f"problem/rhod_amp_{i+1}={A*abs(rd):.16g}", f"problem/rhod_phase_{i+1}={np.angle(rd):.16g}",
                f"problem/vd_amp_{i+1}={A*abs(vx):.16g}", f"problem/vd_phase_{i+1}={np.angle(vx):.16g}"]
        if rot: out += [f"problem/vdy_amp_{i+1}={A*abs(vy):.16g}", f"problem/vdy_phase_{i+1}={np.angle(vy):.16g}"]
        out += [f"problem/eps_{i+1}={a.eps[i]:.16g}", f"dust/taus_{i+1}={a.ts[i]:.16g}"]
    if a.tlim > 0: tlim = a.tlim
    elif s.real > 1e-3: tlim = a.tgrow/s.real
    else: tlim = a.tgrow*2.0*np.pi/max(abs(s.imag), 1e-3)          # oscillatory: tgrow periods
    out += [f"dust/nspecies={N}", f"particles/ppc={N}", f"time/tlim={tlim:.6g}"]
    if rot:
        kap2 = 2.0*(2.0 - a.qshear)*a.omega**2
        g0 = fpg - kap2 - a.k**2
        print(f"# sigma = {s.real}  Im(s) = {s.imag}  (gas-only: {'sigma '+str(np.sqrt(g0)) if g0 > 0 else 'stable, omega '+str(np.sqrt(-g0))})", file=sys.stderr)
    else:
        print("# sigma =", s.real, " Im(s) =", s.imag, " (gas-only sigma =", np.sqrt(max(1.0/(1+sum(a.eps)) - a.k**2, 0)), ")", file=sys.stderr)
    if a.out:
        apply_overrides(a.input, a.out, out); print(a.out)
    else:
        print(" ".join(out))

def species(a):
    """--eps/--ts lists, or --nspec N: N species, eps0 split evenly, ts spaced linearly (or
    logarithmically with --ts-log) between --ts-min and --ts-max.  Krapp et al. (2024) fig. 6
    says "linear distribution from 1e-4 to 10", but its plotted curve (sigma = 0.2 at the
    gas cutoff, 0.02 at k/kJ = 1.4) is the log-uniform mixture; the linear one has a tail
    three times stronger (0.28, 0.07).  Their sigma also carries a factor sqrt(1 + eps0)."""
    if a.nspec:
        a.eps = [a.eps0/a.nspec]*a.nspec
        a.ts = list(np.logspace(np.log10(a.ts_min), np.log10(a.ts_max), a.nspec)) if a.ts_log \
               else list(np.linspace(a.ts_min, a.ts_max, a.nspec))
    return a

def cmd_curve(a):
    species(a); ks = np.linspace(a.kmin, a.kmax, a.n)
    if a.omega > 0.0:
        if a.fpg is None: sys.exit("--fpg is required with --omega")
        kap2 = 2.0*(2.0 - a.qshear)*a.omega**2
        print("# k  sigma_dust  sigma_noBR  sigma_gasonly(eps=0)  sigma_frozen_gas(dust alone)  Im(s)_dust")
        for k in ks:
            s, _ = fastest_rot(k, a.eps, a.ts, a.fpg, a.omega, a.qshear)
            s0, _ = fastest_rot(k, a.eps, a.ts, a.fpg, a.omega, a.qshear, br=False)
            sf, _ = fastest_rot(k, a.eps, a.ts, a.fpg, a.omega, a.qshear, fixed_gas=True)
            g = np.sqrt(max(a.fpg - kap2 - k*k, 0.0))
            print(f"{k:.4f}  {s.real:.6f}  {s0.real:.6f}  {g:.6f}  {sf.real:.6f}  {s.imag:+.6f}")
        return
    print("# k/kJ  sigma_dust  sigma_gasonly(no BR)  sigma_gas(eps=0)")
    for k in ks:
        s, _, _ = fastest(k, a.eps, a.ts); s0, _, _ = fastest(k, a.eps, a.ts, br=False)
        g = np.sqrt(max(1.0/(1+sum(a.eps)) - k*k, 0.0))
        print(f"{k:.4f}  {s.real:.6f}  {s0.real:.6f}  {g:.6f}")

def cmd_fit(a):
    import re
    lines = [l for l in open(a.hst) if l.startswith('#') and '[1]=' in l]
    cols = [m.group(1) for m in re.finditer(r'\[\d+\]=(\S+)', lines[0])]
    d = np.loadtxt(a.hst); t = d[:, 0]
    sel = (t >= a.tmin) & (t <= (a.tmax if a.tmax > 0 else t[-1]))
    names = [c[:-2] for c in cols if c.endswith('_c')]
    if a.auto:
        # the linear window: the longest [tmin, T] over which every column's log-amplitude
        # is a single exponential (max residual < auto) and stays below amp_max -- long
        # runs of slow modes are overtaken by other modes of the 3D box (e.g. the vertical
        # dust Jeans mode, which rotation does not stabilize, grows from round-off) or
        # saturate; the linear rate is what the early window carries
        amps = [np.hypot(d[:, cols.index(nm+'_c')], d[:, cols.index(nm+'_s')]) for nm in names]
        T = t[sel][-1]; tt = t[sel]
        for n in range(len(tt), max(int(0.15*len(tt)), 8), -1):
            w = sel & (t <= tt[n-1]); good = True
            for amp in amps:
                if np.any(amp[w] > a.amp_max): good = False; break
                y = np.log(np.maximum(amp[w], 1e-300)); pf = np.polyfit(t[w], y, 1)
                if np.max(np.abs(y - np.polyval(pf, t[w]))) > a.auto: good = False; break
            if good: T = tt[n-1]; break
        sel = sel & (t <= T)
        print(f"# linear window t = {t[sel][0]:.2f}..{T:.2f} (auto: residual < {a.auto}, amp < {a.amp_max})")
    for nm in names:
        ic = cols.index(nm+'_c'); isn = cols.index(nm+'_s')
        amp = np.sqrt(d[:, ic]**2 + d[:, isn]**2)
        ok = sel & (amp > 0)
        p = np.polyfit(t[ok], np.log(amp[ok]), 1)
        # the mode |v| e^{Re s t} cos(kx + arg v + Im s t): c = |.| cos theta, s = -|.| sin theta
        th = np.unwrap(np.arctan2(-d[ok, isn], d[ok, ic])); pw = np.polyfit(t[ok], th, 1)
        print(f"{nm:10s} sigma = {p[0]:.6f}   Im(s) = {pw[0]:+.6f}   (amp {amp[ok][0]:.3e} -> {amp[ok][-1]:.3e} over t {t[ok][0]:.2f}-{t[ok][-1]:.2f})")

p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='cmd', required=True)
def spec_args(q):
    q.add_argument('--eps', type=float, nargs='+', default=[0.01]); q.add_argument('--ts', type=float, nargs='+', default=[1.0])
    q.add_argument('--nspec', type=int, default=0, help='N species: eps0 split evenly, ts linear in [ts-min, ts-max]')
    q.add_argument('--eps0', type=float, default=0.01); q.add_argument('--ts-min', type=float, default=1e-4); q.add_argument('--ts-max', type=float, default=10.0); q.add_argument('--ts-log', action='store_true')
def rot_args(q):
    q.add_argument('--omega', type=float, default=0.0, help='> 0: the shearing box (Omega = 1 is the unit)'); q.add_argument('--qshear', type=float, default=1.5)
    q.add_argument('--fpg', type=float, default=None, help='four_pi_G (required with --omega; default kJ = 1 otherwise)')
q = sub.add_parser('check'); q.set_defaults(f=cmd_check)
q = sub.add_parser('eig'); q.add_argument('--k', type=float, required=True); spec_args(q); rot_args(q); q.add_argument('--tlim', type=float, default=-1.0); q.add_argument('--amp', type=float, default=1e-4); q.add_argument('--tgrow', type=float, default=6.0, help='run for tgrow e-foldings'); q.add_argument('--nobr', action='store_true')
q.add_argument('--nx1', type=int, default=128, help='cells per wavelength'); q.add_argument('--ny', type=int, default=32, help='transverse cells (cubic cells)')
q.add_argument('--input', default=None, help='base athinput'); q.add_argument('--out', default=None, help='write the overrides into a copy of --input (prints its path)'); q.set_defaults(f=cmd_eig)
q = sub.add_parser('curve'); spec_args(q); rot_args(q); q.add_argument('--kmin', type=float, default=0.05); q.add_argument('--kmax', type=float, default=1.3); q.add_argument('--n', type=int, default=51); q.set_defaults(f=cmd_curve)
q = sub.add_parser('fit'); q.add_argument('hst'); q.add_argument('--tmin', type=float, default=0.0); q.add_argument('--tmax', type=float, default=-1.0); q.add_argument('--auto', type=float, default=0.0, help='> 0: restrict to the linear window, max log-amplitude residual (e.g. 0.02)'); q.add_argument('--amp-max', type=float, default=0.02, help='with --auto: drop data once any amplitude exceeds this'); q.set_defaults(f=cmd_fit)
a = p.parse_args(); a.f(a)
