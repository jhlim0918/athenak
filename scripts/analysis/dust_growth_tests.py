#!/usr/bin/env python3
"""Checks of the dust growth test suite D3 (scripts/analysis/dust_growth_tests.sh).

  python3 scripts/analysis/dust_growth_tests.py [validation/run/dust_growth]

(a) growth = none: every output of the new binary is byte-identical to the parent's.
(b) uniform box: St_mean(t) vs S(t) = [sqrt(S0) + 0.5 sqrt(pi/8) rho_d sqrt(3 alpha) t]^2
    at three timesteps (first-order operator split: error ~ dt).
(c) fragmentation: St_mean saturates at S_frag = v_frag^2/(3 alpha c_s^2).
(d) drift: first-cycle d t_s/dt of each species vs sqrt(pi/8) rho_d |dv_NSH,s|/2, with
    dv_NSH from the multi-species NSH solution.
(e) restart (bitwise) and AMR with particle splitting (AMR vs uniform to round-off).
"""
import filecmp, glob, os, re, sys
import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else 'validation/run/dust_growth'
COEF = np.sqrt(np.pi/8.0)
# shared by every dust_nsh input (inputs/dust/dust_nsh*.athinput)
NSH = dict(taus=np.array([0.01, 0.1, 1.0]), eps=np.array([0.2, 0.5, 0.3]), etavk=0.05,
           q=1.5, omega=1.0)


def read_phst(fn):
    """Columns by name; repeated header lines (restarts) are skipped."""
    cols = None
    rows = []
    with open(fn) as f:
        for l in f:
            if l.startswith('#'):
                if '[1]=' in l and cols is None:
                    cols = [m.group(1) for m in re.finditer(r'\[\d+\]=(\S+)', l)]
                continue
            rows.append([float(x) for x in l.split()])
    d = np.array(rows)
    return {c: d[:, i] for i, c in enumerate(cols)}


def read_pvtk(fn):
    """AthenaK particle VTK (float or double, big-endian): positions, every SCALARS block
    and VECTORS pvel."""
    with open(fn, 'rb') as f:
        data = f.read()
    head = data[:400].decode('latin1')
    t = float(re.search(r'time=\s*([\d.eE+-]+)', head).group(1))
    i = data.find(b'POINTS')
    j = data.find(b'\n', i)
    tok = data[i:j].split()
    npts = int(tok[1])
    dt = '>f8' if tok[2] == b'double' else '>f4'
    sz = np.dtype(dt).itemsize
    out = {'time': t, 'xyz': np.frombuffer(data, dt, 3*npts, j+1).reshape(npts, 3)}
    for m in re.finditer(rb'SCALARS (\S+) \S+\nLOOKUP_TABLE default\n', data):
        out[m.group(1).decode()] = np.frombuffer(data, dt, npts, m.end()).astype(float)
    i = data.find(b'VECTORS pvel')
    if i >= 0:
        j = data.find(b'\n', i) + 1
        out['vel'] = np.frombuffer(data, dt, 3*npts, j).reshape(npts, 3)
    return out


def nsh_dv(taus, eps, etavk, q, omega):
    """|v_s - u| of each species in the multi-species NSH equilibrium (gas forcing
    a_x = 2 Omega etavk; velocities relative to the shear)."""
    n = len(taus)
    a = np.zeros((2*n+2, 2*n+2))
    b = np.zeros(2*n+2)
    # unknowns: u_x, u_y, then (v_x,s, v_y,s)
    a[0, 1] = 2*omega
    b[0] = -2*omega*etavk
    a[1, 0] = -(2-q)*omega
    for s in range(n):
        ix, iy = 2+2*s, 3+2*s
        a[0, ix] += eps[s]/taus[s]; a[0, 0] -= eps[s]/taus[s]
        a[1, iy] += eps[s]/taus[s]; a[1, 1] -= eps[s]/taus[s]
        a[ix, iy] = 2*omega; a[ix, ix] = -1/taus[s]; a[ix, 0] = 1/taus[s]
        a[iy, ix] = -(2-q)*omega; a[iy, iy] = -1/taus[s]; a[iy, 1] = 1/taus[s]
    x = np.linalg.solve(a, b)
    return np.array([np.hypot(x[2+2*s]-x[0], x[3+2*s]-x[1]) for s in range(n)])


def same_output(a, b):
    """Byte comparison; restart files are compared after their parameter block (ending
    at <par_end>), which echoes every parameter read, including the new defaulted
    <dust>/growth = none."""
    if not a.endswith('.rst'):
        return filecmp.cmp(a, b, shallow=False)
    da, db = open(a, 'rb').read(), open(b, 'rb').read()
    ka, kb = da.find(b'<par_end>'), db.find(b'<par_end>')
    return ka > 0 and kb > 0 and da[ka:] == db[kb:]


def test_a():
    print('(a) growth = none vs parent commit: byte comparison of all outputs '
          '(restart files past the parameter echo)')
    ok = True
    for case in ('nsh2d', 'amr3d', 'yj14'):
        base, new = f'{RUN}/a/base/{case}', f'{RUN}/a/new/{case}'
        files = sorted(os.path.relpath(p, base) for p in glob.glob(f'{base}/**/*',
                       recursive=True) if os.path.isfile(p) and not p.endswith('log.txt'))
        same = [f for f in files if same_output(f'{base}/{f}', f'{new}/{f}')]
        diff = sorted(set(files) - set(same))
        ok &= not diff
        print(f'    {case:6s}: {len(same)}/{len(files)} files identical'
              + (f'; DIFFER: {diff[:5]}' if diff else ''))
    print('    PASS' if ok else '    FAIL')
    return ok


def analytic_b(t, s0=1e-3, rhod=1.0, alpha=1e-3):
    return (np.sqrt(s0) + 0.5*COEF*rhod*np.sqrt(3*alpha)*t)**2


def test_b():
    print('(b) uniform growth vs S(t) = [sqrt(S0) + 0.5 sqrt(pi/8) rho_d sqrt(3 alpha) t]^2')
    errs = []
    for c in ('0.45', '0.225', '0.1125'):
        h = read_phst(f'{RUN}/b/cfl{c}/dgrow_box.phst')
        ex = analytic_b(h['time'])
        rel = h['St_mean']/ex - 1
        spread = np.max((h['St_max'] - h['St_min'])/h['St_mean'])
        errs.append(np.max(np.abs(rel)))
        print(f'    cfl {c:6s}: S(t={h["time"][-1]:.1f}) = {h["St_mean"][-1]:.6f} '
              f'(exact {ex[-1]:.6f}); max |rel err| = {errs[-1]:.3e}; '
              f'max particle spread = {spread:.1e}')
    order = np.log2(np.array(errs[:-1])/np.array(errs[1:]))
    print(f'    convergence order (halving dt): {order.round(3)}')
    ok = errs[0] < 5e-3 and np.all(np.abs(order - 1) < 0.1)
    print('    PASS' if ok else '    FAIL')
    return ok


def test_c(alpha=1e-3, vf=0.01):
    sfrag = vf**2/(3*alpha)
    print(f'(c) fragmentation cap S_frag = v_f^2/(3 alpha) = {sfrag:.6f}')
    h = read_phst(f'{RUN}/c/dgrow_box.phst')
    late = h['time'] > 12
    s = h['St_mean'][late]
    print(f'    t > 12: St_mean in [{s.min():.6f}, {s.max():.6f}] '
          f'(rel. to S_frag: {s.min()/sfrag-1:+.2e} .. {s.max()/sfrag-1:+.2e}); '
          f'St_max overall = {h["St_max"].max():.6f}')
    t_reach = h['time'][np.argmax(h['St_mean'] >= 0.99*sfrag)]
    t_exp = (np.sqrt(0.99*sfrag) - np.sqrt(1e-3))/(0.5*COEF*np.sqrt(3*alpha))
    print(f'    reaches 0.99 S_frag at t = {t_reach:.2f} (growth law: {t_exp:.2f})')
    ok = np.max(np.abs(s/sfrag - 1)) < 0.01
    print('    PASS' if ok else '    FAIL')
    return ok


def test_d():
    print('(d) first-cycle growth rate vs the NSH drift |v_s - u|/2 (alpha_t = 0)')
    dv = nsh_dv(**NSH)
    rhod = NSH['eps'].sum()
    expect = COEF*rhod*0.5*dv
    ok = True
    for case in ('rz2d', 'sb3d', 'sb3d_nofargo'):
        files = sorted(glob.glob(f'{RUN}/d/{case}/pvtk/*.vtk'))
        p0, p1 = read_pvtk(files[0]), read_pvtk(files[1])
        o0, o1 = np.argsort(p0['ptag']), np.argsort(p1['ptag'])
        dt = p1['time'] - p0['time']
        rate = (p1['ptaus'][o1] - p0['ptaus'][o0])/dt
        sp = p0['pspecies'][o0].astype(int)
        line = []
        for s in range(3):
            r = rate[sp == s]
            err = np.max(np.abs(r/expect[s] - 1))
            ok &= err < 1e-9
            line.append(f's{s+1}: {r.mean():.10e} (rel {err:.1e})')
        print(f'    {case:12s} ' + '; '.join(line))
    print(f'    expected     ' + '; '.join(f's{s+1}: {expect[s]:.10e}' for s in range(3)))
    print('    PASS' if ok else '    FAIL')
    return ok


def test_e():
    print('(e) restart and AMR')
    f_full = sorted(glob.glob(f'{RUN}/e/full/pvtk/*.vtk'))
    last = os.path.basename(f_full[-1])
    same = filecmp.cmp(f_full[-1], f'{RUN}/e/rst/pvtk/{last}', shallow=False)
    pf, pr = read_pvtk(f_full[-1]), read_pvtk(f'{RUN}/e/rst/pvtk/{last}')
    dts = np.max(np.abs(np.sort(pf['ptaus']) - np.sort(pr['ptaus'])))
    print(f'    restart at t~2 -> t={pf["time"]:.1f}: final pvtk byte-identical: {same}; '
          f'max |dt_s| = {dts:.1e}; S range [{pf["ptaus"].min():.5f}, '
          f'{pf["ptaus"].max():.5f}]')
    ok = same
    ha = read_phst(f'{RUN}/e/amr/nsh3d_amr.phst')
    log = open(f'{RUN}/e/amr/log.txt').read()
    done = 'DUST_GROWTH_SUMMARY' in log
    print(f'    AMR run (refine t=1, derefine t=4, split_on_refine, 4 ranks) completed: '
          f'{done}; npart {ha["npart"][0]:.0f} -> {ha["npart"].max():.0f}; '
          f'St_mean {ha["St_mean"][0]:.4f} -> {ha["St_mean"][-1]:.4f}')
    hs = read_phst(f'{RUN}/e/split/nsh3d_amr.phst')
    # max/min must be bitwise; the mass-weighted columns are sums over 2.75x as many
    # particles of 1/8 the mass after the split, so they move by summation round-off
    exact = all(np.all(hs[c] == hs[c][0]) for c in ('St_max', 'St_min'))
    wdev = max(np.max(np.abs(hs[c]/hs[c][0] - 1)) for c in
               ('St_mean', 'fm_St>0.01', 'fm_St>0.03', 'fm_St>0.1'))
    const = exact and wdev < 1e-11   # ~ N eps for N ~ 3e5 serial additions
    split = hs['npart'].max() > hs['npart'][0]
    t_split = hs['time'][np.argmax(hs['npart'] > hs['npart'][0])]
    print(f'    split check (restart t={hs["time"][0]:.3f}, zero growth rate -> t='
          f'{hs["time"][-1]:.2f}): npart {hs["npart"][0]:.0f} -> {hs["npart"].max():.0f} '
          f'at t={t_split:.2f}; St_max/St_min bitwise constant: {exact}; mass-weighted '
          f'columns within {wdev:.1e} (St_min {hs["St_min"][0]:.12f} vs species value '
          f'0.01)')
    ok &= done and const and split and hs['St_min'][0] > 0.0101
    print('    PASS' if ok else '    FAIL')
    return ok


if __name__ == '__main__':
    results = {}
    for name, fn in (('a', test_a), ('b', test_b), ('c', test_c), ('d', test_d),
                     ('e', test_e)):
        try:
            results[name] = fn()
        except Exception as exc:  # report and continue with the other tests
            print(f'    ERROR: {exc!r}')
            results[name] = False
    print('SUMMARY: ' + ' '.join(f'{k}={"PASS" if v else "FAIL"}' for k, v in results.items()))
