#!/usr/bin/env python3
"""Switch an AthenaK shearing-box restart file between orbital advection ON (FARGO) and OFF.

  rst_orbital_advection.py <src.rst> <out.rst> off|on

With orbital advection the stored gas momenta are perturbations about the background shear
v_sh = -q Omega x; without it they are the full momenta.  This rewrites the hydro data of
every MeshBlock (any refinement level; ghosts included, at their own x) and sets
<shearing_box> orbital_advection in the file's parameter block:
    off:  M2 += rho v_sh,   E += M2_rel v_sh + rho v_sh^2/2
    on :  M2 -= rho v_sh,   E -= M2_full v_sh - rho v_sh^2/2
(isothermal files have no energy).  Particles are untouched: their velocities are
shear-relative in both modes.  Hydro-only restarts (no MHD/radiation/z4c/turbulence)."""
import sys, re, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
from rst_refine2 import read_rst

def main():
    src, dst, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    assert mode in ("on", "off")
    S = read_rst(src)
    text = S["text"]
    def param(block, key):
        m = re.search(r"<%s>(.*?)(\n<|\Z)" % block, text, re.S)
        v = re.search(r"\n\s*%s\s*=\s*([^\s#]+)" % key, m.group(1))
        return v.group(1) if v else None
    q = float(param("shearing_box", "qshear")); om = float(param("shearing_box", "omega0"))
    oa = (param("shearing_box", "orbital_advection") or "true").lower() in ("true", "1")
    if (mode == "off") != oa:
        raise SystemExit(f"{src} already has orbital_advection = {oa}")
    ms = S["msize"]; x0, x1 = ms[0], ms[3]
    ng = int(S["bind"][0]); n1 = int(S["bind"][1]); N1 = int(S["mind"][1])
    nroot1 = N1//n1
    hyd = S["hyd"].copy(); ideal = S["nhyd"] == 5
    for m in range(S["nmb"]):
        lx1, lev = int(S["lloc"][m, 0]), int(S["lloc"][m, 3])
        w = (x1 - x0)/(nroot1 << (lev - S["root"]))          # block width at its level
        dx = w/n1; xb = x0 + lx1*w
        x = xb + (np.arange(n1 + 2*ng) - ng + 0.5)*dx           # cell centres incl. ghosts
        vsh = (-q*om*x)[None, None, :]
        rho = hyd[m, 0]
        t = vsh if mode == "off" else -vsh                      # velocity added to v_y
        if ideal:
            hyd[m, 4] += hyd[m, 2]*t + 0.5*rho*t*t
        hyd[m, 2] += rho*t
    # parameter block: set orbital_advection in <shearing_box>
    val = "false" if mode == "off" else "true"
    lines = text.split("\n"); blk = None; done = False; out = []
    for l in lines:
        st = l.strip()
        if st.startswith("<") and st.endswith(">"):
            if blk == "<shearing_box>" and not done:
                out.insert(len(out) - (1 if out and out[-1].strip() == "" else 0),
                           f"orbital_advection = {val}"); done = True
            blk = st
        if blk == "<shearing_box>" and st.split("=")[0].strip() == "orbital_advection":
            l = f"orbital_advection = {val}"; done = True
        out.append(l)
    assert done, "no <shearing_box> block"
    newtext = "\n".join(out).encode()
    raw = S["raw"]; ho = S["hyd_off"]
    with open(dst, "wb") as f:
        f.write(newtext); f.write(raw[S["text_len"]:ho])
        f.write(hyd.astype(np.float64).tobytes()); f.write(raw[ho + hyd.nbytes:])
    print(f"{src} -> {dst}: orbital_advection {str(oa).lower()} -> {val}, {S['nmb']} MeshBlocks, "
          f"{'ideal' if ideal else 'isothermal'}, q Omega = {q*om}")

if __name__ == "__main__":
    main()
