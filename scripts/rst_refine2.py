#!/usr/bin/env python3
"""Refine an AthenaK restart file by a factor 2 in every direction (uniform mesh, hydro +
dust particles, no MHD/radiation/z4c/turbulence): the 64^3 -> 128^3 restart of a
stratified streaming-instability run, written for ANY number of MPI ranks.

  rst_refine2.py <src.rst> <template.rst> <out.rst> <nranks>

<src.rst>       the restart to refine (uniform mesh, MeshBlocks of nx^3 cells, any ranks)
<template.rst>  a restart of the SAME problem at twice the resolution with the same MeshBlock
                size, from AthenaK itself (e.g. the stage-1 input with mesh/nx* doubled and
                time/nlim=0): it supplies the MeshBlock list in AthenaK's gid order and the
                binary header structs of the refined mesh; its data are not used
<nranks>        the rank count the refined restart will be run with

Gas: the conserved variables (ghosts included in the file) are prolongated with the
minmod-limited linear scheme of AthenaK's ProlongCC (child averages equal the parent:
conservative); ghost cells are filled periodically (the code refills them by its
boundary conditions before the first step).  Particles: every particle becomes 2^3
children at +-dx_old/4 in each direction with mass/8, the same split as
Particles::SplitOnRefine (positions and the IPX1/IPY1/IPZ1 registers shifted, all else
copied); a child that would leave the box in x (shear-periodic) takes the opposite
offset instead; y and z wrap periodically.  Child tags are 8*tag + c (unique); the
sampling level PLEV (if stored) becomes the refined mesh's level.
Time, cycle and the output counters are kept; dt is halved (CFL at the finer cells).
The input-parameter block is the source's with <mesh> nx1..3 doubled."""
import sys, struct, numpy as np

REAL = np.float64
IPX, IPY, IPZ, IPX1, IPY1, IPZ1, IPM = 0, 2, 4, 6, 8, 10, 16
PGID, PTAG = 0, 1

def read_rst(fname, nranks_hint=None):
    b = open(fname, "rb").read()
    k = b.index(b"<par_end>"); k = b.index(b"\n", k) + 1
    text = b[:k].decode()
    o = k
    nmb, root = struct.unpack_from("<ii", b, o); o += 8
    msize = np.frombuffer(b, REAL, 9, o).copy(); o += 72
    mind = np.frombuffer(b, np.int32, 19, o).copy(); o += 76
    bind = np.frombuffer(b, np.int32, 19, o).copy(); o += 76
    time, dt = struct.unpack_from("<dd", b, o); o += 16
    ncyc, = struct.unpack_from("<i", b, o); o += 4
    lloc = np.frombuffer(b, np.int32, 4*nmb, o).reshape(nmb, 4).copy(); o += 16*nmb
    cost = np.frombuffer(b, np.float32, nmb, o).copy(); o += 4*nmb
    ng, nx1, nx2, nx3 = bind[0], bind[1], bind[2], bind[3]
    nout = (nx1 + 2*ng, nx2 + 2*ng, nx3 + 2*ng)
    # particle counts: nranks ints, then (nrd, nid), then size_t data_size; find nranks
    for nr in ([nranks_hint] if nranks_hint else range(1, 65537)):
        oo = o + 4*nr
        nrd, nid = struct.unpack_from("<ii", b, oo)
        ds, = struct.unpack_from("<Q", b, oo + 8)
        nhyd = ds//(8*nout[0]*nout[1]*nout[2]) if ds % (8*nout[0]*nout[1]*nout[2]) == 0 else 0
        if nrd == 17 and nid in (3, 4) and nhyd in (4, 5):
            break
    else:
        raise RuntimeError("could not locate the particle header")
    cnt = np.frombuffer(b, np.int32, nr, o).copy(); o = oo + 8 + 8
    hyd_off = o
    hyd = np.frombuffer(b, REAL, nmb*nhyd*nout[2]*nout[1]*nout[0], o).reshape(
        nmb, nhyd, nout[2], nout[1], nout[0]).copy()
    o += hyd.nbytes
    pc = nrd + nid; parts = []
    for r in range(nr):
        n = int(cnt[r]); a = np.frombuffer(b, REAL, pc*n, o).reshape(pc, n).copy()
        parts.append(a); o += a.nbytes
    assert o == len(b), f"{fname}: {len(b) - o} trailing bytes"
    return dict(text=text, nmb=nmb, root=root, msize=msize, mind=mind, bind=bind, time=time,
                dt=dt, ncyc=ncyc, lloc=lloc, cost=cost, cnt=cnt, nrd=nrd, nid=nid,
                nhyd=nhyd, ds=ds, hyd=hyd, parts=np.concatenate(parts, axis=1), nout=nout,
                hyd_off=hyd_off, text_len=k, raw=b)

def load_balance(cost, nranks):          # Mesh::LoadBalance, ported
    nb = len(cost); rlist = np.zeros(nb, int); j = nranks - 1
    total = float(np.sum(cost, dtype=np.float32)); target = total/nranks; my = 0.0
    for i in range(nb - 1, -1, -1):
        my += float(cost[i]); rlist[i] = j
        if my >= target and j > 0:
            j -= 1; total -= my; my = 0.0; target = total/(j + 1)
    return rlist

def prolong(u):   # u (nv, nz, ny, nx) periodic -> (nv, 2nz, 2ny, 2nx), minmod linear
    out = np.repeat(np.repeat(np.repeat(u, 2, 1), 2, 2), 2, 3)
    for ax in (1, 2, 3):
        dl = u - np.roll(u, 1, ax); dr = np.roll(u, -1, ax) - u
        s = np.where(dl*dr > 0, np.sign(dl)*np.minimum(np.abs(dl), np.abs(dr)), 0.0)
        s2 = np.repeat(np.repeat(np.repeat(s, 2, 1), 2, 2), 2, 3)
        idx = np.arange(out.shape[ax]) % 2           # 0: left child, 1: right child
        shape = [1, 1, 1, 1]; shape[ax] = out.shape[ax]
        out += s2*(0.25*(2*idx - 1)).reshape(shape)
    return out

def main():
    src, tpl, dst, nranks = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    S = read_rst(src); T = read_rst(tpl)
    ng = int(S["bind"][0]); n1, n2, n3 = (int(x) for x in S["bind"][1:4])
    assert (n1, n2, n3) == tuple(int(x) for x in T["bind"][1:4]) and ng == int(T["bind"][0])
    N1, N2, N3 = (int(x) for x in S["mind"][1:4])
    assert tuple(int(x) for x in T["mind"][1:4]) == (2*N1, 2*N2, 2*N3)
    assert np.all(S["lloc"][:, 3] == S["root"]) and np.all(T["lloc"][:, 3] == T["root"]), \
        "uniform meshes only"
    # 1. global interior fields of the source
    nv = S["nhyd"]; G = np.zeros((nv, N3, N2, N1))
    for m in range(S["nmb"]):
        lx1, lx2, lx3 = S["lloc"][m, :3]
        G[:, lx3*n3:(lx3+1)*n3, lx2*n2:(lx2+1)*n2, lx1*n1:(lx1+1)*n1] = \
            S["hyd"][m, :, ng:ng+n3, ng:ng+n2, ng:ng+n1]
    F = prolong(G)
    # 2. blocks of the refined mesh in the template's gid order, ghosts periodic
    Fp = np.pad(F, ((0, 0), (ng, ng), (ng, ng), (ng, ng)), mode="wrap")
    H = np.zeros((T["nmb"], nv, n3 + 2*ng, n2 + 2*ng, n1 + 2*ng))
    for m in range(T["nmb"]):
        lx1, lx2, lx3 = T["lloc"][m, :3]
        H[m] = Fp[:, lx3*n3:lx3*n3 + n3 + 2*ng, lx2*n2:lx2*n2 + n2 + 2*ng,
                  lx1*n1:lx1*n1 + n1 + 2*ng]
    # 3. particles: split 2^3, new gid by position, grouped by the new ranks
    p = S["parts"]; nrd = S["nrd"]; nid = S["nid"]
    ms = S["msize"]; x0 = ms[0:3]; x1 = ms[3:6]; L = x1 - x0; dxo = L/np.array([N1, N2, N3])
    kids = []
    for c in range(8):
        q = p.copy(); off = [((c >> d) & 1)*2 - 1 for d in range(3)]
        for d, (ip, ir) in enumerate(((IPX, IPX1), (IPY, IPY1), (IPZ, IPZ1))):
            sh = off[d]*0.25*dxo[d]*np.ones(q.shape[1])
            if d == 0:                                # x: shear-periodic -> stay inside
                out = (q[ip] + sh < x0[0]) | (q[ip] + sh >= x1[0]); sh[out] *= -1
            q[ip] += sh; q[ir] += sh
            if d > 0:
                q[ip] = x0[d] + np.mod(q[ip] - x0[d], L[d])
        q[IPM] /= 8.0
        q[nrd + PTAG] = 8*np.rint(p[nrd + PTAG]) + c
        kids.append(q)
    q = np.concatenate(kids, axis=1)
    nb = np.array([2*N1//n1, 2*N2//n2, 2*N3//n3])
    ijk = np.floor((q[[IPX, IPY, IPZ]].T - x0)/(L/nb)).astype(int)
    ijk = np.clip(ijk, 0, nb - 1)
    gid_of = {tuple(T["lloc"][m, :3]): m for m in range(T["nmb"])}
    gid = np.array([gid_of[tuple(t)] for t in ijk])
    q[nrd + PGID] = gid
    if nid >= 4: q[nrd + 3] = T["root"]          # PLEV: the sampling level is the new level
    rlist = load_balance(T["cost"], nranks)
    prank = rlist[gid]; order = np.argsort(prank, kind="stable"); q = q[:, order]
    cnt = np.bincount(prank, minlength=nranks).astype(np.int32)
    # 4. header text: the source's, <mesh> nx1..3 doubled
    lines = S["text"].split("\n"); blk = None
    for i, l in enumerate(lines):
        s = l.strip()
        if s.startswith("<") and s.endswith(">"): blk = s
        elif blk == "<mesh>" and s.split("=")[0].strip() in ("nx1", "nx2", "nx3"):
            key = s.split("=")[0].strip(); val = int(s.split("=")[1].split("#")[0])
            lines[i] = f"{key} = {2*val}"
    text = "\n".join(lines)
    # 5. write
    with open(dst, "wb") as f:
        f.write(text.encode())
        f.write(struct.pack("<ii", T["nmb"], T["root"]))
        f.write(T["msize"].astype(REAL).tobytes()); f.write(T["mind"].tobytes()); f.write(T["bind"].tobytes())
        f.write(struct.pack("<dd", S["time"], 0.5*S["dt"])); f.write(struct.pack("<i", S["ncyc"]))
        f.write(T["lloc"].tobytes()); f.write(T["cost"].tobytes())
        f.write(cnt.tobytes()); f.write(struct.pack("<ii", nrd, nid))
        f.write(struct.pack("<Q", T["ds"]))
        f.write(H.astype(REAL).tobytes())
        # step 5: one block per rank, each variable-major over THAT rank's particles
        starts = np.concatenate([[0], np.cumsum(cnt)])
        for r in range(nranks):
            f.write(np.ascontiguousarray(q[:, starts[r]:starts[r+1]]).astype(REAL).tobytes())
    m0 = S["parts"][IPM].sum(); m1 = q[IPM].sum()
    print(f"{src} ({N1}x{N2}x{N3}, {S['nmb']} MBs, {len(S['cnt'])} ranks, {S['parts'].shape[1]} particles)")
    print(f"  -> {dst} ({2*N1}x{2*N2}x{2*N3}, {T['nmb']} MBs, {nranks} ranks, {q.shape[1]} particles; "
          f"{cnt.min()}-{cnt.max()} per rank)")
    print(f"  gas mass ratio {F[0].sum()/8/G[0].sum():.16f}, dust mass ratio {m1/m0:.16f}, "
          f"t = {S['time']:.10g}, dt = {0.5*S['dt']:.6g}, cycle {S['ncyc']}")

if __name__ == "__main__":
    main()
