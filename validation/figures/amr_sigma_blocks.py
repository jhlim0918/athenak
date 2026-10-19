"""Face-on Sigma_p/Sigma_g0 of an AMR dust_dpm bin dump on its finest grid (coarser cells
fill r x r pixels; columns summed over z with each block's own dz), and the outline of every
refined MeshBlock projected on x-y.  Writes <out>.map (float64 nf x nf, x fastest, then a
header line in <out>.txt: t/P nf) and <out>.blocks (x0 x1 y0 y1 level per refined block).
  python amr_sigma_blocks.py <dump.bin> <out_prefix>"""
import sys, types, numpy as np
sys.modules["h5py"] = types.ModuleType("h5py")
sys.path.insert(0, "/Users/jaysmac/athenak-worktrees/si-strat-amr/vis/python")
import bin_convert as bc
f, out = sys.argv[1], sys.argv[2]
d = bc.read_binary(f); g = np.array(d["mb_geometry"]); lg = np.array(d["mb_logical"])
q = np.array(d["mb_data"]["dustdpm"]); n = d["nx1_mb"]
x0, x1 = d["x1min"], d["x1max"]; L = x1 - x0
lmin, lmax = lg[:, 3].min(), lg[:, 3].max()
nroot = int(round(L/(g[np.argmin(lg[:, 3]), 1] - g[np.argmin(lg[:, 3]), 0])))
nf = nroot*n*2**(lmax - lmin)
S = np.zeros((nf, nf))
for b in range(d["n_mbs"]):
    r = 2**(lmax - lg[b, 3]); dz = (g[b, 5] - g[b, 4])/n
    col = q[b].sum(axis=0)*dz/np.sqrt(2*np.pi)                 # (j, i)
    i0 = int(round((g[b, 0] - x0)/L*nf)); j0 = int(round((g[b, 2] - x0)/L*nf))
    S[j0:j0+n*r, i0:i0+n*r] += np.repeat(np.repeat(col, r, 0), r, 1)
S.astype("<f8").tofile(out + ".map")
open(out + ".txt", "w").write(f"{d['time']/(2*np.pi):.6f} {nf} {x0} {x1}\n")
with open(out + ".blocks", "w") as fb:
    for b in range(d["n_mbs"]):
        if lg[b, 3] > lmin:
            fb.write(f"{g[b,0]} {g[b,1]} {g[b,2]} {g[b,3]} {lg[b,3]-lmin}\n")
print(f"{f}: t = {d['time']/(2*np.pi):.2f} P, {d['n_mbs']} MeshBlocks, {int((lg[:,3]>lmin).sum())} refined, map {nf}^2")
