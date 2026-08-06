#!/usr/bin/env python3
"""Figures for the multigrid/refinement-track document (Phase 0a/0b).

Reads AthenaK .bin outputs from ../run/bin and history files from ../run;
writes PNGs into this directory. Needs only numpy + matplotlib.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "run"

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#1a1a19", "#52514e"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight",
})


def read_bin(filename):
    """AthenaK .bin reader (v1.1). Returns per-block data + logical locations,
    so refined meshes can be drawn block by block."""
    fp = open(filename, "rb")
    fp.seek(0, 2); filesize = fp.tell(); fp.seek(0, 0)
    assert fp.readline().split()[0] == b"Athena"
    npre = int(fp.readline().split(b"=")[-1])
    ph = {}
    for _ in range(npre - 1):
        k, v = [x.strip() for x in fp.readline().decode().split("=")]
        ph[k] = v
    locsize = int(ph["size of location"]); varsize = int(ph["size of variable"])
    nvars = int(fp.readline().split(b"=")[-1])
    var_list = [v.decode() for v in fp.readline().split()[1:]]
    hsize = int(fp.readline().split(b"=")[-1])
    header = [ln.decode().split("#")[0].strip() for ln in fp.read(hsize).split(b"\n")]
    header = [ln for ln in header if ln]

    def get(block, key):
        blk = "<none>"
        for ln in header:
            if ln.startswith("<"):
                blk = ln; continue
            k, v = ln.split("=")
            if blk == f"<{block}>" and k.strip() == key:
                return v.strip()
        raise KeyError(f"{block}/{key}")

    nghost = int(get("mesh", "nghost"))
    mesh = {k: float(get("mesh", k)) for k in
            ("x1min", "x1max", "x2min", "x2max", "x3min", "x3max")}
    locfmt = np.float64 if locsize == 8 else np.float32
    varfmt = np.float64 if varsize == 8 else np.float32
    blocks = []
    while fp.tell() < filesize:
        idx = np.frombuffer(fp.read(24), dtype=np.int32) - nghost
        n1 = idx[1]-idx[0]+1; n2 = idx[3]-idx[2]+1; n3 = idx[5]-idx[4]+1
        logical = np.frombuffer(fp.read(16), dtype=np.int32).copy()   # lx1,lx2,lx3,level
        geom = np.frombuffer(fp.read(6*locsize), dtype=locfmt).copy()  # x1min..x3max
        data = np.fromfile(fp, dtype=varfmt, count=n1*n2*n3*nvars)
        blocks.append({"logical": logical, "geom": geom,
                       "data": data.reshape(nvars, n3, n2, n1)})
    fp.close()
    return {"blocks": blocks, "vars": var_list, "mesh": mesh}


def draw_blocks(ax, fd, title):
    """Draw the x1-x2 MeshBlock layout, colored by refinement level."""
    lev_colors = {0: "#ffffff", 1: "#dbe9fb", 2: "#b9d5f6"}
    for b in fd["blocks"]:
        lev = int(b["logical"][3])
        g = b["geom"]
        if abs(g[4] - fd["mesh"]["x3min"]) > 1e-12:   # one z-layer only
            continue
        ax.add_patch(Rectangle((g[0], g[2]), g[1]-g[0], g[3]-g[2],
                               facecolor=lev_colors.get(lev, "#8fbdf0"),
                               edgecolor=INK, lw=0.9))
    m = fd["mesh"]
    for xb in (m["x1min"], m["x1max"]):
        ax.axvline(xb, color=C2, lw=2.5, zorder=5)
    ax.set_xlim(m["x1min"], m["x1max"]); ax.set_ylim(m["x2min"], m["x2max"])
    ax.set_aspect("equal"); ax.set_xlabel("$x$ (radial)"); ax.set_title(title, fontsize=10)
    ax.grid(False)


# ------------------------------------------------- fig 1: mesh layouts (policy)
fd_patch = read_bin(sorted((RUN/"bin").glob("shwave2_smr.hydro_w.*.bin"))[0])
fd_ring = read_bin(sorted((RUN/"bin").glob("ring_fargo.hydro_w.*.bin"))[0])

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.7))
draw_blocks(axes[0], fd_patch, "patch refinement (FARGO must be off)")
draw_blocks(axes[1], fd_ring, "annular refinement (FARGO works per level)")
axes[0].set_ylabel("$y$ (azimuthal)")
fig.savefig(HERE/"mg_mesh_policy.png")
plt.close(fig)


print("wrote:", [p.name for p in sorted(HERE.glob("mg_*.png"))])
