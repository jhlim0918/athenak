# Quadratic level-jump ghosts for the multigrid self-gravity solver (`mg_fc_interp`)

Branch `mg-fc-quadratic` (from `fargo-amr-mhd` at 4214ca95). New option
`<gravity> mg_fc_interp = linear | quadratic` (default `linear`, bit-identical to 4214ca95;
diagnostic third value `quadratic_all_levels`, sec. 1 "Which MG levels"). Commits: ddae4ca6
(quadratic ghosts), f61441dc (3-D test input), 64a9e22e (quadratic on the finest MG level
only), plus this notes commit.
Run directories, patched inputs and scripts: `validation/run/mg_fc_quadratic/` (the scripts,
inputs and a README are force-added; the logs stay local, every number is copied below).

## 1. What changed and why

At a face between two refinement levels the 7-point operator of each block reads one ghost
cell. `MultigridBoundaryValues::ProlongateFCMG` fills both sides:

* fine block, face neighbour COARSER: ghost from the coarse value `c` across the face
  (received into `coarse_buf_`) and the first fine interior cell `f1`;
* coarse block, face neighbour FINER: ghost from its first interior cell `c1` and the 2x2
  face average `A1` of the fine block's first layer (packed by `FillCoarseMG`, received into
  the coarse block's own ghost cell).

The linear pair `g_f = (2c~ + f1)/3`, `g_c = (4 A1 - c1)/3` is flux-matched (the coarse face
flux equals the sum of the four fine fluxes, so the composite operator is conservative), but
both are linear interpolants: ghost errors O(h^2), flux errors O(h), force at the interface
first order (6.0e-4 ... 3.0e-5 on R. Li's test, root 16 ... 256).

### 1-D normal direction (face at x = 0; fine side x > 0, cell h; coarse side x < 0, cell 2h)

Nodes: coarse c1 at -h, fine f1 at h/2, f2 at 3h/2; fine ghost at -h/2, coarse ghost at +h.

* fine ghost, quadratic Lagrange through (-h, h/2, 3h/2) at -h/2:
  `g_f = (8/15) c~ + (2/3) f1 - (1/5) f2` (weights 8/15, 2/3, -1/5; sum 1).
* 7-point fluxes: fine face `(f1 - g_f)/h` (area h^2), coarse face `(g_c - c1)/(2h)`
  (area 4h^2). Conservation: `g_c - c1 = 2 mean_4(f1 - g_f)`. With the fine ghost above and
  `mean_4 c~ = c1 + (d2_a + d2_b)/32` (see below):
  `g_c = -(1/15) c1 + (2/3) A1 + (2/5) A2 - (1/30)(d2_a + d2_b)`,
  `A2` = 2x2 average of the SECOND fine layer. This is also the quadratic Lagrange
  interpolant through (-h, h/2, 3h/2) evaluated at +h (weights -1/15, 2/3, 2/5), with the
  O(h^2) transverse bias of the face averages removed: a 2x2 average equals the point value
  at the coarse face-cell centre line + (h^2/8)(phi_aa + phi_bb); (2/3 + 2/5)(h^2/8) =
  (2/15) h^2, and `d2 = 4 h^2 phi_aa + O(h^4)`, so the `-(1/30) d2` term cancels it exactly
  for quadratics.

### Transverse fit on the fine side (value c~ at the fine ghost's transverse position)

`c~` is the coarse layer across the face evaluated at the quarter-coarse-cell offsets
(xi, eta) = (+-1/4, +-1/4) (coarse units) from a total-degree-2 fit on a 3x3 patch:

    c~ = c + xi Da c + eta Db c + (xi^2/2) d2_a + (eta^2/2) d2_b + xi eta m,
    Da = slope of the quadratic through the 3 cells at the centre cell (3-point derivative),
    d2 = w0 - 2 w1 + w2 (the three consecutive cells), m = Da Db c (mixed term).

The mixed term averages out over the four ghosts but is needed pointwise (without it the
ghost error is O(h^2) at every cell where phi_ab != 0). The received patch is half + ngh
cells per transverse direction, the halo on the side of the parent's interior only (the
`icoar` receive extents; the coarser-neighbour slot (f1,f2) is this block's child parity),
so the 3-cell stencils are centred except at the outer edge of the received data, which is
the coarse neighbour's transverse block edge: there they are one-sided (cells t, t+1, t+2 or
t-2, t-1, t). With 3-point one-sided derivatives the fit is still exact for quadratics.
Mean over the four ghosts: `c + (d2_a + d2_b)/32` (slopes and m cancel by symmetry, for any
stencil).

### Coarse side and exact conservation

The coarse ghost needs d2 of ITS interior layer c1 with exactly the stencils the fine side
used for that cell; the rule "centred, one-sided (3 cells inward) at this block's
transverse edges" reproduces it, because the fine side is one-sided exactly at the coarse
neighbour's block edges (fine parity 0 at the lower edge, parity 1 at the upper edge) and
centred everywhere else (including the interior seam between the two fine children). The
coarse block computes it from its own interior data at the same exchange, so the discrete
flux match holds to round-off at every V-cycle exchange (checked with random data in
`validation/run/mg_fc_quadratic/check_formulas.py`: max |(g_c - c1) - 2 mean(f1 - g_f)| =
8.9e-16; ghost errors for a smooth non-polynomial phi fall 8x per halving of h for the
centred and all one-sided stencil positions).

### Data, buffers and messages: none changed

The coarse ghost uses A1 and A2 only in the combination `B = (2/3) A1 + (2/5) A2`, so in
quadratic mode `FillCoarseMG` packs B into the very `coarse_buf_` ghost slot that carried A1
(the slot the nface == 1 `icoar` send reads). The coarse block receives B where it received
A1 and evaluates `g_c = -(1/15) c1 + B - (1/30)(d2_a + d2_b)` in place. No change to
`coarse_buf_` size, index tables, `icoar`/`ifine` extents, the rank-packed metadata, MPI
message sizes, or the same-rank path (which uses the same pack/unpack kernels). The fine side
needs nothing new either: c~ uses the already-received halo (including the transverse corner
cell, which is in the received product box), f2 is its own second interior cell.

Alternatives considered (task statement):
(a) sending A1 and A2 separately would need a second normal layer in the `icoar` face send,
    an A2 store on the coarse side (mg ngh = 1 leaves no ghost layer for it) and larger
    messages; (b) the c2 variant (second COARSE layer to the fine side: fine ghost
    -(1/14) c~2 + (5/6) c~1 + (5/21) f1) would need a second normal coarse layer in
    `coarse_buf_` (which has only ngh ghost layers), larger `ifine` sends/`icoar` receives,
    and full transverse fits on both coarse layers. Option "A2 packed as B" needs neither.

### Which MG levels use it

The discrete answer is set by the finest MG level of the MeshBlocks (each block at its own
resolution); the coarser block levels, the octets and the root grid only drive convergence.
`mg_fc_interp = quadratic` therefore applies the quadratic pair on the FINEST MG level only and
keeps the symmetric linear pair on the coarser block levels (`Multigrid::FCQuadratic(shift,
ncells)`: shift == 0 and >= 4 cells per block). First implemented on every block level with
>= 4 cells (ddae4ca6; still available as `mg_fc_interp = quadratic_all_levels`): same converged
answer (cube N=64 identical to 10 digits), but an irregular period-2 contraction (up to 0.45
per V-cycle on the N=128 cube, sec. 2.4); finest-only restores the linear operator's
contraction exactly. With 4^3 blocks the two are identical (the 2-cell level is linear anyway).

### Fallbacks and invariants

* MG levels with fewer than 4 cells per block (half < 2: not 3 transverse coarse cells) keep
  the linear pair on BOTH sides (both checks use the level's cell count, which is the same
  for every block of an MG level, so the two sides always agree; likewise the shift). These
  levels (and the octet/root levels, `ProlongateOctetBoundariesFluxCons`, unchanged) only
  affect convergence.
* Edge/corner neighbours (injection) unchanged: the 7-point operator reads face ghosts only.
* `mg_fc_interp = quadratic` requires `mg_nghost = 1` (fatal error otherwise). An unknown value
  is a fatal error.
* Shearing box: the x1-boundary blocks share one level (MGGravityDriver check), so no level
  jump crosses the shear-periodic faces; slab-open x3: x3-face blocks are root level
  (`CheckSlabBlockLevels`). Both sides of every level jump only read interior data of the two
  blocks involved (the coarse side never reads its own ghosts: one-sided at its edges), so
  neither BC interacts with the new stencils.
* After the solve `RetrieveResult` copies the ghosts into `pgrav->phi`; the gas force is the
  centred difference through them (SourceTerms::AddSelfGravity), so it becomes second order
  with no further change.
* The composite operator becomes non-symmetric (fine ghost reads f2, coarse ghost reads the
  transverse d2). The red-black SOR smoother in the FAS V-cycle converges as before (sec. 2.4).

Code: `src/multigrid/multigrid_bvals.cpp` (FillCoarseMG face loop; ProlongateFCMG both
branches; helpers `FCQuadCoarseAtGhosts`, `FCQuadStencilStart`), `src/multigrid/multigrid.hpp`
(`Multigrid::SetFCInterp/GetFCInterp/FCQuadratic`), `src/gravity/mg_gravity.cpp` (input
parameter, read in the MGGravityDriver constructor).

## 2. Results

Binaries: `build-before/athena_4214ca95` (copy of the starting commit's MPI build),
`build-mpi/src/athena` (this branch, MPI), `build-fft/src/athena` (this branch, MPI +
kokkos-fft, needed for the slab-open test). 4 MPI ranks unless stated. Numbers in
parentheses = log2 of the ratio to the previous row (convergence order).
Command-line overrides cannot ADD a key, so the run directory holds copies of the test
inputs with `mg_fc_interp` (and `show_defect`) inserted: `mgiw_{lin,quad}.athinput`,
`shwave/shwave_{lin,quad}.athinput`, `slab/slab_{lin,quad}.athinput`,
`swing/swing_{ring,patch}_quad.athinput`, `swing_amr/swing_amr_{line,quad}.athinput`,
`swing_mhd/mhd_ring_{line,quad}.athinput`, `cube/cube2lev.athinput` (adds a level-2 region).

### 2.1 R. Li's test as a slab one block thick (mg_interface_wave, refined square [0.25,0.75]^2)

    cd validation/run/mg_fc_quadratic
    ./ladder.sh BIN TAG NP 4 [overrides]   # 4^3 blocks, N = 16..256, x3max = refined x3max = 4/N
    ./ladder.sh BIN TAG NP 8 [overrides]   # 8^3 blocks, nx3 = 8, x3max = 8/N, N = 32..256
    INP=mgiw_quad.athinput ./ladder.sh ../../../build-mpi/src/athena quad 4 4
    (gravity/threshold=-1 gravity/niteration=40, FMG; table.py TAG MB N-list prints the tables)

4^3 blocks, LINEAR (identical output lines to 4214ca95 at every N):

| N | interface force linf | interface force L2 | force linf | force rel L2 | phi linf | phi rel L2 |
|---|---|---|---|---|---|---|
| 16 | 6.009e-04 | 3.521e-04 | 6.081e-04 | 7.079e-02 | 1.490e-05 | 1.478e-02 |
| 32 | 2.600e-04 (1.21) | 1.006e-04 (1.81) | 2.600e-04 (1.23) | 1.812e-02 (1.97) | 4.353e-06 (1.78) | 3.764e-03 (1.97) |
| 64 | 1.243e-04 (1.06) | 3.355e-05 (1.58) | 1.243e-04 (1.06) | 4.616e-03 (1.97) | 1.115e-06 (1.96) | 9.425e-04 (2.00) |
| 128 | 6.122e-05 (1.02) | 1.380e-05 (1.28) | 6.122e-05 (1.02) | 1.189e-03 (1.96) | 2.808e-07 (1.99) | 2.357e-04 (2.00) |
| 256 | 3.048e-05 (1.01) | 6.464e-06 (1.09) | 3.048e-05 (1.01) | 3.135e-04 (1.92) | 7.074e-08 (1.99) | 5.895e-05 (2.00) |

4^3 blocks, QUADRATIC:

| N | interface force linf | interface force L2 | force linf | force rel L2 | phi linf | phi rel L2 |
|---|---|---|---|---|---|---|
| 16 | 5.768e-04 | 3.458e-04 | 5.768e-04 | 7.020e-02 | 8.151e-06 | 1.163e-02 |
| 32 | 1.521e-04 (1.92) | 8.806e-05 (1.97) | 1.521e-04 (1.92) | 1.785e-02 (1.98) | 1.924e-06 (2.08) | 2.777e-03 (2.07) |
| 64 | 3.929e-05 (1.95) | 2.196e-05 (2.00) | 3.929e-05 (1.95) | 4.479e-03 (2.00) | 4.598e-07 (2.06) | 6.803e-04 (2.03) |
| 128 | 9.865e-06 (1.99) | 5.476e-06 (2.00) | 9.865e-06 (1.99) | 1.121e-03 (2.00) | 1.153e-07 (1.99) | 1.683e-04 (2.01) |
| 256 | 2.493e-06 (1.98) | 1.367e-06 (2.00) | 2.493e-06 (1.98) | 2.802e-04 (2.00) | 2.940e-08 (1.97) | 4.189e-05 (2.01) |

The interface force falls x3.79, 3.87, 3.98, 3.96 per doubling (0.035% of the peak force
7.12e-3 at 256; linear 0.43%). R. Li's composite solver: 2.7e-4 -> 1.2e-6 (16 -> 256); ours
5.8e-4 -> 2.5e-6 (linf), 3.5e-4 -> 1.4e-6 (interface L2): the same factor (~230) over the
four doublings; the absolute level differs with the error measure (our interface L2 at 256,
1.37e-6, is close to his 1.2e-6). What remains at the interface is the
ordinary O(h^2) error of the coarse level: the uniform ROOT grid (refinement=none, identical
before/quadratic) has force linf 5.614e-04, 1.444e-04, 3.635e-05, 9.100e-06, 2.275e-06 and rel L2
8.008e-02, 2.035e-02, 5.109e-03, 1.279e-03, 3.197e-04 (order 2.00); the refined mesh's global
force error is now below the uniform root grid's at every N (2.80e-4 vs 3.20e-4 at 256).

8^3 blocks (nx3 = 8):

| N | linear interface linf | quadratic interface linf | quad interface L2 | quad force rel L2 | quad phi linf | quad phi rel L2 |
|---|---|---|---|---|---|---|
| 32 | 1.876e-04 | 1.512e-04 | 8.776e-05 | 1.785e-02 | 1.855e-06 | 2.766e-03 |
| 64 | 1.242e-04 (0.60) | 3.900e-05 (1.96) | 2.192e-05 (2.00) | 4.481e-03 (1.99) | 4.603e-07 (2.01) | 6.772e-04 (2.03) |
| 128 | 5.958e-05 (1.06) | 9.764e-06 (2.00) | 5.461e-06 (2.00) | 1.121e-03 (2.00) | 1.154e-07 (2.00) | 1.679e-04 (2.01) |
| 256 | 3.045e-05 (0.97) | 2.468e-06 (1.98) | 1.362e-06 (2.00) | 2.803e-04 (2.00) | 2.888e-08 (2.00) | 4.182e-05 (2.01) |

(linear 8^3 output lines identical to 4214ca95 at every N. Final code, quadratic on the finest
MG level; `quadratic_all_levels` agrees to 3-4 digits here because 40 V-cycles do not fully
converge the thin slab's slow mode, sec. 2.4: e.g. 256 interface linf 2.468041e-06 vs
2.468238e-06.)

### 2.2 Genuinely 3-D: refined cube [0.25,0.75]^3 in a periodic unit cube, wave (2,1,1)

`inputs/tests/mg_interface_wave_cube.athinput` (committed f61441dc; 8^3 blocks, 30 FMG
V-cycles, mg_fc_interp = quadratic in the file):

    cd validation/run/mg_fc_quadratic/cube
    ./run_cube.sh BIN TAG MB {linear|quadratic} "N list" [overrides];  python3 table_cube.py TAG MB N,N,..

| blocks | N | blocks | linear interface linf | quadratic interface linf | quad interface L2 | quad force rel L2 | quad phi linf | quad phi rel L2 |
|---|---|---|---|---|---|---|---|---|
| 8^3 | 32 | 120 | 4.566e-04 | 1.427e-04 | 8.275e-05 | 1.988e-02 | 9.943e-07 | 4.697e-04 |
| 8^3 | 64 | 960 | 2.234e-04 (1.03) | 3.600e-05 (1.99) | 2.060e-05 (2.01) | 4.994e-03 (1.99) | 2.034e-07 (2.29) | 1.074e-04 (2.13) |
| 8^3 | 128 | 7680 | 1.096e-04 (1.03) | 9.104e-06 (1.98) | 5.129e-06 (2.01) | 1.250e-03 (2.00) | 4.528e-08 (2.17) | 2.629e-05 (2.03) |
| 16^3 | 64 | 120 | 2.234e-04 | 3.599e-05 | 2.058e-05 | 4.995e-03 | 1.987e-07 | 1.071e-04 |
| 16^3 | 128 | 960 | 1.095e-04 (1.03) | 9.093e-06 (1.98) | 5.120e-06 (2.01) | 1.250e-03 (2.00) | 4.499e-08 (2.14) | 2.627e-05 (2.03) |

Linear global rel L2 / phi linf (8^3): 2.004e-02, 5.072e-03, 1.286e-03 / 4.741e-06, 1.197e-06,
3.040e-07. Uniform root grid (refinement=none, 8^3): force linf 1.374e-04, 3.455e-05, 8.648e-06,
rel L2 2.116e-02, 5.318e-03, 1.331e-03 (for this wave the O(h^2) errors of the 7-point operator
and of the cell-averaged source cancel in phi, which converges at 4th order on the uniform grid:
rel L2 8.3e-6, 5.2e-7, 3.2e-8; the refined mesh's phi error is the level jump's, 2nd order).

Nested levels (`cube/cube2lev.athinput` = the cube input + a level-2 cube [0.375,0.625]^3,
8^3 blocks; 176 / 1408 / 11264 blocks):

| N | linear interface linf | quadratic interface linf | quad interface L2 | quad force rel L2 | quad phi linf |
|---|---|---|---|---|---|
| 32 | 4.567e-04 | 1.427e-04 | 7.834e-05 | 1.987e-02 | 9.963e-07 |
| 64 | 2.234e-04 | 3.599e-05 (1.99) | 1.951e-05 (2.01) | 4.991e-03 (1.99) | 2.081e-07 (2.26) |
| 128 | 1.096e-04 | 9.102e-06 (1.98) | 4.856e-06 (2.01) | 1.249e-03 (2.00) | 4.635e-08 (2.17) |

### 2.3 Bit-identity, MPI, FMG

All checks repeated with the final code (64a9e22e) unless noted.

* Linear (default) vs 4214ca95, identical: all `# MG_INTERFACE_WAVE` lines (slab 4^3 N=16..256,
  8^3 N=32..256, refined cube 8^3 N=32..128); the six shwave lines of sec. 2.5; the refined
  swing ring history (`swing_selfgrav_smr.athinput`, output1/data_format=%.15e, 4 ranks, to
  t = 8): `cmp` identical (swing/before_ring vs swing/lin_ring and swing/linF_ring), peak
  d_cos/A = -3.584414.
* Uniform meshes, quadratic vs 4214ca95: identical (slab N=16..256, cube N=32..128).
* Quadratic, 1 vs 4 ranks: slab 4^3 N=16..256: all 10 printed digits equal except the last
  digit of phi_linf / force linf / phi rel L2 at N=128, 256 (reduction order, 1e-10 relative);
  cube N=64: identical.
* FMG + 30 V-cycles vs no FMG + 60 V-cycles vs threshold = 0 (stagnation exit), quadratic cube
  N=64: identical to 10 digits (interface linf 3.5996920600e-05). Slab N=64 quadratic, 400
  V-cycles, FMG vs no FMG: agree to 10 digits (3.9291421242e-05 vs ...240e-05); the 40-cycle
  ladder value 3.929140145e-05 is converged to 6 digits.

### 2.4 V-cycle contraction (defect L2 per V-cycle, `show_defect = 2`)

`python3 defsum.py LOG...`: geometric-mean defect ratio over V-cycles 2..11 (min, max), the
round-off floor and the first cycle within 2x of it. 30 FMG V-cycles, 4 ranks
(logs cube/log_defF_*, cube/log_2lev_defF_*, shwave/log_defF_*, log_defF_slab8_*).

| problem | linear | quadratic (finest level) | quadratic_all_levels |
|---|---|---|---|
| cube N=64, 8^3 blocks | 0.188 (0.15-0.20), floor 8.5e-11 at 13 | 0.193 (0.18-0.20), 8.4e-11 at 13 | 0.229 (0.18-0.25), 8.4e-11 at 15 |
| cube N=128, 8^3 blocks | 0.233 (0.16-0.29), 5.1e-10 at 15 | 0.228 (0.14-0.28), 5.0e-10 at 15 | 0.268 (0.14-0.45), 5.0e-10 at 16 |
| nested-level cube N=64 | 0.223 (0.18-0.24), 3.9e-10 at 14 | 0.229 (0.17-0.26), 3.8e-10 at 15 | 0.210 (0.12-0.42), 3.8e-10 at 14 |
| shwave SMR N=64, time0=0.37 (shear, 16^3) | 0.184 (0.17-0.20), 5.2e-12 at 14 | 0.186 (0.18-0.19), 5.2e-12 at 15 | 0.188 (0.16-0.23), 5.2e-12 at 15 |
| slab 8^3 blocks N=128 (60 cycles) | 0.493, then 0.9399 | 0.465, then 0.9399 | 0.468, then 0.9399 |
| slab 4^3 blocks N=64 (30 cycles) | 0.477 (0.34-0.66), then 0.940 | 0.489 (0.32-0.77), then 0.940 | = quadratic |

The non-symmetric composite operator converges with the existing red-black SOR (omega 1.15)
and FAS V-cycle; no damping is needed. With quadratic ghosts on the finest level only, the
contraction is that of the linear operator and round-off is reached in the same number of
cycles (+-1). Quadratic ghosts on the coarser block levels too give a period-2 alternation
(the red-black ordering alternates between V-cycles, coffset_ ^= 1) and 1-2 more cycles.
The slow asymptotic mode of the one-block-thick slab (0.94 at N=64, both modes, identical) is
pre-existing and geometric, not caused by the ghosts: uniform slab 0.05-0.07; slab fully
refined (no level jump) 0.07; 2 blocks thick (nx3 = 8 with 4^3 blocks) 0.75; 4 blocks thick
0.51 (round-off 4e-11 by ~30 cycles); 20x more coarsest-grid sweeps (temporary build, not
committed) only 0.94 -> 0.92 (logs: run/mg_fc_quadratic/slowmode). With 40 FMG cycles the
slab ladder values are converged to ~6 digits at N=64 (400-cycle runs) and to 3-4 digits at
N=256 with 8^3 blocks (sec. 2.1): enough for the tables, not for bit-level comparisons.

### 2.5 Regressions in quadratic mode

Refined swing (FMG + 10 V-cycles per step, 4 ranks, t = 8), peak d_cos/A (uniform 256^2
reference -3.618801):

| run | linear | quadratic | quadratic_all_levels |
|---|---|---|---|
| ring (`swing_selfgrav_smr.athinput`) | -3.584414 (= 4214ca95, hst identical) | -3.584413 | -3.584413 |
| patch (`swing_selfgrav_smr_ypatch.athinput`) | -3.533085 | -3.533080 | -3.533079 |

(dirs swing/{before,lin,linF,quad,quadall}_ring, swing/{lin,quad,quadall}_patch; the
quadratic inputs are swing/swing_{ring,patch}_quad.athinput, the all-levels runs used the
same inputs with the ddae4ca6 binary.) max |d_cos(quad) - d_cos(lin)|/A over the history:
ring 4.6e-6 (t = 7.55), patch 1.1e-5 (t = 7.61); phase lock max|d_sin|/A 1.25e-13 (quad) vs
1.35e-13 (lin). The swing peak is set by the hydro resolution blend (Phase 2a: the gravity-off
control deviates more than the gravity-on run), so the second-order interface force moves it
by 1e-6 only: the remaining 1% (ring) / 2.4% (patch) gap to the uniform 256^2 reference is not
a gravity effect. AMR moving ring (`swing_selfgrav_amr.athinput`, regrids every 5 cycles,
to t = 2, swing_amr/): both modes run without warnings; max |difference|/A = 4.2e-7;
d_cos/A(t=2) = -0.562672 in both.

MHD ring swing with gravity (`validation/run/fargo_amr/mhd/swing/mesh_smr.athinput`, b0 = 1,
orbital advection, ring |x| < 0.78, to t = 6.3, 4 ranks, output9/dt=100; copies
swing_mhd/mhd_ring_{line,quad}.athinput): d_cos/A minimum -0.708151 (linear) / -0.708150
(quadratic) at t = 4.84, maximum 1.972546 / 1.972544 at t = 5.86, max |quad - lin|/A =
1.9e-6; max |div B| 1.71e-13 in both. Same benign input warnings in both logs.

Phase-2a shwave SMR (`mg_poisson_shear_smr.athinput`, profile=shwave, N/4-cell blocks, fixed
30 V-cycles, `shwave/run_shwave.sh BIN TAG shwave_{lin,quad}.athinput`), max_rel / l2_rel
against the root-grid discrete solution (quadratic and quadratic_all_levels identical to all
printed digits):

| N | time0 | linear max_rel | quadratic max_rel | linear l2_rel | quadratic l2_rel |
|---|---|---|---|---|---|
| 32 | 0 | 3.352876e-03 | 2.642658e-03 | 3.728380e-03 | 3.422997e-03 |
| 64 | 0 | 8.191375e-04 | 6.667282e-04 | 9.265186e-04 | 8.672227e-04 |
| 128 | 0 | 2.025278e-04 | 1.674184e-04 | 2.310491e-04 | 2.181603e-04 |
| 32 | 0.37 | 6.565953e-03 | 4.832882e-03 | 7.133657e-03 | 6.290987e-03 |
| 64 | 0.37 | 1.611402e-03 | 1.220058e-03 | 1.781766e-03 | 1.600534e-03 |
| 128 | 0.37 | 4.021719e-04 | 3.064099e-04 | 4.451544e-04 | 4.036451e-04 |

(linear = 4214ca95 to all digits.) Both second order; quadratic 18-26% lower max_rel. Since the
reference is the root-grid discrete solution, this test cannot show the interface order by
itself (the fine region is more accurate than the reference).

Slab-open sheet SMR (`mg_poisson_slab_smr.athinput`, FFT build, mesh N^3 with N/4-cell blocks,
sheet_k = N/2, threshold = 0 as in the input), max_rel (l2_rel):

    cd validation/run/mg_fc_quadratic/slab
    mpirun -np 4 ../../../../build-fft/src/athena -i slab_{lin,quad}.athinput mesh/nx1=N mesh/nx2=N \
      mesh/nx3=N meshblock/nx1=N/4 meshblock/nx2=N/4 meshblock/nx3=N/4 problem/sheet_k=N/2 \
      problem/rho0={1.0,0.0}

| N | linear rho0=1 | quadratic rho0=1 | linear rho0=0 | quadratic rho0=0 |
|---|---|---|---|---|
| 32 | 1.840012e-01 (3.353e-01) | 1.637163e-02 (8.063e-03) | 1.639711e-02 | 1.637163e-02 |
| 64 | 8.727037e-02 (1.654e-01) | 4.562643e-03 (2.080e-03) | 4.568265e-03 | 4.562643e-03 |
| 128 | 4.240683e-02 (8.242e-02) | 1.196617e-03 (5.242e-04) | 1.197976e-03 | 1.196617e-03 |

With the uniform background the linear ghosts leave a horizontally uniform offset
(3/4) 4piG rho0 h_f^2 in the refined slab (the k_perp = 0 sector, phi ~ z^2); the quadratic
ghosts are exact for a quadratic phi, so the offset vanishes: quadratic rho0=1 equals the pure
mode sector (rho0=0) to all printed digits and converges at 2nd order (1.84, 1.93). 19x lower
max_rel at 64^3. Residual norms max_rel_all 4.5e-12 (quad) / 4.9e-12 (lin).

## 3. Open issues / risks

* The composite operator is non-symmetric in quadratic mode. Convergence is fine in every test
  here (sec. 2.4), but a solver change that assumes symmetry (e.g. a CG wrapper) would need care.
* The one-block-thick slab geometry converges slowly asymptotically in BOTH modes (sec. 2.4);
  pre-existing, unrelated to the ghosts; thin-slab runs need ~30-40 V-cycles for 7 digits.
* MG levels with 2 cells per block use the linear pair (convergence only). Octet and root levels
  keep their own linear flux-conserving ghosts (convergence only).
* `mg_nghost > 1` is refused in quadratic mode (the linear code fills only the inner ghost layer
  at level jumps anyway).
* Not tested: dust self-gravity runs (the potential comes from the same solver, so they
  inherit the change), restarts (no state is added), GPU builds (the new code is plain Kokkos
  inside the existing per-block kernels, with fixed-size local arrays; the fine-side branch
  adds a 3x3 gather per coarse face cell, negligible next to the smoother).
* The swing tests barely move (1e-6 to 1e-5 of A): there the refined/uniform difference is
  dominated by the hydro, so the dynamic regressions show safety, not benefit. The benefit is
  in the force at level jumps themselves (sec. 2.1-2.2) and in the k_perp = 0 sector
  (slab sheet, 19x).

## 4. Recommendation

Use `mg_fc_interp = quadratic` for runs where the self-gravity force at level jumps matters
(collapse/fragmentation with AMR, dust self-gravity on refined meshes): the interface force
becomes second order (x4 per doubling) at no cost in V-cycles or communication. The default
stays `linear` for bit-for-bit continuity of existing runs; switching the default is a one-line
change in `src/gravity/mg_gravity.cpp` once the user's production setups have been compared.
