# Multigrid force at a level jump: R. Li's self_gravity_wave poisson test on our solver (2026-11-02)
pgen src/pgen/tests/mg_interface_wave.cpp (port of the poisson mode of athenak-dust amrSG_dev
src/pgen/tests/self_gravity_wave.cpp): rho = 1 + 0.1 S cos(2 pi (2x + y) + 0.37) on [0,1]^2, 4 pi G = 1,
level-1 square [0.25,0.75]^2, 4-cell blocks, root N = 16..256. Our multigrid needs cubic 3-D blocks, so the
2-D problem is a periodic slab one 4^3 block thick (x3 in [0, 4/N]); the solution does not depend on x3.
Force = the gas source term's centred difference of phi through the solver-filled ghosts. Interface cells =
first/last active cell on a block face with a neighbour on another level (as in self_gravity_wave).
    mpirun -np P $ATH -i base.athinput mesh/nx1=N mesh/nx2=N mesh/x3max=4/N refined_region1/x3max=4/N \
        [mesh_refinement/refinement=none]          -> log_amr_N, log_uni_N ("# MG_INTERFACE_WAVE {...}")
Converged: gravity/threshold=-1 niteration 30 vs 80 agree to 7 digits (log_fix*). threshold=0 runs end with
"Failed to converge after 40 iterations" AFTER the measurement (stagnation exit at round-off defect).

## Fix A (2026-11-02, UNCOMMITTED in src/multigrid/multigrid_bvals.cpp, ProlongateFCMG coarse-face branches)
Cause: the transverse slope of the coarse value for the fine ghost was clamped one-sided at both ends of the
coarse patch with the centred weight 0.125, i.e. half the right offset at clamped cells (every cell with
4-cell blocks). The received coarse buffer already holds ngh halo cells on the side of the parent's interior
(icoar recv extents; the coarser-neighbour slot (f1,f2) = this block's child parity), so: centred slope where
the halo exists, one-sided with weight 1/4 per cell only at the genuine outer edge. Uniform meshes untouched.
Logs log_fixA_N (4^3 blocks), log_fixA_mb8_{amr,uni}_N (8^3 blocks, nx3 = 8, x3max = 8/N):
  interface force linf (4^3): 6.0e-4, 2.6e-4, 1.2e-4, 6.1e-5, 3.0e-5 (8.5% -> 0.43% of peak), ~2x per doubling
  (before 1.0e-3 ... 7.0e-4 flat); phi_linf 2nd order; global acc rel L2 3.1e-4 at 256 = uniform 3.2e-4.
  8^3 blocks: 0.43% at 256, ~2x per doubling; refined mesh below uniform globally at every N.
Remaining gap to R. Li's 1.2e-6 (2nd order): linear normal interpolation of the fine/coarse ghosts (plan B:
quadratic + flux-matched coarse ghost).
