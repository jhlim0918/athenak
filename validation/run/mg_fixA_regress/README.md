# Regression of the multigrid fine-ghost fix (commit d86c16ac), 2026-11-02
athena_before = build of 0583ec1a, athena_after = with the fix; run.sh runs every case into <case>_<before|after>/.
- swing_unif (uniform 128^2, MG gravity): user.hst bit-identical.
- swing_ring / swing_patch / swing_amr: peak d_cos/A -3.583481 -> -3.584414, -3.532422 -> -3.533085,
  -3.510989 -> -3.511909 (uniform 256^2: -3.618801; validation/run/fargo_amr/swing/unif256).
- mhd_ring (mesh_smr.athinput, B0 = 1, to 6.3): c_y vs theory (t <= 6) 2.66% -> 2.65%.
- poisson_slab_smr: sheet max_rel 8.726856e-2 -> 8.727037e-2. poisson_shear_smr (profile=modes): its residual
  metric is O(1) on any refined mesh (naive per-block Laplacian), unchanged.
- shwave_smr/: the Phase-2a shwave ladder (8^3/16^3/32^3 blocks at 32/64/128^3, qomt 0 and t0 = 0.37, 30 V-cycles);
  before reproduces multigrid_sb_validation sec. 2a to every digit; after: interface peak 2nd order.
- MPI: swing ring 4 ranks vs 1 rank differ by 1.0e-12 (before) and 4.8e-13 (after) -- pre-existing round-off;
  refined Poisson N = 64, 1 vs 4 ranks: identical to all printed digits.
