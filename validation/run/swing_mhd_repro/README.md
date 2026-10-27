# Refined-mesh field error in MHD shearing boxes: minimal reproducer (2026-10-27)

## The defect
In the magnetized swing test (validation PDF, Phase 4, "Magnetized swing amplification"),
refined shearing boxes integrated **without orbital advection** get the azimuthal field
perturbation c_y wrong by 10-20% while the uniform twins follow linear theory to ~3%.
div B stays at round-off.

## Reproducer
`inputs/tests/swing_mhd_repro.athinput`, run as a pair (as is, and with
`mesh_refinement/refinement=none`). Each member takes 1-3 min on one core.
- Thin 3-D shearing box: 64 x 64 x 4 cells, 8 x 8 x 4 MeshBlocks.
- Ideal MHD, adiabatic c_s = 1, HLLD + PLM + RK2.
- No gravity; no orbital advection.
- Uniform B0 = 1 along y.
- A leading shearing wave: kx0 = -6, ky = 1, amp 1e-4.
- A static level-1 ring over |x| < pi/4.

`python3 repro.py NJOBS [variant ...]` runs pairs and scores
`dev = max_t |X_ring - X_uni| / max_t |X_uni|` over t <= 6. For deviations against linear
theory, use `../swing_mhd/swing_ode.py` with the variant's B0, q and Omega, and dx = 2 pi/64.

## Results (64-cell root, t <= 6)

| variant | ring vs uniform, c_y | uniform vs theory, c_y | ring vs theory, c_y | note |
|---|---|---|---|---|
| base (HLLD, PLM, RK2) | 0.22 | 0.075 | 0.226 | the defect |
| b0 = 0 | -- (delta: 0.03) | | | no field: none |
| b0 = 0.5 | 0.11 | 0.041 | 0.142 | scales with B0 |
| HLLE / LLF | 0.19 / 0.16 | 0.27 / 0.27 | 0.36 / 0.34 | not the Riemann solver |
| PPM4 | 0.23 | 0.039 | 0.222 | not the reconstruction |
| RK3, CFL 0.1 | 0.22, 0.22 | | | not the integrator or time step |
| Omega = 0.01, q = 150 (pure shear flow) | 0.11 | 0.10 | 0.18 | rotation not needed |
| amp 1e-5 / 1e-3 | 0.215 / 0.224 | | | **linear in the wave amplitude** |
| amp 0 | 1e-20 (absolute) | | | |

Full size (128^2 x 16, B0 = 1, no gravity):
- With HLLE: ring 0.167 against uniform 0.040 versus theory.
- Static ring versus uniform twin at a 64-cell root: 0.22. At a 128-cell root: 0.16.
- This is barely convergent; the uniform twin's own error falls from 7.3% to 1.7%.

## Narrowing so far
1. **No shear, no defect.** With q = 0 at a 128-cell root, the ring is closer to theory than
   the uniform mesh (0.19 against 0.25 in c_y).
2. **A uniform flow along the interface is not enough.** `inputs/tests/linear_wave_mhd_tflow_smr.athinput`
   with `../tflow_smr/tflow_smr.py` has no shearing box and no gravity:
   - a 2-D MHD linear wave (slow, Alfven, fast) crosses a level-1 slab;
   - the flow V = 0, 1, 2 and the field run along the slab faces (or the flow V = 2 across them);
   - the slab lands at 0.8x the uniform-64 error in every case, which is correct.
3. **Kinematic winding is a normal interface error.** A uniform Bx = 0.1 with no wave
   (`problem/bx0`, new swing option) should keep By = 1 - 0.15 t exactly:
   - the uniform mesh is exact;
   - the ring has an interface error of 4.3e-4, 1.1e-4 and 2.6e-5 at 64, 128 and 256-cell roots;
   - that is second order, so it is not the defect.
4. **There is also a y-uniform By offset in the ring.**
   - It is about 1.4e-4 against B0 = 1, independent of the wave amplitude.
   - It comes from the background shear flow on the refined mesh.
   - It does not project on the wave, so it is not the defect either.
5. EMF correction at fine/coarse edges exists (`src/bvals/flux_correct_fc.cpp`, MHD
   SendFlux/RecvFlux).

So the defect is a linear error of the refined MHD scheme for a wave whose field
perturbation is wound up by a **sheared** flow across the level interfaces. It needs:
- a field;
- refinement;
- the velocity shear (a uniform tangential flow does not trigger it).

It is independent of the solver, reconstruction, integrator, rotation and gravity, and it
converges at roughly order 0.5.

Suggested next steps:
- Dump fine/coarse EMFs at the interface edges for one step of the base pair.
- Compare the coarse face fluxes with the sum of the fine ones, and the prolongated
  ghost-face fields with the fine solution.
- Check `AverageBoundaryFluxes`/`SumBoundaryFluxes` for the Ez edges along y, where the
  shear makes the EMF vary along the interface.

## EMF dump at the interface (2026-10-27)
A temporary, environment-guarded dump in `src/mhd/mhd_tasks.cpp` records the E_z (x3e) and
E_y edge EMFs on every MeshBlock x1 face:
- before correction: this block's own EMFs, after `CornerE`;
- after correction: after `RecvAndUnpackFluxFC`.

Usage: `ATHENA_EMF_DUMP_TIME=<t> ATHENA_EMF_DUMP_FILE=<prefix>`. The disarm does not work,
so the files hold the last stage-1 step of the run.

Ring, uniform 64 and uniform 128 were compared against the linear-theory EMF
`E_z = -q Omega x A c_x cos(psi) - A w_x B0 sin(psi)`. The wave part is
E(amp = 1e-4) - E(amp = 0), which removes the background.

1. **The EMF correction is correct.** After correction the coarse edge equals the mean of
   the two fine edges to 1e-20, and the fine EMFs are not changed.
2. **The fine side's own interface EMFs are inaccurate.**
   - At t = 0.5: 14% off theory (wave part), against 5.7% for uniform 64 at the same face.
     Uniform 128 is about 1% (2nd order).
   - The correction then imposes these values on the coarse side.
   - By t = 5 the whole box is affected: 12-15% everywhere outside the ring.
   - So the defect enters through the fine block's EMFs at the interface, which it builds
     from ghost cells prolongated from the coarse level.
   - **Next:** dump the fine ghost-zone primitives, bcc and face fields against the exact
     (or uniform-128) values; inspect `src/bvals/prolongation.cpp` (face-field and
     cell-centred prolongation, and whether ghost bcc/w0 are rebuilt after prolongation
     in a sheared flow).
3. **Separate, second defect: the background equilibrium is not preserved.**
   - With conserved-variable prolongation and an adiabatic gas, the fine cells next to the
     interface develop a spurious radial velocity, |v_x| = 2.8e-4 by t = 0.03, growing about
     7e-5 per step. The same happens in hydro (B0 = 0).
   - Cause: the ghost-cell total energy holds the large shear kinetic energy, which is
     quadratic in x; linear prolongation of E gives the wrong ghost pressure.
   - With B0 = 1 the coarse block's own interface EMF carries a spurious background of
     about 7e-4. It is removed by the correction.
   - `<mesh_refinement> prolong_primitives = true` or an isothermal gas makes the background
     exact (|v_x| ~ 1e-16).
   - This background error does not project on the wave. Primitive prolongation leaves the
     wave defect unchanged (c_y: ring 0.216 against uniform 0.075 versus theory).
   - Production adiabatic runs on refined shearing boxes without orbital advection should
     use `prolong_primitives = true`.

## Caveat found on the way
AthenaK's 2-D shearing box is x-z only: `shearing_box_r_phi` is "not yet implemented".
A 2-D x-y input runs **silently** with plain periodic x1 boundaries. Without orbital
advection that puts a qΩLx velocity jump at the boundary and the run is garbage. With
orbital advection the hydro test happened to look fine. A guard would help.
