# x1-face levels in the shearing box: port of Rixin Li's shear-face machinery

Branch `fargo-amr-x1face` (from `fargo-amr-mhd` at 0583ec1a), 2026-11-02.
Source: Rixin Li's athenak branch `amrSG_dev` (head 26aaded5, merge base with ours 1f3e1ed9).

## 1. What was ported, from where

| Our file | From Rixin | Notes |
| --- | --- | --- |
| `src/shearing_box/shearing_box_pull.cpp` | e9f71fac (+804d8005) | `ShearingBox::PlanPull`, `WaitPull`, `ShearingBoxCC::PullSend`, `PullRecv`. Adapted: slope limiter = our `ProlongCC` minmod (his `PROLONG_CC_SLOPE` default is the same expression), TENO/WENO-Z(MC) remap like PPMX, plan timers removed, `nvar_` -> our `nvar`. |
| `src/shearing_box/shearing_box_plan.cpp` | e9f71fac | `BuildCandidates`, `LayoutPeerMessages`, unchanged. |
| `src/shearing_box/shearing_box_flux.cpp` | e9f71fac | `PlanFluxMatch/SendFluxMatch/RecvFluxMatch/ClearFluxMatch` (one uniform face level) and `PlanFluxPair/SendFluxPair/RecvFluxPair` (face levels). Same adaptations as the pull. |
| `src/shearing_box/shearing_box.{hpp,cpp}`, `shearing_box_cc.cpp`, `shearing_box_tasks.cpp` | e9f71fac | Only the new members and the `face_levels` dispatch (InitRecv -> PlanPull; ClearRecv/ClearSend -> WaitPull; PackAndSendCC -> PullSend; RecvAndUnpackCC -> PullRecv; new RecvAndUnpackCoarseCC). **Our stock two-step exchange, `ConsistentDx2`, local-id MPI tags, `SetX1BndryMBs/ReinitAfterMeshUpdate` and the non-FARGO momentum offset are kept unchanged**; his refactor of the stock path (`Pieces`, `Set`, the two-step coarse exchange `PackAndSendCoarseCC`) was not ported: it is unused when the pull is on. |
| `src/bvals/bvals.hpp`, `bvals_cc.cpp`, `flux_correct_cc.cpp`, `prolongation.cpp` | e9f71fac | `ShearSlotOx1`, `skip_shear_x1_flux`, `skip_shear_x1_prol`. Added by us: the same skip in the hydro `PrimToConsFineBndry` (`prolong_prims.cpp`), otherwise `prolong_primitives = true` would overwrite the collected x1 ghosts with stale primitives. |
| `src/hydro/hydro.{hpp,cpp}`, `hydro_tasks.cpp` | e9f71fac | Flags set in the Hydro constructor; flux matching inside `SendFlux`/`RecvFlux`, once-per-stage unpack guard (`flux_unpacked`), coarse x1 ghosts in `RecvU_Shr`, physical x3 BCs before the pull with non-periodic x3. Adapted: the flux matching is planned only in `InitRecv` of the main stages (`flux_planned`), so the parabolic STS stages (which also call SendFlux/RecvFlux) are untouched. |
| `src/mesh/mesh_refinement.{hpp,cpp}`, `meshblock_tree.{hpp,cpp}` | 804d8005 | Shear-face rule: `ShearFaceGeom`, `ShearFaceLevelLimit`, `ShearWindow`, `ShearFaceCheckDue`, `ShearFaceTreeRefine`, `ShearFaceDerefineOK`, `CheckShearFaceLevels`, `GetLeafLocations`. Adapted: `ShearFaceGeom::on` requires `Mesh::sbox_face_levels`; a static `CheckShearFaceRule` also checks static meshes at the tree build (over all x2 offsets); his `CheckCriteria` split is inlined in our `CheckForRefinement` (our ring sync, slab-x3 cap, refinement-interval filter kept). Dust/gravity/profiling/`stop_max_nmb` parts left out. |
| `src/mesh/build_tree.cpp`, `mesh.{hpp,cpp}` | (ours) | `Mesh::sbox_face_levels`; the uniform-face-level check (I1) is skipped for face-levels runs. |
| `src/pgen/tests/shwave.cpp` | 804d8005 | `band_n` scheduled refinement boxes (`ShwaveBands`, enrolled on restarts), `zmode_amp`. Our moving-ring criterion and non-FARGO ICs kept. |
| `inputs/shearing_box/hydro_shwave_{smr_box,smr_facebox,amr}.athinput` | 26aaded5 | unchanged |

Not ported: his orbital-advection changes (`BuildMixedLists` etc.; ours is kept), his gravity,
dust, particles, outputs, `coarse_physical_bcs` (our tree already applies physical BCs to the
coarse arrays, `HydroBCsCoarse`), amplitude-aware prolongation, `x2_face_parallel`.
a12d6232 and 36cf4575 contain nothing for this feature (dust/gravity/load balance).

## 2. Gating (what runs which path)

`Mesh::sbox_face_levels` (mesh.cpp) is true only for: `<shearing_box>` present, 3-D, both x1
faces `shear_periodic`, mesh refinement (static or adaptive), **no `<mhd>`, `<gravity>`,
`<particles>`, `<radiation>`**, orbital advection on, and `<shearing_box> x1_face_levels`
(default true). Then:

- the x1 faces may change level along x2 (static `<refined_region>`s and AMR);
- `ShearingBoxCC::face_levels/coarse_on`: pull exchange for u0 and coarse_u0;
- `pbval_u->skip_shear_x1_flux/prol`;
- the AMR shear-face rule replaces our x1-boundary refine/derefine vetoes and graded cap
  (CheckForRefinement); static meshes are checked against the rule at the tree build.

Otherwise (MHD, gravity, particles, non-FARGO, uniform meshes, `x1_face_levels = false`) the
previous behaviour is unchanged: one uniform level on both x1 faces (CheckShearingBoxRefinement,
fatal), the boundary vetoes and graded cap, the stock same-level exchange for CC and FC
variables. MHD never touches any new code path (the flags are set only by Hydro), and its
results are bit-identical to 0583ec1a (section 3A).

x1 flux matching (`ShearingBoxCC::flux_on`, key `<shearing_box> x1_flux_match`): hydro with
orbital advection and shear-periodic x1 in 3-D; default **on with mesh refinement** (face levels:
cell-pair matching `PlanFluxPair`; one uniform face level, e.g. with gravity: `PlanFluxMatch`),
**off on uniform meshes** (AthenaK's results there are unchanged; true makes them conservative
too). Never in the non-FARGO mode (the x1 fluxes of y momentum and energy differ between the two
faces by the shear boost; mass alone would be matchable, not done).

Requirements of the pull (fatal otherwise): `meshblock/nx2 >= 4 nghost`; a sender finer by d
levels needs `nghost*2^d <= nx1` and nx2, nx3 divisible by 2^d (AMR rule keeps d <= 1).

## 3. Validation

All runs on the Mac in `validation/run/x1face/` (untracked), MPI build `build-mpi/` of this
branch (`athena_after`) and of 0583ec1a (`athena_before_0583ec1a`, same compiler/flags, built
from `git archive 0583ec1a`). Analysis: `python3 validation/x1face_analyze.py REF RUN...`
(copy of `validation/run/x1face/analyze.py`), bin payload comparison: `cmpbin.py A/bin B/bin`
(payload after the parameter header). The machine was heavily loaded by other jobs (load ~50
on 18 cores), so timings are not meaningful.

### A. Regression

Binaries: `athena_before_0583ec1a`, `athena_after` (= commit 1ce2a722, as ported), `athena_fix`
(= a9c6fe55, with the flux-matching offset fix of section 4.1). `IN=inputs`, run from a fresh
directory under `validation/run/x1face/`.

| Test | Command | Result |
| --- | --- | --- |
| MHD ring, OA on | `$ATH -i $IN/tests/swing_mhd_repro.athinput shearing_box/orbital_advection=true output1/data_format=%.17e` (before vs after) | `SwingMHD.user.hst` **bit-identical** (303 rows) |
| MHD uniform, OA on | same + `mesh_refinement/refinement=none` | **bit-identical** (292 rows) |
| Hydro uniform 128^2 (shwave) | `$ATH -i $IN/shearing_box/hydro_shwave_smr_box.athinput mesh_refinement/refinement=none` (before vs after) | hydro.hst and user.hst **bit-identical** (uniform meshes stay on the stock path, matching off) |
| Hydro refined, opt-out | `base.athinput` (`fargo-amr` validation/run/fargo_amr) with `x1_face_levels = false`, `x1_flux_match = false` added, `time/tlim=1.0`, vs before | hst identical, 11/11 bin payloads **bit-identical** |
| Swing + MG gravity, uniform | `$ATH -i $IN/tests/swing_selfgrav_smr.athinput mesh_refinement/refinement=none output1/data_format=%.15e` | peak d_cos/A = **-3.511418** at t = 7.804 (identical history) |
| Swing + MG gravity, ring | `mpirun -np 3 $ATH -i $IN/tests/swing_selfgrav_smr.athinput output1/data_format=%.15e` | fix: **-3.583473** (old -3.583481); max\|d_sin\|/A 3.8e-7 (old 1.3e-13); max\|d_cos - old\|/max\|old\| = 2.3e-6. As ported (killed at t = 6.8): 1.0e-2 deviation, d_sin 2.7e-3 |
| Swing + MG gravity, y patch | `mpirun -np 3 $ATH -i $IN/tests/swing_selfgrav_smr_ypatch.athinput output1/data_format=%.15e` | fix: **-3.532415** (old -3.532422); d_sin 3.9e-7 (old 6.1e-13); deviation 2.1e-6. As ported: -3.500331, d_sin 5.8e-3 |
| Serial vs 4 ranks, facebox (level jump across x1) | `$ATH -i hydro_shwave_smr_facebox_bin.athinput time/tlim=2.0` vs `mpirun -np 4 ...` | 3/3 bin payloads **bit-identical** (hst sums differ in the last bits: reduction order) |
| Serial vs 4 ranks, adaptive box (to t = 5, crosses the face band) | `hydro_shwave_amr_bin.athinput time/tlim=5.0` | 6/6 bin payloads **bit-identical** |

The swing ring/patch runs keep the uniform face level (MG gravity) and differ from 0583ec1a only
through the x1 flux matching (now on by default on refined meshes): with it switched off the
refined hydro path is bit-identical (opt-out row). The changes of the peak (8e-6 A) and of the
phase lock (1e-13 -> 4e-7) are the effect of replacing the inner-face fluxes by the remapped
outer-face ones; `x1_flux_match = false` restores the old results exactly.
The `_bin` inputs are Rixin's with an `<output2>` bin dump of hydro_u every 1.0.

### B. Rixin's tests (linear compressible shearing wave, JG05, amp 4e-4, plm + hlle, to t = 8)

Inputs `inputs/shearing_box/hydro_shwave_{smr_box,smr_facebox,amr}.athinput` (Rixin, 26aaded5),
serial except u256 (`mpirun -np 3`). rms definition (Rixin's, reproduced): the run's dVyc
(user.hst column 3) linearly interpolated to the output times of the uniform 256^2 run,
`rms = sqrt(mean (dVyc - dVyc_256)^2)` (`x1face_analyze.py`, column rms_lin; rms_spline = both
cubic-spline interpolated to t = 0.1 k). Mass drift: hydro.hst column 3, (M(8) - M(0))/M(0) and
max over t.

| Run | Command | rms_lin (fix) | rms_lin (as ported) | dM/M(t=8) (fix) | max\|dM\|/M (fix) |
| --- | --- | --- | --- | --- | --- |
| uniform 128^2 | `smr_box mesh_refinement/refinement=none` | **6.804e-5** (Rixin 6.8e-5) | same | -1.8e-11 | 2.5e-11 |
| uniform 256^2 | `+ mesh/nx1=256 mesh/nx2=256` | reference | | | |
| static box | `smr_box` | 5.183e-5 | 5.046e-5 | -1.1e-16 | 6.5e-14 |
| box at the x1 face | `smr_facebox` | 5.186e-5 | 5.047e-5 | 8.9e-16 | 5.2e-14 |
| adaptive box | `amr` | 5.764e-5 | 5.085e-5 | 8.2e-15 | 6.5e-14 |
| static box, no x1 matching | `smr_box_keys shearing_box/x1_flux_match=false` | 5.183e-5 | | 1.5e-11 | 1.8e-11 |
| adaptive, no x1 matching | `amr_keys shearing_box/x1_flux_match=false` | 5.771e-5 | | -5.9e-9 | 1.0e-8 |
| adaptive, original offset | `amr_keys2 shearing_box/x1_flux_match_offset=exchange` | 5.085e-5 (= as ported, bit-identical history) | | | |

Rixin: refined 4-5e-5, uniform 128^2 6.8e-5. As ported we reproduce his numbers (5.05-5.09e-5).
With the offset fix the refined runs give 5.18-5.76e-5, which equals the runs without any x1
matching (5.183e-5 / 5.771e-5): the corrected matching moves the same flux and is
accuracy-neutral, as it should be; the 5% "improvement" of the original offset is the artefact
that also breaks the swing phase lock (section 4.1). All refined runs lie between the uniform
128^2 and 256^2 runs, as Rixin states. The adaptive run made 84 / deleted 56 blocks (92 at t = 8);
its bands never needed the shear-face rule.

Shear-face rule stress test (ours): `inputs/hydro_shwave_amr_l2face.athinput` = the adaptive
input with num_levels = 3, band3 (the box at the outer x1 face, t = 4-6) at level 2, tlim = 6,
`mesh_refinement/max_nmb_per_rank=2048`. The rule acted (first regrid with the level-2 face box:
"12 blocks refined"; later "4 blocks refined, 24 kept from derefining", ...): 490 blocks created,
126 deleted, no fatal check. Mass: dM/M(t=6) = -4.1e-14 (fix), -5.2e-15 (as ported); rms_lin
vs u256 to t = 6: 3.26e-5 (fix), 1.45e-5 (as ported).

Nonlinear wave (`problem/amp=0.3`, to t = 8; `*_keys` inputs = Rixin's with `x1_flux_match`
and `oa_flux_correct` exposed for overrides):

| Run | x1 matching | OA flux corr. | dM/M(t=8) | max\|dM\|/M |
| --- | --- | --- | --- | --- |
| static box | on (fix) | on | -6.8e-15 | 3.6e-14 |
| box at x1 face | on (fix) | on | -3.7e-15 | 8.0e-14 |
| adaptive box | on (fix) | on | -2.6e-15 | 5.8e-14 |
| static box | off | on | -6.7e-7 | 6.7e-7 |
| static box | off | off | -2.4e-6 | 4.6e-6 |
| box at x1 face | off | on | 5.9e-5 | 6.1e-5 |
| box at x1 face | off | off | 4.6e-5 | 4.8e-5 |
| adaptive box | off | on | 8.2e-6 | 8.3e-6 |

(as ported, i.e. the original offset: box/facebox/amr 8.4e-15 / -1.8e-14 / -1.7e-14.)
Rixin: with the corrections 1e-13-1e-14, without 1e-4-1e-7: reproduced.
Commands: `$ATH -i inputs/hydro_shwave_{smr_box,smr_facebox,amr}_keys.athinput problem/amp=0.3
[shearing_box/x1_flux_match=false] [shearing_box/oa_flux_correct=false]`.

### C. Our conservation case (compressible shwave amp 0.1, static patch interior in x1,
shear-periodic x1; `fargo-amr` validation/run/fargo_amr/base.athinput)

`fargo-amr` validation/run/fargo_amr/{base,amr}.athinput copied to `validation/run/x1face/inputs`.

| Run | Command | dM/M |
| --- | --- | --- |
| cons_fc_sp (static patch, t = 2), 0583ec1a | (existing run, fargo-amr) | **3.84e-9** |
| cons_fc_sp, this branch | `$ATH -i base.athinput time/tlim=2.0` | **1.9e-14** (as ported -3.8e-15) |
| cons_amr_sp (moving box, t = 4), 0583ec1a | `$ATH -i amr.athinput time/tlim=4.0` | -2.15e-9 (face columns vetoed: 98 created / 84 deleted) |
| cons_amr_sp, this branch | same | **1.1e-14** (as ported -2.3e-14); the box now refines the face columns too (112 / 98) |

The ~4e-9 drift with shear-periodic x1 on refined meshes is gone (round-off).

## 4. Findings, open issues, risks

### 4.1 Deviation from Rixin's design: the offset of the flux matching (fixed, commit a9c6fe55)
`Hydro::InitRecv` plans the matching right after `ShearingBox::InitRecv(time)`, which uses
`time + dt` in the last RK stage (the offset of the exchange at the END of that stage). The
fluxes of the stage, however, come from ghost zones exchanged with `yshear(mesh time)`. Rixin's
code therefore maps the authoritative face fluxes onto the other face one step of shear away
(~1 cell here) in the last stage. Mass/momentum conservation is unaffected, but the swing phase
lock broke (max|d_sin|/A 6e-3 vs 1e-13; peak -3.500 vs -3.532 on the patch). We plan the
matching with `yshear(mesh time)` (`<shearing_box> x1_flux_match_offset = ghost`, default;
`exchange` restores the original). With it the matching is accuracy-neutral (identical rms to
the runs without matching) and the swing runs change by 2e-6 only. Rixin's 4-5e-5 rms is
reproduced only with the original offset (bit-identical histories); please tell Rixin.

Why the original offset brings the refined runs CLOSER to u256 (5.05 vs 5.18e-5; stress case
1.45 vs 3.26e-5) needs a word. AthenaK's stock convention itself lags: the RK2 predictor state
is at t + dt but its x1 ghosts are exchanged with yshear(t) (InitRecv adds dt only in the last
stage). Mapping the last-stage fluxes with yshear(t + dt) is physically the right offset for
that state, so it partly undoes the stock lag at the faces, which the 256^2 reference (half
the dt) suffers less. It does so inconsistently with the receiver's own ghost cells, which is
what breaks the swing symmetry. The default (`ghost`) is the do-no-harm choice: identical to
the runs without matching up to round-off-level remap differences (1e-6 relative), only
conservative. Fixing the stock stage lag itself (ghosts of stage s with the stage time) would
be a separate change, affecting uniform meshes and MHD too; not done here.

### 4.2 Other open issues / risks
- **Not covered by the new path** (still uniform x1-face level, I1, with the old vetoes):
  MHD, any `<gravity>` (multigrid shear planes need I1; FFT gravity is uniform-only),
  `<particles>`/dust (shear routing by face-block maps), the non-FARGO mode. Lifting these
  needs: FC pull + EMF matching (MHD); octet shear fill for mixed face levels (MG); particle
  maps at the finest face granularity (Rixin's `BuildShearMaps` would be the template).
- Flux matching is on by default on refined meshes, so refined hydro shearing-box results
  change at the 1e-6 level (swing peak 8e-6 A, d_sin 4e-7 A). `x1_flux_match = false` gives
  the old results bit for bit. In the non-FARGO mode no matching (shear boost).
- Non-periodic x3 with face levels (stratified boxes): ported (`ApplyPhysicalBCs` before the
  pull; our tree's `HydroBCsCoarse` gives the coarse physical BCs) but **not tested**.
- The pull is re-planned every stage on the host (Rixin's design; `PlanPull` loops over the
  local face blocks and their candidates); fine for these sizes, unmeasured at scale.
- The shear-face rule does a mesh check every cycle until the first regular check (cycles
  1-9 in the tests: "check between the mesh checks"); harmless, as in Rixin's code.
- `ShearingBox`'s destructor is non-virtual (as before); `ShearingBoxCC` frees its requests.
- cpplint line length (90) is exceeded in the ported code (Rixin's lines) and a few skip blocks.
- `meshblock/nx2 >= 4 nghost` is required with face levels (fatal otherwise).
