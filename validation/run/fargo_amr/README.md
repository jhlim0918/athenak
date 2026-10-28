# Phase 5: orbital advection (FARGO) across x2 level jumps — test runs

Results are written up in `multigrid_selfgravity_validation.tex`, section "Phase 5";
the method in `multigrid_shear_implementation.tex`, section "Phase 5".
All runs were on the Mac (2026-10-28), branch `fargo-amr`, serial build `build/`, MPI build
`build-mpi/`, reference build of the parent commit d3edb948.
`$ATH` is the athena binary; `$IN` is `inputs/shearing_box`.

## Regression (bit-identical to the parent commit)
Run each input with the parent-commit binary and the new one, then compare the bin
payloads (after the header) and the .hst files. The new run needs `oa_annular = true`
in `<shearing_box>` (patch a copy of the input: command-line overrides only replace keys
that already exist):

    epicycle_smr_ring  epicycle_amr_ring  epicycle_amr_ring2      (time/tlim=6.28)
    hydro_incompress_shwave_smr_ring{,2}  hydro_incompress_shwave_amr_ring  (tlim=1)
    epicycle  hydro_incompress_shwave  hydro_compress_shwave  mhd_compress_shwave
                                          (uniform: no patch needed)

## Shearing wave (vortical, JG05), directories here
    unif128   $ATH -i $IN/hydro_incompress_shwave.athinput output1/data_format=%.15e \
                   mesh/nx1=128 mesh/nx2=128 output2/file_type=bin
    unif      $ATH -i $IN/hydro_incompress_shwave.athinput output1/data_format=%.15e
    ring      $ATH -i $IN/hydro_incompress_shwave_smr_ring.athinput output1/data_format=%.15e
    ypatch    $ATH -i $IN/hydro_incompress_shwave_smr_ypatch.athinput output1/data_format=%.15e
    amrring   $ATH -i $IN/hydro_incompress_shwave_amr_ring.athinput output1/data_format=%.15e
    amrbox    $ATH -i $IN/hydro_incompress_shwave_amr_box.athinput output1/data_format=%.15e
Analysis: `python3 compare.py unif128 unif ring ypatch amrring amrbox` (x-KE, mass);
`python3 errmap.py 00010` (error next to the x2 level jumps);
figure: `julia figures/plot_fargo_amr_shwave.jl` from `validation/`.

## Conservation (compressible shwave, amp 0.1, periodic x1), directories cons_*
`base.athinput` (static patch) and `amr.athinput` (moving box) are in this directory.

    cons_fc / cons_nofc       $ATH -i base.athinput mesh/ix1_bc=periodic mesh/ox1_bc=periodic \
                                   time/tlim=2.0 [shearing_box/oa_flux_correct=false]
    cons_fc_sp / cons_nofc_sp same with shear-periodic x1
    cons_amr_fc / cons_amr_nofc  $ATH -i amr.athinput ... time/tlim=4.0
    cons_cold_new             base.athinput, periodic x1, hydro/iso_sound_speed=0.05
                              problem/amp=0.001 time/nlim=200 (OA timestep limit binds)
MPI: the same with `mpirun -np 3|4|7` and the MPI build; all bin dumps bit-identical.

## Swing with multigrid gravity, directory swing
    unif       $ATH -i inputs/tests/swing_selfgrav_smr.athinput mesh_refinement/refinement=none
    ring       $ATH -i inputs/tests/swing_selfgrav_smr.athinput
    patch      $ATH -i inputs/tests/swing_selfgrav_smr_ypatch.athinput
    *_nog      the same with hydro_srcterms/self_gravity=false
(all with output1/data_format=%.15e). Analysis: `python3 swing/cmp.py` from `swing/`.

## MHD with orbital advection on rings (branch fargo-amr-mhd), directory mhd
Reproducer pairs (inputs/tests/swing_mhd_repro.athinput, 64^2 x 4, B0 = 1, no gravity,
static ring |x| < pi/4); `fargo.athinput` is that input with orbital advection on:
    nf_ring / nf_uni      $ATH -i inputs/tests/swing_mhd_repro.athinput [mesh_refinement/refinement=none]
    f_ring / f_uni        $ATH -i fargo.athinput [mesh_refinement/refinement=none]
    f_ring_nofc           $ATH -i fargo.athinput shearing_box/oa_flux_correct=false
    z3, z3_nofc           $ATH -i z3.athinput [...]     (x1 and x3 level jumps)
    amr, amr_nofc         $ATH -i amr.athinput [...]    (moving ring, AMR)
    z3s/z3m, amrs/amrm    *_bin.athinput, serial / mpirun -np 4 (bin dumps bit-identical)
(all with output1/data_format=%.15e). Or: python3 validation/run/swing_mhd_repro/repro.py
6 base fargo fargo_nofc. History columns 12/13 (0-based) are bmis/bnrm.
Full magnetized swing with gravity, directory mhd/swing (3 MPI ranks):
    smrF_b0.5, smrF_b1.0  mpirun -np 3 $ATH -i smr.athinput shearing_box/orbital_advection=true problem/b0=...
    amrF_b1.0             mpirun -np 3 $ATH -i amr.athinput shearing_box/orbital_advection=true problem/b0=1.0
Twins: the uniform runs uF_b0.5, uF_b1.0 of validation/run/swing_mhd/runs (uniform MHD with
orbital advection is bit-identical between the two branches). Figure:
julia figures/plot_fargo_amr_mhd.jl
Across x2 level jumps (second step), directory mhd/stepB (inputs ypatch/ybox in it,
oa_annular = false): yp/yp_nofc static patch |x|<pi/4, |y|<pi/2; yb/yb_nofc moving box
(ring_y* window); yps/ypm, ybs/ybm serial vs 4 ranks (*_bin inputs); f_ring, z3, amr:
first-step runs repeated with the second-step binary (histories bit-identical);
yb_q0 (qshear=0) and yb_q0_nf (qshear=0, orbital_advection=false): the div(B) jumps at
derefinement are pre-existing. Full swing on the patch: mhd/swing/ypF_b0.5, ypF_b1.0
    mpirun -np 3 $ATH -i ypatch.athinput shearing_box/orbital_advection=true problem/b0=...
