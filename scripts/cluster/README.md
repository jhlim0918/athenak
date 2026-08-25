# Cluster kit for the SC14 gravito-turbulence runs (Phase 3)

Reproduction of Shi & Chiang (2014, ApJ 789, 34) with the multigrid slab-gravity
stratified shearing box.  Everything here assumes the repo root as working
directory and a cluster build with **FFT enabled** (the slab boundary needs
kokkos-fft):

```bash
cmake -B build-cluster -D Athena_ENABLE_MPI=ON -D Athena_ENABLE_FFT=ON
cmake --build build-cluster -j
```

kokkos-fft is fetched by CMake at *configure* time — run the configure step on a
node with network access (or pre-clone `github.com/kokkos/kokkos-fft` and point
`FETCHCONTENT_SOURCE_DIR_KOKKOS-FFT` at it).

## 1. The standard beta = 10 run

`gt_sc14_full.pbs` — PBS template for the full-size box (256 x 256 x 48, 768
MeshBlocks of 16^3, 64 ranks; edit queue/model/module lines).  It freezes the
input and provenance into `runs/gt_sc14_b10/` and uses `-t` to guarantee a final
restart dump inside the walltime.  Expected duration at 64 ranks: a few hours to
t = 300 (the transient phase t ~ 10-25 has the smallest timesteps).

Sanity marks while it runs (from `GravitoTurb.user.hst`, via
`julia scripts/plot_gravito_turb_hist.jl runs/gt_sc14_b10 64 64` — pass the true
Lx Ly): Q dips to ~0.6-0.7 during the cooling collapse, rebounds and settles at
~1.33; alpha_total ~ 0.055-0.058; alpha' ~ 0.042 = 4/(9*gamma*(gamma-1))/beta.

## 2. The beta scan (Table 1 / Figures 3-4)

`gt_sc14_beta_chain.sh` — implements SC14's "morphing": each constant-beta run
restarts from the FINAL rst of an earlier run (10 -> 20 -> 40 -> 80 -> 120, and
10 -> {4, 5, 8}), with the cooling time overridden on the command line and tlim
extended by the SC14 Table-1 duration.  `print` shows the plan; `run <beta>`
executes one stage (wrap it in a PBS job with the same resources as the b10 run):

```bash
scripts/cluster/gt_sc14_beta_chain.sh print
scripts/cluster/gt_sc14_beta_chain.sh run 20        # after b10 finishes
```

Environment knobs: `ATHENA` (binary path), `NRANKS`, `MPIEXEC`.

Time-average the trailing 100-200/Omega of each stage for the alpha(beta)
comparison; SC14's fits are <alpha> ~ 1/(Omega t_cool) (their Fig. 3) and
alpha' = 4/(9*gamma*(gamma-1))/(Omega t_cool) (eq. 21, Fig. 4).  Their
fragmentation boundary is t_cool <~ 3/Omega — the beta = 4 and 5 stages should
remain turbulent, beta = 3 (not in the chain) fragments.

## Notes

- Vertical boundaries are `diode` (no-inflow outflow): plain `outflow` feeds a
  gravity-driven inflow runaway (see the Phase 3 section of
  `validation/multigrid_shear_implementation.pdf`).
- Restart continuation after a walltime kill:
  `mpiexec -np N build-cluster/src/athena -r runs/<run>/rst/<last>.rst -d runs/<run> ...`
  (parameters are embedded in the rst; command-line overrides still apply).
- CLI overrides only work for parameters that appear in the input file.
- The hi-res twin (512 x 512 x 96, SC14 ".hi") is the same input with
  mesh/nx1=512 mesh/nx2=512 mesh/nx3=96 — 8x the cells, ~256 ranks recommended.
