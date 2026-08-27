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

`gt_sc14_full.pbs` — PBS template (bash; Cray PE + cray-pals + cray-fftw
modules).  Submit it **from the run directory**: `cd <rundir> && qsub
<repo>/scripts/cluster/gt_sc14_full.pbs`.  Three conveniences worth knowing:

- **Nothing below the `#PBS` block needs editing when you change the size.**
  Ranks, nodes and ranks-per-node are derived from `$PBS_NODEFILE`
  (`NRANKS=$(wc -l < $PBS_NODEFILE)`, `NNODES=$(sort -u $PBS_NODEFILE | wc -l)`),
  so changing only the `select=` line changes the launch.  This also prevents
  the classic waste of allocating `mpiprocs=256` and then launching `-ppn 32`:
  NAS charges whole nodes, so unused cores are billed anyway.
- **It is idempotent.**  If `<rundir>/rst` already holds dumps it continues from
  the newest one; otherwise it starts fresh from `$INPUT` and freezes a copy.
  A job cut off by the walltime is resumed by resubmitting the same script.
- **It reserves time for the final restart dump** by reading the job's own
  walltime from `qstat` and passing `walltime - GUARD_MIN` to athena's `-t`.

Paths are variables at the top (`ATHENAK`, `INPUT`, `RUNDIR`, `OVERRIDES`), and
per-run parameters are best passed as `OVERRIDES` rather than by editing the
tracked input file (which causes `git pull` conflicts on the cluster):

```bash
cd $RUNS/gt_sc14_b5hi
OVERRIDES="hydro_srcterms/bcool_beta=5 time/tlim=400" qsub -v OVERRIDES \
  $ATHENAK/scripts/cluster/gt_sc14_full.pbs
```

Sanity marks while it runs (from `GravitoTurb.user.hst`, via
`julia scripts/plot_gravito_turb_hist.jl runs/gt_sc14_b10 64 64` — pass the true
Lx Ly): Q dips to ~0.6-0.7 during the cooling collapse, rebounds and settles at
~1.33; alpha_total ~ 0.055-0.058; alpha' ~ 0.042 = 4/(9*gamma*(gamma-1))/beta.

## 2. The hi-res set: tc = 10, 5, 4, 3 (fragmentation boundary)

SC14's ".hi" resolution is 512 x 512 x 96 (8 cells/H) --- their tc = 3/4/5 rows
exist ONLY at this resolution, so the fragmentation-boundary test needs it.
Restarts cannot cross resolutions, so this family has its own anchor:

```bash
# 1. anchor: beta = 10 at hi-res, from scratch (300/Omega)
#    8-12 FILLED nodes: select=8:ncpus=256:mpiprocs=256:model=tur_ath
mpiexec -np 2048 build-cluster/src/athena \
  -i inputs/shearing_box/gravito_turb_sc14_hi.athinput \
  -d runs/gt_sc14_b10hi -t <walltime-10min>

# 2. the three low-beta stages morph from it and are INDEPENDENT --
#    submit them concurrently (each in its own PBS job)
HIRES=1 NRANKS=2048 scripts/cluster/gt_sc14_beta_chain.sh run 5    # 200/Omega
HIRES=1 NRANKS=2048 scripts/cluster/gt_sc14_beta_chain.sh run 4    # 200/Omega
HIRES=1 NRANKS=2048 scripts/cluster/gt_sc14_beta_chain.sh run 3    #  20/Omega
```

Work and cost (Athena Turin, SBU = wall-hours x nodes x 12, MAU = one 256-core
node -- always fill the nodes, partial nodes waste allocation proportionally):

| stage | duration | work vs the 256^2 run | |
|---|---|---|---|
| 10.hi anchor | 300/Om | 16.4x | |
| 5.hi, 4.hi | 200/Om each | 10.9x each | concurrent |
| 3.hi | 20/Om | 1.1x | fragments |
| total | 720/Om | 39.4x | ~230-760 SBU |

Wall time on 8 filled nodes: anchor ~1-3 h, then ~1-2 h for the concurrent
branches (throughput-dependent; calibrate first, below).

**Calibrate before committing.** One short job pins the throughput to ~10% and
checks that 6144 blocks decompose and the slab solver scales at 2048 ranks:

```bash
mpiexec -np 2048 build-cluster/src/athena \
  -i inputs/shearing_box/gravito_turb_sc14_hi.athinput \
  -d runs/gt_cal time/tlim=2.0 output2/dt=100 output3/dt=100 output4/dt=100
```
Cycles/wall-second from its log x 145,000 cycles gives the anchor's wall time.

Expected results (SC14 Table 1, ".hi" rows; time-average the trailing
100/Omega):

| run | alpha_Reyn | alpha_grav | alpha | alpha' | dv [HOm] | <cs> |
|---|---|---|---|---|---|---|
| tc=10.hi | 0.0170 | 0.0382 | 0.0552 | 0.0406 | 1.79 | 2.17 |
| tc= 5.hi | 0.0419 | 0.0582 | 0.100 | 0.0819 | 2.20 | 2.01 |
| tc= 4.hi | 0.0569 | 0.0650 | 0.122 | 0.100 | 2.31 | 1.96 |
| tc= 3.hi | fragments -- no steady state (their criterion: t_cool <~ 3/Omega) |

tc=4 surviving as turbulent while tc=3 fragments is a genuine prediction test
of the criterion, not a re-fit.  For tc=3, watch rho_max: once fragments pass
the Truelove ceiling the densities are resolution artifacts (see the Phase-2
collapse program), so report fragmentation TIME and morphology, not peak rho --
or rerun it with AMR.

## 3. The full beta scan (Table 1 / Figures 3-4)

`gt_sc14_beta_chain.sh` — implements SC14's "morphing": each constant-beta run
restarts from the FINAL rst of an earlier run (10 -> 20 -> 40 -> 80 -> 120, and
10 -> {4, 5, 8}), with the cooling time overridden on the command line and tlim
extended by the SC14 Table-1 duration.  `print` shows the plan; `run <beta>`
executes one stage (wrap it in a PBS job with the same resources as the b10 run):

```bash
scripts/cluster/gt_sc14_beta_chain.sh print         # standard-res family
HIRES=1 scripts/cluster/gt_sc14_beta_chain.sh print # ".hi" family
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
