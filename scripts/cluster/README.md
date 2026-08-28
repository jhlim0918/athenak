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

`gt_sc14_full.pbs` — PBS template (bash; Cray PE + cray-pals + cray-fftw).
Submit it **from the run directory**, which is where outputs land:

```bash
cd <rundir> && qsub <repo>/scripts/cluster/gt_sc14_full.pbs
```

`ATHENAK` and `INPUT` are variables at the top.  Ranks and ranks-per-node come
from `$PBS_NODEFILE`, so changing only the `select=` line changes the launch --
and every allocated core is used (NAS charges whole nodes, so `mpiprocs=256`
with `-ppn 32` would bill 8x what it computes).  `-t 47:45:00` leaves time for a
final restart dump inside the 48 h walltime; to continue, swap the `-i` line for
`-r $(ls rst/*.rst | tail -1)`.  Per-run parameters go on the mpiexec line
(`hydro_srcterms/bcool_beta=5`) rather than into the tracked input file, which
would conflict on the next `git pull`.

## 2. Hi-res (".hi", 512 x 512 x 96): performance flags -- READ BEFORE SUBMITTING

SC14's ".hi" resolution is 8x the cells of the standard run and, measured on
Athena Turin (2026-08), costs about **4,700 SBU for the beta=10 anchor alone**
(~10-12k SBU for the tc = 10/5/4/3 set).  Decide against the allocation before
launching; the standard-resolution scan below delivers most of the science at
~1/10 the cost.  The flags that matter, each measured:

- **The code is memory-BANDWIDTH-bound: budget by nodes, not cores.**
  Measured: 11.8 s*node per cycle at hi-res regardless of how many of a node's
  256 cores are ranked.  Consequences:
  - `select=3:ncpus=256:mpiprocs=256` (768 ranks = 768 blocks, 1 block/rank)
    is the balanced full-node launch: ~3.3 s/cycle with the solver fix.
  - `select=6:ncpus=256:mpiprocs=128` (still 768 ranks) doubles the bandwidth
    per rank: ~half the wall time at the SAME total SBU.  Preferred when queue
    time matters.
  - **Never 512 ranks**: 768 blocks / 512 ranks = 1.5 blocks/rank, so half the
    ranks carry double load -- 25% of the charge is wasted (the load-balance
    warning at startup is telling you this).
- **Fixed gravity iterations, not the stagnation rule** (in the input already):
  `threshold=-1.0, niteration=6, full_multigrid=false`.  The automatic rule
  (`threshold=0.0`) polishes the defect to 1e-11 -- five orders below the
  discretization error -- at 3.25 s/solve vs 1.1 s for 6 warm-started V-cycles,
  and the extra V-cycles multiply the replicated coarsest-grid work that added
  nodes cannot reduce.
- **32^3 MeshBlocks** (in the input): 768 blocks, root grid 16x16x3 -- same
  coarsest grid as the validated standard run, half the ghost overhead of 16^3.
  The two startup warnings about the coarsest level (cannot reach one cell;
  768 DOF) are expected for SC14's 12H-tall box and are shared by the validated
  standard-resolution run.
- **Timestep reality**: dt ~ 1.4e-3 in the transient (measured), ~2.2e-3 in
  steady state -- about 155k cycles for the 300/Omega anchor.  Walltime at
  3 full nodes: ~5.5 days (restart-chain through 48 h jobs); at 6 half-packed
  nodes: ~2.7 days.
- Restarts read their parameters from the rst file: when continuing a run that
  predates these settings, pass `gravity/threshold=-1.0 gravity/niteration=6`
  on the command line.

The chain mechanics (anchor from scratch, tc = 5/4/3 morphed, `HIRES=1`) are
unchanged; SC14 Table-1 ".hi" targets:

| run | alpha_Reyn | alpha_grav | alpha | alpha' | dv [HOm] | <cs> |
|---|---|---|---|---|---|---|
| tc=10.hi | 0.0170 | 0.0382 | 0.0552 | 0.0406 | 1.79 | 2.17 |
| tc= 5.hi | 0.0419 | 0.0582 | 0.100 | 0.0819 | 2.20 | 2.01 |
| tc= 4.hi | 0.0569 | 0.0650 | 0.122 | 0.100 | 2.31 | 1.96 |
| tc= 3.hi | fragments -- no steady state (criterion: t_cool <~ 3/Omega) |

## 3. The standard-resolution beta scan (the current campaign)

The chosen set is tc = {3, 4, 5, 40, 80} at 256 x 256 x 48, all morphed from
the FINISHED standard-resolution beta=10 run (runs/gt_sc14_b10 -- do not rerun
it).  Every stage except tc=80 sources b10 directly, so **tc = 3, 4, 5, 40 can
run concurrently** (one filled node each); tc=80 follows tc=40:

```bash
scripts/cluster/gt_sc14_beta_chain.sh print
NRANKS=256 scripts/cluster/gt_sc14_beta_chain.sh run 3     #  20/Omega
NRANKS=256 scripts/cluster/gt_sc14_beta_chain.sh run 4     # 200/Omega
NRANKS=256 scripts/cluster/gt_sc14_beta_chain.sh run 5     # 200/Omega
NRANKS=256 scripts/cluster/gt_sc14_beta_chain.sh run 40    # 300/Omega
NRANKS=256 scripts/cluster/gt_sc14_beta_chain.sh run 80    # 400/Omega, after 40
```

The chain passes `gravity/threshold=-1.0 gravity/niteration=6` automatically
(measured 3x cheaper than the stagnation rule at unchanged physical accuracy).
Estimated cost of the whole set: ~1.1k-1.4k SBU; wall ~2-3 days with the four
independent branches concurrent.  At this resolution the paper offers direct
Table-1 comparisons for tc = 40 (alpha 1.44e-2) and 80 (0.89e-2); tc = 3/4/5
exist in the paper only at hi-res, so those compare qualitatively
(fragmentation yes/no and trend) unless the hi-res set is run later.

Time-average the trailing 100-200/Omega of each stage; SC14's fits are
<alpha> ~ 1/(Omega t_cool) (their Fig. 3) and
alpha' = 4/(9*gamma*(gamma-1))/(Omega t_cool) (eq. 21, Fig. 4).
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
