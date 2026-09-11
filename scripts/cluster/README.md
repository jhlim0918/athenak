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

`gt_sc14_full.slurm` is the TACC Vista twin (Slurm, `ibrun`, Grace-Grace `gg`
queue, `gcc`/`openmpi`/`fftw3` modules; build with `-D Kokkos_ARCH_ARMV9_GRACE=ON`).
Same submit-from-the-run-directory convention, run directories in `$SCRATCH`:

```bash
cd <rundir> && sbatch $HOME/athenak-multigrid/scripts/cluster/gt_sc14_full.slurm
```

Set `#SBATCH -A` to your allocation first.  256 ranks = 2 nodes x 128 of 144
cores (3 MeshBlocks per rank, as on NAS); the hi-res twin is `-N 6 -n 768`.

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

### The beta=3 fragmentation continuation (with AMR)

The campaign plan: run the hi-res beta=10 anchor to saturation, then morph to
beta=3 -- fragmentation out of ESTABLISHED gravito-turbulence (the SC14
protocol; a from-scratch beta=3 run fragments during the initial cooling
collapse instead, which is a weaker statement) -- with adaptive refinement
following the fragments.  This is the configuration no FFT-based code can run.

Requirements already baked into the anchor input (do not undo):
- `nghost = 4` (AMR needs even; restart files store ghosts, so it cannot be
  changed at restart);
- dormant `<mesh_refinement>` / `<amr_criterion0>` blocks (refinement=none,
  rho > 10 criterion, 3 levels) -- present so the restart can enable them.

The continuation itself:

```bash
mpiexec -np $NRANKS -ppn $PPN $ATHENAK/build-cluster/src/athena \
  -r <anchor>/rst/<last>.rst -d ./ -t <guard> \
  hydro_srcterms/bcool_beta=3 time/tlim=<t_anchor_end + 20> \
  mesh_refinement/refinement=adaptive \
  gravity/threshold=-1.0 gravity/niteration=6 >> run.log 2>&1
```

Verified end to end at small scale (uniform nghost=4 run -> restart with
refinement=adaptive + beta flip): the tree rebuilds at root level and the
criterion starts refining as clumps form.  Notes: fragments drifting into the
two x1 boundary block-columns (|x| > 28H) will not refine (shear-face policy);
expect the timestep to collapse with the fragments -- the run is over, physics-
wise, once fragments pass the resolved ceiling even at 32 cells/H, so 20/Omega
of tlim is ample.  Load balancing under AMR is dynamic; keep max_nmb_per_rank
headroom (64 set) and expect the block count to grow by a few hundred.

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
## Backing up to Lou

`backup_run.sh <rundir>` archives a finished run to NAS mass storage as two
tars via `shiftc` (streamed -- no local copy -- checksummed, auto-retried):
`<name>_data.tar` (bin/ + rst/) and `<name>_meta.tar` (histories, logs, PBS
files, provenance -- small, so the history is retrievable without touching the
bulk).  `module load shift` first if `shiftc` is missing; monitor with
`shiftc --status`; restore with `shiftc --extract-tar lou:athenak_backups/<name>_data.tar <dest>`.

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

## 3. The Baehr, Zhu & Yang (2022) dust runs (dust track, two stages)

BR_L_10 / noBR_L_10 of their Table 1 (512^2 x 256, Lx = Ly = (80/pi) H_g,
Lz = (40/pi) H_g, beta = 10, Q0 = 1.02, St = 1, Z = 0.01, 1.5e6 particles), in SC14
code units with H_g = cs0/Omega = 2.16878.  The gas is run alone to the saturated
state first; the particles are inserted into a restart dump.

```bash
# stage 1: gas only to t = 50 (dumps every 10)
cd $SCRATCH/gtb/stage1 && sbatch $HOME/athenak-multigrid/scripts/cluster/gt_baehr_stage1.slurm
# stage 2: restart from the t = 50 dump, insert the particles, run to t = 100
cd $SCRATCH/gtb/BR_L_10   && BR=true  sbatch $HOME/athenak-multigrid/scripts/cluster/gt_baehr_stage2.slurm
cd $SCRATCH/gtb/noBR_L_10 && BR=false sbatch $HOME/athenak-multigrid/scripts/cluster/gt_baehr_stage2.slurm
# three species (St = 0.01, 0.1, 1; Z = 0.01/3 each): the dust3 add-on, same dump
cd $SCRATCH/gtb/noBR_3sp && BR=false ADDON=$HOME/athenak-multigrid/inputs/shearing_box/gravito_turb_baehr_dust3.athinput \
    sbatch $HOME/athenak-multigrid/scripts/cluster/gt_baehr_stage2.slurm
# ... with the radial pressure gradient (dust drift) on: OVR="hydro_srcterms/const_accel=true"
```

Multi-species layers: `<dust>/nspecies` and `taus_s` in the add-on, `problem/dust_Z_s`
per species (default `dust_Z/nspecies`); the species are interleaved in every cell so
each has the same count and distribution.  With back-reaction off the species are
independent, so one run is several single-species experiments.  A cheaper proof of
concept: rerun stage 1 with `mesh/nx1=256 mesh/nx2=256 mesh/nx3=128 meshblock/nx1=32
meshblock/nx2=32 meshblock/nx3=32` (0.1 H_g cells, 256 blocks = 2 nodes at 128 ranks
each, ~16x cheaper) and restart the dust stages from its dump with the same add-ons
and `particles/ppc` scaled to the new cell count (ppc is per cell: 0.02235 x 8 keeps
1.5e6 particles per species).

Stage 2 reads the add-on input `inputs/shearing_box/gravito_turb_baehr_dust.athinput`
on top of the dump's own input (`-r dump -i addon`: blocks merge, the add-on wins):
`<particles>/restart_insert = true` makes the reader skip the (absent) particle
section and the generator insert the layer (Gaussian in z of width H_g, uniform in
x,y, at rest, equal masses summing to Z x gas mass; deterministic and
decomposition-independent).  Dust physics: PC2 coupling under rk2, TSC, self-gravity
on, diode faces remove escaping particles (`DUST_ESCAPE_SUMMARY`, history column
`d_escaped`).  To continue a stage-2 run from its own dump (which carries the
particles) use `-r <dump> particles/restart_insert=false` without `-i`.  Outputs:
`hst` (SC14 columns + `d_mass d_px d_py d_pz d_ke d_wrey d_escaped`), `bin` of
`hydro_w`, `grav_phi` and `dust_dpm` (the module's own dust density), `pvtk`, and
`phst` (the particle history: momenta, kinetic energies, escaped mass, max dust density,
per-species means and dispersions -- `sig_z` is H_d).  Their Table 1 from a finished
run: `python3 scripts/analysis/baehr_table1.py <rundir> --t0 50 --t1 80 --roche`.

Cost (after the 2026-09-08 slab-plane rewrite, commit noted in the implementation
doc): the multigrid's slab boundary used to gather and transform the whole density on
every rank -- ~25 s per cycle for this box on 16 nodes, i.e. never finishing.  Each
rank now handles only its own planes (one 8 MB Allreduce per solve), and a 1M-cell
box runs at 1.1 s/cycle on one core.  The step is the floor halo's free fall,
dt ~ 3e-3 (the SC14 box: 4e-3), so ~17,000 cycles per 50/Omega; expect well under
1 s/cycle on 16 nodes (2048 ranks, 8 blocks each): a few hours per stage.  Check the
first `elapsed=` lines of run.log and scale down the node count if it is faster than
needed.  The earlier "memory-bandwidth-bound, 11.8 s*node/cycle" numbers of sec. 2 were
dominated by the old slab gather and no longer apply.
