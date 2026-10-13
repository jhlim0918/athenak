#!/bin/bash
#SBATCH -J si_As_wz
#SBATCH -p gg                    # Vista Grace-Grace CPU nodes (2 x 72 cores, no GPU)
#SBATCH -N 1
#SBATCH -n 128                   # 256 MeshBlocks of 16^2 -> 2 per rank
#SBATCH -t 48:00:00              # gg queue maximum
#SBATCH -A AST25015
#SBATCH -o slurm.%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jay.lim1@unlv.edu

# Yang & Zhu (2021) Model As, single species, uniform 256^2, seeded (6,6) eigenmode, to
# t = 1000 orbits -- the run of si_modelAs.slurm with the hydro scheme that passed the
# linear tests (2026-09 linB/ZY21, local_runs in active_projects/si_linmode_tests):
#
#     WENO-Z + HLLE + IMEX2+, CFL 0.25, 1 particle per cell, float64 vtk dumps
#
# (the tracked input has all of this except float32 dumps).  At
# CFL <= 0.3 WENO-Z grew linB at 100% of the analytic rate with residuals ~1e-5 of the
# mode, where ppmx + Roe stalled; at CFL 0.4 WENO-Z is unstable with rk2 -- keep 0.25.
#
# THE INPUT.  AthenaK's command line can only change parameters already in the file, and
# vtk_dbl is not in streaming_modelAs.athinput, so the FIRST job writes a run-local copy,
# input_wenoz.athinput, with vtk_dbl added and the scheme checked.  The tracked input
# is not modified.  Restarts never re-read the input (the .rst carries the parameters),
# so every later job simply continues from the newest rst/*.rst.
#
# COST.  dt = 1.95e-4 at CFL 0.25 (Model As box 0.2 H at 256^2), so 1000 P = 6283/Omega
# is 3.2e7 cycles.  Measured for the same scheme at 1 ppc on a 256^2 mesh: 0.7-0.9 ms per
# cycle on 128-256 AMD Zen 5 cores (h-3) -> ~6-9 h here if Grace is comparable; allow
# for 18 h.  Almost certainly ONE job; the script chains itself anyway (below).  Check
# the rate in run.log after the first hour (a line every ndiag = 10000 cycles).
#
# CHAINING.  athena stops at 47:45 wall time and writes a final restart; if the log says
# "Terminating on wall clock limit" the job resubmits this script from the same directory,
# at most MAX_JOBS times in total.  "Terminating on time limit" = done.
#
# BUILD.  Needs a build-cluster binary that knows <outputN>/vtk_dbl (AthenaK 3d421569,
# on dust-multigrid): cd $HOME/athenak-multigrid && git pull && cmake --build
# build-cluster -j 16   (modules loaded first -- see README.md).
#
# SUBMIT FROM THE RUN DIRECTORY in $SCRATCH:
#     mkdir -p $SCRATCH/si/modelAs_wenoz && cd $SCRATCH/si/modelAs_wenoz
#     sbatch $HOME/athenak-multigrid/scripts/cluster/si_modelAs_wenoz.slurm
#
# OTHER RESOLUTIONS.  NX (cells per side, default 256) and MB (MeshBlock side, default 16)
# are read from the environment; the first job passes them as mesh/meshblock overrides and
# records NX in grid.txt, which later jobs check.  512^2 has its own wrapper with the right
# job size: si_modelAs_wenoz_512.slurm.
#
# OTHER RECONSTRUCTIONS.  RECON (default wenoz) replaces the tracked input's reconstruct
# line; everything else is unchanged.  The derived input is input_$RECON.athinput and the
# choice is recorded in recon.txt, which later jobs check.  ppmx at 512^2 has its own
# wrapper: si_modelAs_ppmx_512.slurm.

ATHENAK=$HOME/athenak-multigrid
TRACKED=$ATHENAK/inputs/dust/streaming_modelAs.athinput
SCRIPT=${SCRIPT:-$ATHENAK/scripts/cluster/si_modelAs_wenoz.slurm}   # what the chain resubmits
NX=${NX:-256}                            # cells per side (the tracked input is 256)
MB=${MB:-16}                             # MeshBlock side
RECON=${RECON:-wenoz}                    # reconstruction (the tracked input is wenoz)
EXE=$ATHENAK/build-cluster/src/athena
INPUT=input_$RECON.athinput              # run-local derived input (first job writes it)
MAX_JOBS=4

module load gcc openmpi fftw3

cd $SLURM_SUBMIT_DIR
mkdir -p rst

die() { echo "ERROR: $*" | tee -a run.log; exit 1; }

# grid sanity: whole MeshBlocks, at least one per rank
[ $((NX % MB)) = 0 ] || die "NX=$NX is not a multiple of MB=$MB"
[ $(( (NX/MB)*(NX/MB) )) -ge ${SLURM_NTASKS:-1} ] || die "only $(( (NX/MB)*(NX/MB) )) MeshBlocks for $SLURM_NTASKS ranks"

# the binary must support float64 vtk output
grep -q vtk_dbl $EXE 2>/dev/null || strings $EXE | grep -q vtk_dbl \
  || die "$EXE has no vtk_dbl: git pull and rebuild build-cluster (see header)"

# --- first job: derive the input and verify every change -------------------------------
if ! ls rst/*.rst > /dev/null 2>&1; then
  grep -q '^reconstruct = wenoz' $TRACKED || die "tracked input no longer has reconstruct = wenoz"
  sed -e "s/^reconstruct = wenoz.*/reconstruct = $RECON         # set by si_modelAs_wenoz.slurm (RECON=$RECON)/" \
      -e '/^variable   = dust_dpm$/a\
vtk_dbl    = true' \
      -e '/^variable   = hydro_w$/a\
vtk_dbl    = true' \
      $TRACKED > $INPUT
  for want in "^reconstruct = $RECON " '^rsolver     = hlle ' '^cfl_number = 0.25 ' \
              '^ppc           = 1.0 ' '^integrator = imex2\+$' '^coupling      = imex$' \
              '^tlim       = 6283.185307179586'; do
    grep -qE "$want" $INPUT || die "derived input lacks '$want' -- has $TRACKED changed?"
  done
  [ "$(grep -c '^vtk_dbl    = true' $INPUT)" = 2 ] || die "derived input: vtk_dbl not set on both dumps"
  diff $TRACKED $INPUT > input_$RECON.diff
  echo $NX > grid.txt
  echo $RECON > recon.txt
elif [ "$(cat grid.txt 2>/dev/null)" != "$NX" ]; then
  die "this directory holds a $(cat grid.txt 2>/dev/null)^2 run but NX=$NX: wrong script for this run directory?"
elif [ "$(cat recon.txt 2>/dev/null || echo wenoz)" != "$RECON" ]; then
  die "this directory holds a $(cat recon.txt 2>/dev/null || echo wenoz) run but RECON=$RECON: wrong script for this run directory?"
fi

{
  echo "# job $SLURM_JOB_ID on $SLURM_JOB_NUM_NODES node(s): $SLURM_JOB_NODELIST, $SLURM_NTASKS ranks"
  echo "# started $(date); commit $(git -C $ATHENAK rev-parse --short HEAD); grid ${NX}^2, MeshBlocks ${MB}^2, reconstruct $RECON"
} >> run.log

if ls rst/*.rst > /dev/null 2>&1; then
  RST=$(ls rst/*.rst | tail -1)
  echo "# restarting from $RST" >> run.log
  ibrun $EXE -r $RST -d ./ -t 47:45:00 >> run.log 2>&1
else
  echo "# first job: $INPUT (differences from the tracked input in input_$RECON.diff)" >> run.log
  ibrun $EXE -i $INPUT -d ./ -t 47:45:00 \
    mesh/nx1=$NX mesh/nx2=$NX meshblock/nx1=$MB meshblock/nx2=$MB >> run.log 2>&1
fi
echo "# job $SLURM_JOB_ID ended $(date)" >> run.log

# --- chain: resubmit only after a wall-clock stop, at most MAX_JOBS jobs in total -------
echo $SLURM_JOB_ID >> jobs.list
last=$(grep -E "^Terminating on" run.log | tail -1)
if [ "$last" = "Terminating on wall clock limit" ]; then
  if [ $(wc -l < jobs.list) -lt $MAX_JOBS ]; then
    echo "# wall clock stop: resubmitting" >> run.log
    sbatch $SCRIPT
  else
    echo "# wall clock stop but MAX_JOBS=$MAX_JOBS reached: not resubmitting" >> run.log
  fi
else
  echo "# final state: ${last:-no termination message (crash? see slurm.$SLURM_JOB_ID.out)}" >> run.log
fi
