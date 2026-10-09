#!/bin/bash
#SBATCH -J be_collapse_ts23
#SBATCH -p astro                 # UNLV Physics, Zhu group partition (no time limit)
#SBATCH -w h-2                   # AMD EPYC 9825, 2 x 144 cores, 754 GB
#SBATCH -N 1
#SBATCH -n 128                   # the paper used 160 cores (30.7 h)
#SBATCH --hint=nomultithread     # one rank per physical core
#SBATCH --mem=200G
#SBATCH -o slurm.%j.out

# Collapse of a magnetized cloud, Tomida & Stone (2023) ApJS 266, 7, sec. 4.4 (Fig. 12):
# inputs/tests/be_collapse_mhd_ts23.athinput (be_collapse pgen, MHD + barotropic EOS +
# multigrid FMG with multipole BCs + Jeans AMR).  Submit FROM the run directory:
#   R=/data/limjay/runs/be_collapse_ts23
#   mkdir -p $R/run1 && cd $R/run1 && sbatch $R/be_collapse_ts23_h2.slurm
# If rst/*.rst exists the job continues from the newest one (FRESH=1 forces a new start).
#
# CAP = <mesh_refinement>/max_nmb_per_rank (default 64).  It sizes every array, so keep it
# a small multiple of the blocks per rank in use (512 root blocks / 128 ranks = 4 at
# t = 0); the run stops with a fatal error if the mesh outgrows it -- continue from the
# last restart with a larger CAP.
#
# STOP watchdog: the paper ends at rho_c = 1e-8 g/cc = 8.8e9 code units (t = 6.03).  A
# background loop reads rho_max from the history rows this job appends and kills mpirun
# once it exceeds RHO_STOP (default 8.8e9), logging "# STOP" in run.log; exit code 4.
# Dumps (mhd_w, mhd_bcc) every DUMP_DT (default 0.05), restarts every 0.25.
# Optional: ATHENAK, BUILD (default build-collapse), TLIM.

set -u
ATHENAK=${ATHENAK:-/data/limjay/athenak}
BUILD=${BUILD:-build-collapse}
INPUT=$ATHENAK/inputs/tests/be_collapse_mhd_ts23.athinput
CAP=${CAP:-64}
RHO_STOP=${RHO_STOP:-8.8e9}
DUMP_DT=${DUMP_DT:-0.05}
WD_INTERVAL=${WD_INTERVAL:-60}
TLIM=${TLIM:-}
FRESH=${FRESH:-0}

ATH=$ATHENAK/$BUILD/src/athena
[ -x "$ATH" ] || { echo "no athena binary at $ATH -- build it first" >&2; exit 1; }
[ -r "$INPUT" ] || { echo "no input at $INPUT" >&2; exit 1; }
cd "$SLURM_SUBMIT_DIR"

OVR=(mesh_refinement/max_nmb_per_rank=$CAP output2/dt=$DUMP_DT output3/dt=$DUMP_DT
     output4/dt=0.25)
[ -n "$TLIM" ] && OVR+=(time/tlim=$TLIM)
RST=""
if [ "$FRESH" != "1" ]; then RST=$(ls rst/*.rst 2>/dev/null | tail -1); fi
if [ -n "$RST" ]; then
  START=(-r "$RST"); MODE="continuing from $RST"
else
  START=(-i "$INPUT"); MODE="fresh start from $(basename "$INPUT")"
fi

{
  echo "# job ${SLURM_JOB_ID} on $(hostname), ${SLURM_NTASKS} ranks"
  echo "# started $(date); tree $ATHENAK; commit $(git -C "$ATHENAK" rev-parse --short HEAD 2>/dev/null || echo unknown)$(git -C "$ATHENAK" diff --quiet HEAD -- src inputs 2>/dev/null || echo ' + uncommitted changes in src/inputs')"
  echo "# binary $ATH"
  echo "# $MODE"
  echo "# overrides: ${OVR[*]}"
} >> run.log

HST=be_collapse.user.hst
H0=$([ -f "$HST" ] && wc -l < "$HST" || echo 0)
rm -f STOP_RHO

mpirun -np "$SLURM_NTASKS" --bind-to core "$ATH" "${START[@]}" -d ./ "${OVR[@]}" \
  >> run.log 2>&1 &
MPI_PID=$!

(
  while kill -0 "$MPI_PID" 2>/dev/null; do
    sleep "$WD_INTERVAL"
    [ -f "$HST" ] || continue
    if tail -n +$((H0 + 1)) "$HST" | grep -v '^#' |
       awk -v s="$RHO_STOP" '$3+0 >= s+0 {f=1} END {exit !f}'; then
      echo "# STOP $(date): rho_max >= $RHO_STOP -- stopping the run; last rows:" >> run.log
      tail -n 3 "$HST" >> run.log
      touch STOP_RHO
      kill -TERM "$MPI_PID" 2>/dev/null; sleep 30; kill -KILL "$MPI_PID" 2>/dev/null
      exit 0
    fi
    if tail -n +$((H0 + 1)) "$HST" | grep -qi 'nan'; then
      echo "# WATCHDOG $(date): nan in $HST -- killing the run" >> run.log
      touch STOP_RHO
      kill -TERM "$MPI_PID" 2>/dev/null; sleep 30; kill -KILL "$MPI_PID" 2>/dev/null
      exit 0
    fi
  done
) &
WD_PID=$!

wait "$MPI_PID"; RC=$?
{ kill "$WD_PID"; wait "$WD_PID"; } 2>/dev/null
[ -f STOP_RHO ] && RC=4
echo "# ended $(date), exit $RC" >> run.log
exit $RC
