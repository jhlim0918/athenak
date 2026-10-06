#!/bin/bash
#SBATCH -J gt_sc14_wzf_b10
#SBATCH -p astro                 # UNLV Physics, Zhu group partition (no time limit)
#SBATCH -w h-3                   # AMD EPYC 9825, 2 x 144 cores (576 threads), 754 GB
#SBATCH -N 1
#SBATCH -n 256                   # 768 MeshBlocks / 256 = 3 per rank, as the plm hse ladder
#SBATCH --hint=nomultithread     # one rank per physical core
#SBATCH --mem=96G                # ~6 GB of mesh data + MPI buffers; the node has 754 GB
#SBATCH -o slurm.%j.out          # stdout+stderr merged

# SC14 morphed beta ladder at STANDARD resolution (256 x 256 x 48, 4 cells/H), hse_outflow
# z boundary, WENO-Z reconstruction WITH first-order flux correction (FOFC).
# Changes from the plm ladder gt_sc14_stndrd_b{10,20,40,80}_hse:
#   hydro/reconstruct = wenoz, hydro/fofc = true      (rk2, hllc, nghost = 4, niteration = 6)
#   hydro_w and grav_phi dumps every 2/Omega (was 10/Omega)
# Why FOFC: plain WENO-Z (job 164) went NaN at t = 15.3, in the cold-start collapse --
# WENO-Z has no positivity guarantee, and one bad cell reaches the whole box through the
# next multigrid solve.  FOFC predicts each stage's update and, in any cell that would
# need a density/pressure floor, replaces its face fluxes with first-order LLF fluxes.
#
# THE INPUT: gravito_turb_sc14.athinput has no <hydro>/fofc key and athena rejects a
# command-line override of a key the input lacks.  A fresh start therefore runs from a
# frozen copy, ./gravito_turb_sc14.athinput, that is the tree's file plus one inserted line
# "fofc = false" (the default) -- the tree stays clean and the run directory records
# exactly what ran.  Restarts carry the key in the .rst parameter block.
#
# WATCHDOG: athena has no NaN check (job 164 ran on NaNs for 1.5 h, dt pinned at 0.06).
# A background loop greps the tail of the history files every WD_INTERVAL s (default 60)
# and kills mpirun at the first nan, logging "# WATCHDOG" in run.log.  Exit code 3.
#
# Three modes, picked automatically.  Submit FROM the run directory:
#
#  1. beta = 10 anchor, fresh start (t = 0 -> 300):
#       R=/data/limjay/runs/gt_sc14_hse_wzfofc
#       mkdir -p $R/b10 && cd $R/b10 && sbatch $R/gt_sc14_hse_wenoz_fofc_h3.slurm
#
#  2. morph stage, restart from the PARENT's final dump with a new beta and tlim
#     (plm-ladder windows: 20: 300->450 from b10, 40: 450->750 from b20, 80: 750->1150 from b40):
#       mkdir -p $R/b20 && cd $R/b20
#       sbatch -J gt_sc14_wzf_b20 --export=ALL,BETA=20,TLIM=450,PARENT=$(ls $R/b10/rst/*.rst | tail -1) $R/gt_sc14_hse_wenoz_fofc_h3.slurm
#
#  3. continuation: if rst/*.rst already exists in the submit directory the job restarts
#     from the newest one (PARENT ignored).  FRESH=1 forces mode 1/2 over existing dumps.
#     After a WATCHDOG kill, check that the newest rst predates the NaN before continuing.
#
# Optional: INTEGRATOR=rk3 (a further change -- note it); DUMP_DT (default 2);
# WD_INTERVAL; ATHENAK=/some/tree for another source tree.

set -u
ATHENAK=${ATHENAK:-/data/limjay/athenak}
TREE_INPUT=$ATHENAK/inputs/shearing_box/gravito_turb_sc14.athinput
BETA=${BETA:-10}
TLIM=${TLIM:-}
PARENT=${PARENT:-}
INTEGRATOR=${INTEGRATOR:-}
DUMP_DT=${DUMP_DT:-2}
WD_INTERVAL=${WD_INTERVAL:-60}
FRESH=${FRESH:-0}

ATH=$ATHENAK/build-mpi/src/athena
[ -x "$ATH" ] || { echo "no athena binary at $ATH -- build it first" >&2; exit 1; }
[ -r "$TREE_INPUT" ] || { echo "no input at $TREE_INPUT" >&2; exit 1; }

cd "$SLURM_SUBMIT_DIR"

COMMON=(mesh/ix3_bc=hse_outflow mesh/ox3_bc=hse_outflow
        hydro/reconstruct=wenoz hydro/fofc=true
        hydro_srcterms/bcool_beta=$BETA
        output2/dt=$DUMP_DT output3/dt=$DUMP_DT)
[ -n "$TLIM" ] && COMMON+=(time/tlim=$TLIM)
[ -n "$INTEGRATOR" ] && COMMON+=(time/integrator=$INTEGRATOR)

RST=""
if [ "$FRESH" != "1" ]; then RST=$(ls rst/*.rst 2>/dev/null | tail -1); fi
if [ -n "$RST" ]; then
  # nghost is baked into the restart's data layout -- never override it on a restart
  START=(-r "$RST"); OVR=("${COMMON[@]}"); MODE="continuing from $RST"
elif [ -n "$PARENT" ]; then
  [ -r "$PARENT" ] || { echo "PARENT $PARENT not readable" >&2; exit 1; }
  [ -n "$TLIM" ] || { echo "a morph stage needs TLIM (parent end + stage length)" >&2; exit 1; }
  START=(-r "$PARENT"); OVR=("${COMMON[@]}"); MODE="morph to beta=$BETA from $PARENT"
else
  [ "$BETA" = "10" ] || { echo "fresh start is the beta=10 anchor; set PARENT for beta=$BETA" >&2; exit 1; }
  INPUT=./$(basename "$TREE_INPUT")
  if ! grep -q '^ *fofc *=' "$TREE_INPUT"; then
    sed '/^ *reconstruct *=/a\
fofc        = false   # first-order flux correction (key added for command-line override)' \
      "$TREE_INPUT" > "$INPUT"
  else
    cp "$TREE_INPUT" "$INPUT"
  fi
  grep -q '^ *fofc *=' "$INPUT" || { echo "failed to add fofc to $INPUT" >&2; exit 1; }
  START=(-i "$INPUT"); OVR=(mesh/nghost=4 "${COMMON[@]}"); MODE="fresh start from $INPUT (tree copy + fofc key)"
fi

{
  echo "# job ${SLURM_JOB_ID} on $(hostname), ${SLURM_NTASKS} ranks"
  echo "# started $(date); tree $ATHENAK; commit $(git -C "$ATHENAK" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "# $MODE"
  echo "# overrides: ${OVR[*]}"
} >> run.log

# NaN watchdog.  It scans only the rows THIS job appends (history files are appended
# across restarts, so rows from an earlier failed job must not trigger it).
rm -f WATCHDOG_NAN
nrows0() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
U0=$(nrows0 GravitoTurb.user.hst); H0=$(nrows0 GravitoTurb.hydro.hst)

mpirun -np "$SLURM_NTASKS" --bind-to core "$ATH" "${START[@]}" -d ./ "${OVR[@]}" >> run.log 2>&1 &
MPI_PID=$!

(
  while kill -0 "$MPI_PID" 2>/dev/null; do
    sleep "$WD_INTERVAL"
    for hn in "GravitoTurb.user.hst $U0" "GravitoTurb.hydro.hst $H0"; do
      set -- $hn; h=$1; n0=$2
      [ -f "$h" ] || continue
      if tail -n +$((n0 + 1)) "$h" | grep -qi 'nan'; then
        echo "# WATCHDOG $(date): nan in $h -- killing the run; last rows:" >> run.log
        tail -n 3 "$h" | cut -c1-120 >> run.log
        touch WATCHDOG_NAN
        kill -TERM "$MPI_PID" 2>/dev/null; sleep 30; kill -KILL "$MPI_PID" 2>/dev/null
        exit 0
      fi
    done
  done
) &
WD_PID=$!

wait "$MPI_PID"; RC=$?
{ kill "$WD_PID"; wait "$WD_PID"; } 2>/dev/null
[ -f WATCHDOG_NAN ] && RC=3
echo "# ended $(date), exit $RC" >> run.log
exit $RC
