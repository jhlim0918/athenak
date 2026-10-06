#!/bin/bash
#SBATCH -J gt_sc14_wz_b10
#SBATCH -p gg                    # Vista Grace-Grace CPU nodes (144 cores each, no GPU)
#SBATCH -N 2                     # nodes
#SBATCH -n 256                   # 768 MeshBlocks / 256 = 3 per rank, as the plm hse ladder
#SBATCH -t 48:00:00              # gg queue maximum
#SBATCH -A AST25015
#SBATCH -o slurm.%j.out          # stdout+stderr merged
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jay.lim1@unlv.edu

# SC14 morphed beta ladder, STANDARD resolution (256 x 256 x 48, 4 cells/H), hse_outflow
# z boundary, with WENO-Z reconstruction.  Twin of gt_sc14_b10_hse.slurm (the plm ladder
# gt_sc14_stndrd_b{10,20,40,80}_hse); the ONLY physics/numerics change is
# hydro/reconstruct = wenoz (rk2, hllc, nghost = 4, niteration = 6, outputs all as before).
# Motivation: on the plm ladder alpha'/eq.21 = 1.00, 0.97, 0.90, 0.82 at beta = 10..80,
# i.e. the resolved stress supplies only 82% of the cooling at beta = 80.  If the missing
# ~0.002 Omega <U> of heating is angular momentum carried by numerical viscosity, a less
# dissipative reconstruction should shrink the gap.
#
# Three modes, picked automatically.  Submit FROM the run directory in $SCRATCH:
#
#  1. beta = 10 anchor, fresh start (t = 0 -> 300):
#       mkdir -p $SCRATCH/gt_sc14_hse_wz/b10 && cd $SCRATCH/gt_sc14_hse_wz/b10
#       sbatch $HOME/athenak-multigrid/scripts/cluster/gt_sc14_hse_wenoz.slurm
#
#  2. morph stage, restart from the PARENT's final dump with a new beta and tlim
#     (plm-ladder windows: 20: 300->450, 40: 450->750 from b20, 80: 750->1150 from b40):
#       mkdir -p $SCRATCH/gt_sc14_hse_wz/b20 && cd $SCRATCH/gt_sc14_hse_wz/b20
#       sbatch -J gt_sc14_wz_b20 --export=ALL,BETA=20,TLIM=450,PARENT=$(ls $SCRATCH/gt_sc14_hse_wz/b10/rst/*.rst | tail -1) \
#              $HOME/athenak-multigrid/scripts/cluster/gt_sc14_hse_wenoz.slurm
#
#  3. continuation: if rst/*.rst already exists in the submit directory the job restarts
#     from the newest one (after a 48 h walltime stop just `sbatch` again with the same
#     --export).  PARENT is ignored then.  FRESH=1 forces mode 1/2 over existing dumps.
#
# Optional: INTEGRATOR=rk3 if WENO-Z + rk2 turns out unstable in the cold-start collapse
# (that is then a second change -- note it); ATHENAK=/some/tree for another source tree.
#
# NB hse_outflow lives on dust-multigrid (35eee230); the multigrid branch lacks it.
# All overrides are command-line so the archived input stays byte-for-byte what produced
# the published runs (every overridden key exists in the input, as athena requires).

set -u
ATHENAK=${ATHENAK:-$HOME/athenak-multigrid}
INPUT=$ATHENAK/inputs/shearing_box/gravito_turb_sc14.athinput
BETA=${BETA:-10}
TLIM=${TLIM:-}
PARENT=${PARENT:-}
INTEGRATOR=${INTEGRATOR:-}
FRESH=${FRESH:-0}

ATH=$ATHENAK/build-cluster/src/athena
[ -x "$ATH" ] || { echo "no athena binary at $ATH -- build it first" >&2; exit 1; }
[ -r "$INPUT" ] || { echo "no input at $INPUT" >&2; exit 1; }

module load gcc openmpi fftw3

cd "$SLURM_SUBMIT_DIR"

# applied in every mode (re-asserted on restarts so a continuation can never silently
# fall back to the parent's settings)
COMMON=(mesh/ix3_bc=hse_outflow mesh/ox3_bc=hse_outflow hydro/reconstruct=wenoz
        hydro_srcterms/bcool_beta=$BETA)
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
  # nghost = 4 matches the plm hse anchor (WENO-Z itself needs >= 3)
  START=(-i "$INPUT"); OVR=(mesh/nghost=4 "${COMMON[@]}"); MODE="fresh start from $(basename "$INPUT")"
fi

{
  echo "# job ${SLURM_JOB_ID} on ${SLURM_JOB_NUM_NODES} node(s): ${SLURM_JOB_NODELIST}, ${SLURM_NTASKS} ranks"
  echo "# started $(date); tree $ATHENAK; commit $(git -C "$ATHENAK" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "# $MODE"
  echo "# overrides: ${OVR[*]}"
} >> run.log

# ibrun takes rank count and placement from -N/-n.  athena's -t guard stops 15 min before
# the Slurm limit so the final restart dump is written and the chain can continue.
ibrun "$ATH" "${START[@]}" -d ./ -t 47:45:00 "${OVR[@]}" >> run.log 2>&1
