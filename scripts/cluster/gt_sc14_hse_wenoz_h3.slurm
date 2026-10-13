#!/bin/bash
#SBATCH -J gt_sc14_wz_b10
#SBATCH -p astro                 # UNLV Physics, Zhu group partition (no time limit)
#SBATCH -w h-3                   # AMD EPYC 9825, 2 x 144 cores (576 threads), 754 GB
#SBATCH -N 1
#SBATCH -n 256                   # 768 MeshBlocks / 256 = 3 per rank, as the plm hse ladder
#SBATCH --hint=nomultithread     # one rank per physical core
#SBATCH --mem=96G                # ~6 GB of mesh data + MPI buffers; the node has 754 GB
#SBATCH -o slurm.%j.out          # stdout+stderr merged

# UNLV h-3 twin of gt_sc14_hse_wenoz.slurm (Vista).  SC14 morphed beta ladder at STANDARD
# resolution (256 x 256 x 48, 4 cells/H), hse_outflow z boundary, WENO-Z reconstruction.
# The only change from the plm ladder gt_sc14_stndrd_b{10,20,40,80}_hse is
# hydro/reconstruct = wenoz (rk2, hllc, nghost = 4, niteration = 6, outputs unchanged).
#
# Three modes, picked automatically.  Submit FROM the run directory:
#
#  1. beta = 10 anchor, fresh start (t = 0 -> 300):
#       mkdir -p /data/limjay/runs/gt_sc14_hse_wz/b10 && cd /data/limjay/runs/gt_sc14_hse_wz/b10
#       sbatch /data/limjay/runs/gt_sc14_hse_wz/gt_sc14_hse_wenoz_h3.slurm
#
#  2. morph stage, restart from the PARENT's final dump with a new beta and tlim
#     (plm-ladder windows: 20: 300->450 from b10, 40: 450->750 from b20, 80: 750->1150 from b40):
#       R=/data/limjay/runs/gt_sc14_hse_wz
#       mkdir -p $R/b20 && cd $R/b20
#       sbatch -J gt_sc14_wz_b20 --export=ALL,BETA=20,TLIM=450,PARENT=$(ls $R/b10/rst/*.rst | tail -1) $R/gt_sc14_hse_wenoz_h3.slurm
#
#  3. continuation: if rst/*.rst already exists in the submit directory the job restarts
#     from the newest one (PARENT ignored).  FRESH=1 forces mode 1/2 over existing dumps.
#
# Optional: INTEGRATOR=rk3 if WENO-Z + rk2 is unstable in the cold-start collapse (a second
# change -- note it); ATHENAK=/some/tree for another source tree.
# All overrides are command-line so the input stays byte-for-byte the archived one.

set -u
ATHENAK=${ATHENAK:-/data/limjay/athenak}
INPUT=$ATHENAK/inputs/shearing_box/gravito_turb_sc14.athinput
BETA=${BETA:-10}
TLIM=${TLIM:-}
PARENT=${PARENT:-}
INTEGRATOR=${INTEGRATOR:-}
FRESH=${FRESH:-0}

ATH=$ATHENAK/build-mpi/src/athena
[ -x "$ATH" ] || { echo "no athena binary at $ATH -- build it first" >&2; exit 1; }
[ -r "$INPUT" ] || { echo "no input at $INPUT" >&2; exit 1; }

cd "$SLURM_SUBMIT_DIR"

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
  START=(-i "$INPUT"); OVR=(mesh/nghost=4 "${COMMON[@]}"); MODE="fresh start from $(basename "$INPUT")"
fi

{
  echo "# job ${SLURM_JOB_ID} on $(hostname), ${SLURM_NTASKS} ranks"
  echo "# started $(date); tree $ATHENAK; commit $(git -C "$ATHENAK" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "# $MODE"
  echo "# overrides: ${OVR[*]}"
} >> run.log

mpirun -np "$SLURM_NTASKS" --bind-to core "$ATH" "${START[@]}" -d ./ "${OVR[@]}" >> run.log 2>&1
echo "# ended $(date), exit $?" >> run.log
