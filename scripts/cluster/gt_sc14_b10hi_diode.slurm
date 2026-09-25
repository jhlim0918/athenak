#!/bin/bash
#SBATCH -J gt_sc14_b10hi_diode
#SBATCH -p gg                    # Vista Grace-Grace CPU nodes (144 cores each, no GPU)
#SBATCH -N 6                     # nodes.  768 ranks needs 6: Vista Grace nodes have 144
#SBATCH -n 768                   # cores, so the NAS 3-node/256-per-node launch is impossible
                                 #   here.  768 blocks / 768 ranks = 1 block/rank, the
                                 #   balanced launch; 128 of 144 cores per node, and the
                                 #   code is memory-bandwidth-bound so idle cores are free.
                                 #   NEVER 512 ranks: 1.5 blocks/rank wastes 25% of the
                                 #   charge to load imbalance (README sec. 2)
#SBATCH -t 48:00:00              # gg queue maximum
#SBATCH -A AST25015
#SBATCH -o slurm.%j.out          # stdout+stderr merged
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jay.lim1@unlv.edu

# SC14 beta = 10 anchor, HI resolution (512 x 512 x 96, 8 cells/H -- SC14's ".hi"), with
# the DIODE z boundary: the control for gt_sc14_b10hi_hse.slurm.  Same input, tree,
# binary, rank layout and chaining -- only the vertical BC differs.  The hse run
# fragmented (one midplane clump with 10% of the mass, rho_max = 387 at t = 170) and lost
# 42% of its mass by t = 170, where standard resolution with either BC does neither.
# This run separates the two candidate causes: if diode at 8 cells/H also fragments,
# the fragmentation is resolution, not the boundary.  Mass loss for comparison (t = 40-170):
# std diode 0.044 %/Omega, std hse 0.055, hi hse 0.24.
#
# COST -- read before submitting.  8x the cells of the standard run: the beta = 10 anchor
# alone was ~4,700 SBU on NAS Turin, and dt ~ 1.4e-3 (transient) to 2.2e-3 (steady) means
# about 155k cycles for 300/Omega.  It will NOT finish in one 48 h job; expect a chain of
# several, which the self-continuation below handles.  Output volume is the other surprise:
# rst every 25/Omega at ~3.4 GB each plus hydro_w/grav_phi every 5/Omega -- roughly 75 GB
# for the full anchor, so check your $SCRATCH headroom first.  The standard-resolution twin
# delivers most of the science at ~1/10 the cost.
#
# Submit FROM the run directory in $SCRATCH; outputs land there:
#     mkdir -p $SCRATCH/gt_sc14_hse/b10hi_diode && cd $SCRATCH/gt_sc14_hse/b10hi_diode
#     sbatch $HOME/athenak-multigrid/scripts/cluster/gt_sc14_b10hi_diode.slurm
#
# The job CONTINUES ITSELF: if rst/*.rst exists in the submit directory it restarts from
# the newest one, otherwise it starts fresh.  So a 300/Omega run that outlives 48 h is just
# `sbatch` again -- which at this resolution you will do several times.  Force a fresh start over existing dumps with
#     sbatch --export=ALL,FRESH=1 ...
# Point at a different source tree with
#     sbatch --export=ALL,ATHENAK=/some/tree ...
#
# ATHENAK defaults to the dust tree on purpose, although diode exists on both branches:
# the hse run used it, so the two runs differ only in the BC.

set -u
ATHENAK=${ATHENAK:-$HOME/athenak-multigrid}
INPUT=$ATHENAK/inputs/shearing_box/gravito_turb_sc14_hi.athinput
FRESH=${FRESH:-0}

ATH=$ATHENAK/build-cluster/src/athena
[ -x "$ATH" ] || { echo "no athena binary at $ATH -- build it first" >&2; exit 1; }
[ -r "$INPUT" ] || { echo "no input at $INPUT" >&2; exit 1; }

module load gcc openmpi fftw3

cd "$SLURM_SUBMIT_DIR"

# the vertical BC: the input already has diode; restated so run.log records it (both keys
# exist in <mesh>, so the override resolves)
BC=(mesh/ix3_bc=diode mesh/ox3_bc=diode)

# continue if there is anything to continue from
RST=""
if [ "$FRESH" != "1" ]; then RST=$(ls rst/*.rst 2>/dev/null | tail -1); fi
if [ -n "$RST" ]; then
  # nghost is baked into the restart's data layout -- never override it on a continuation
  START=(-r "$RST")
  OVR=("${BC[@]}")
  MODE="continuing from $RST"
else
  # no nghost override here: the hi-res input already sets nghost = 4, and with 32^3
  # MeshBlocks that is only ~1.95x ghost overhead (vs 3.4x for 16^3 at the standard res)
  START=(-i "$INPUT")
  OVR=("${BC[@]}")
  MODE="fresh start from $(basename "$INPUT")"
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
