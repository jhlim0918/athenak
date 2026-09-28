#!/bin/bash
#SBATCH -J gt_sc14_b10_amr
#SBATCH -p gg                    # Vista Grace-Grace CPU nodes (144 cores each, no GPU)
#SBATCH -N 4                     # nodes
#SBATCH -n 512                   # 6144 root MeshBlocks / 512 = 12 per rank, balanced on the
                                 #   unrefined mesh; 128 of 144 cores per node (the code is
                                 #   memory-bandwidth-bound, idle cores cost nothing).  Any
                                 #   divisor of 6144 works; the block cap below is PER RANK,
                                 #   so change the two together
#SBATCH -t 48:00:00              # gg queue maximum
#SBATCH -A AST25015
#SBATCH -o slurm.%j.out          # stdout+stderr merged
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jay.lim1@unlv.edu

# SC14 beta = 10 from a cold start on a MULTI-LEVEL adaptive mesh: the standard 4 cells/H
# root grid in 8^3 MeshBlocks, refined on the Jeans number (gravito_turb pgen,
# <amr_criterion0> method = user) to 16 cells/H (LEVELS=3) or 32 cells/H (LEVELS=4), on the
# hydrostatic z boundary.  The fragmentation-boundary experiment: the uniform 4 cells/H run
# does not fragment at beta = 10 and the uniform 8 cells/H run does.  See the header of
# inputs/shearing_box/gravito_turb_sc14_jeans_amr.athinput.
#
# COST -- read before submitting.  There is no sub-cycling, so each level halves dt for
# the whole mesh: LEVELS=3 takes ~2x the cycles of the uniform hi-res run (~3e5 for
# 300/Omega) on ~1/5 of its cells in blocks that cost ~2.4x more per cell, i.e. about the
# charge of the uniform hi-res anchor; LEVELS=4 about 3x that.  Block counts: 6144
# unrefined, ~45000 at the peak of the first collapse (t ~ 20, scaled from the 16H test
# box), ~10000 with one fragment (from the archived hi-res dumps).  It will NOT finish in one 48 h job; the self-continuation
# below handles the chain.
#
# THE BLOCK CAP.  <mesh_refinement>/max_nmb_per_rank sizes every array and whole-array
# operations run over the allocated blocks, used or not, so an oversized cap is paid on
# every cycle (measured: 5x oversize = 1.4x per cycle, 40x = 7x).  CAP defaults to 112 per
# rank (57344 blocks at 512 ranks), sized for the first collapse.  If the run stops on
#     "exceeds <mesh_refinement>/max_nmb_per_rank"
# resubmit with a larger one; it continues from the last restart dump:
#     sbatch --export=ALL,CAP=160 ...
# Once the collapse has passed (t > 30) the mesh holds ~10000 blocks: resubmit the
# continuation with CAP=32 (the cap can change at any restart).
#
# Submit FROM the run directory in $SCRATCH; outputs land there:
#     mkdir -p $SCRATCH/gt_sc14_amr/b10_L3 && cd $SCRATCH/gt_sc14_amr/b10_L3
#     sbatch $HOME/athenak-multigrid/scripts/cluster/gt_sc14_b10_jeans_amr.slurm
#
# The job CONTINUES ITSELF: if rst/*.rst exists in the submit directory it restarts from
# the newest one, otherwise it starts fresh.  Force a fresh start over existing dumps with
#     sbatch --export=ALL,FRESH=1 ...
# Other knobs, all through --export:
#     LEVELS=4        levels on a FRESH start (root + LEVELS-1 refinements)
#     BETA=20         cooling time, for the scan in beta
#     NJEANS=16       Jeans-number threshold on a FRESH start
#     TSTART=50       hold the root mesh until this time (saturated-state start)
#     ATHENAK=/tree   source tree (needs the gravito_turb Jeans criterion and hse_outflow)

set -u
ATHENAK=${ATHENAK:-$HOME/athenak-multigrid}
INPUT=$ATHENAK/inputs/shearing_box/gravito_turb_sc14_jeans_amr.athinput
FRESH=${FRESH:-0}
CAP=${CAP:-112}
LEVELS=${LEVELS:-3}
BETA=${BETA:-10.0}
NJEANS=${NJEANS:-16.0}
TSTART=${TSTART:-0.0}

ATH=$ATHENAK/build-cluster/src/athena
[ -x "$ATH" ] || { echo "no athena binary at $ATH -- build it first" >&2; exit 1; }
[ -r "$INPUT" ] || { echo "no input at $INPUT" >&2; exit 1; }

module load gcc openmpi fftw3

cd "$SLURM_SUBMIT_DIR"

# continue if there is anything to continue from
RST=""
if [ "$FRESH" != "1" ]; then RST=$(ls rst/*.rst 2>/dev/null | tail -1); fi
if [ -n "$RST" ]; then
  # the mesh, the levels and the criterion travel in the restart file; only the cap, which
  # is an allocation size, is reset here
  START=(-r "$RST")
  OVR=(mesh_refinement/max_nmb_per_rank=$CAP)
  MODE="continuing from $RST"
else
  START=(-i "$INPUT")
  OVR=(mesh_refinement/max_nmb_per_rank=$CAP mesh_refinement/num_levels=$LEVELS
       hydro_srcterms/bcool_beta=$BETA problem/amr_njeans=$NJEANS problem/amr_tstart=$TSTART)
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
