#!/bin/bash
#SBATCH -J gt_sc14_b10_hse
#SBATCH -p gg                    # Vista Grace-Grace CPU nodes (144 cores each, no GPU)
#SBATCH -N 2                     # nodes
#SBATCH -n 256                   # 768 MeshBlocks / 256 = 3 per rank, as the archived
                                 #   b10 anchor ran.  128 of 144 cores per node: the code
                                 #   is memory-bandwidth-bound, so idle cores cost nothing
#SBATCH -t 48:00:00              # gg queue maximum
#SBATCH -A AST25015
#SBATCH -o slurm.%j.out          # stdout+stderr merged
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jay.lim1@unlv.edu

# SC14 beta = 10 anchor, STANDARD resolution (256 x 256 x 48, 4 cells/H), with the new
# hydrostatic z boundary.  Re-founds the morphed ladder on a vertical BC that does not
# leak: `diode` turns the ghost into a flat reservoir at the edge density and the Riemann
# solver draws mass in from it, which destroyed every beta >= 80 run (see HANDOFF_ZBC.md).
# hse_outflow extrapolates the ghost hydrostatically under -Omega^2 z instead, so the edge
# cell stays supported.  At beta = 10 it should be indistinguishable from diode -- the halo
# sits on the density floor the whole run -- which is exactly why this is the right anchor
# to revalidate against the archived b10_ng4 numbers before trusting it at high beta.
#
# Submit FROM the run directory in $SCRATCH; outputs land there:
#     mkdir -p $SCRATCH/gt_sc14_hse/b10 && cd $SCRATCH/gt_sc14_hse/b10
#     sbatch $HOME/athenak-multigrid/scripts/cluster/gt_sc14_b10_hse.slurm
#
# The job CONTINUES ITSELF: if rst/*.rst exists in the submit directory it restarts from
# the newest one, otherwise it starts fresh.  So a 300/Omega run that outlives 48 h is just
# `sbatch` again.  Force a fresh start over existing dumps with
#     sbatch --export=ALL,FRESH=1 ...
# Point at a different source tree with
#     sbatch --export=ALL,ATHENAK=/some/tree ...
#
# NB hse_outflow lives on the dust-multigrid branch (35eee230); the multigrid branch does
# NOT have it, so ATHENAK must be the dust tree.  That tree also carries the corrected,
# ~75x faster slab solver (0520aeed) and the <particles> restart-block fix (e02cb8b5).

set -u
ATHENAK=${ATHENAK:-$HOME/athenak-multigrid}
INPUT=$ATHENAK/inputs/shearing_box/gravito_turb_sc14.athinput
FRESH=${FRESH:-0}

ATH=$ATHENAK/build-cluster/src/athena
[ -x "$ATH" ] || { echo "no athena binary at $ATH -- build it first" >&2; exit 1; }
[ -r "$INPUT" ] || { echo "no input at $INPUT" >&2; exit 1; }

module load gcc openmpi fftw3

cd "$SLURM_SUBMIT_DIR"

# the vertical BC, overridden on the command line so the archived input stays byte-for-byte
# what produced the published runs (both keys exist in <mesh>, so the override resolves)
BC=(mesh/ix3_bc=hse_outflow mesh/ox3_bc=hse_outflow)

# continue if there is anything to continue from
RST=""
if [ "$FRESH" != "1" ]; then RST=$(ls rst/*.rst 2>/dev/null | tail -1); fi
if [ -n "$RST" ]; then
  # nghost is baked into the restart's data layout -- never override it on a continuation
  START=(-r "$RST")
  OVR=("${BC[@]}")
  MODE="continuing from $RST"
else
  # nghost = 4 matches the archived b10_ng4 anchor and keeps an AMR beta=3 branch possible
  # (the refinement machinery needs an even nghost).  Drop to 3 for ~30% cheaper cycles if
  # you will never refine this run.
  START=(-i "$INPUT")
  OVR=(mesh/nghost=4 "${BC[@]}")
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
