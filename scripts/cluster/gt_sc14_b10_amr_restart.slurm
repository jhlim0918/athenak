#!/bin/bash
#SBATCH -J gt_sc14_b10_amr1
#SBATCH -p gg                    # Vista Grace-Grace CPU nodes (144 cores each, no GPU)
#SBATCH -N 2                     # nodes
#SBATCH -n 256                   # as the archived run: 768 root MeshBlocks / 256 = 3 per
                                 #   rank; up to 2336 blocks (9 per rank) when refined.
                                 #   The block cap below is PER RANK: change the two together
#SBATCH -t 48:00:00              # gg queue maximum
#SBATCH -A AST25015
#SBATCH -o slurm.%j.out          # stdout+stderr merged
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jay.lim1@unlv.edu

# The archived standard-resolution SC14 beta = 10 run on the hydrostatic z boundary,
# RESTARTED FROM ITS t = 0 DUMP with adaptive refinement switched on.  Same initial
# condition, same 16^3 MeshBlocks, same physics; the mesh refines on the Jeans number of
# the gravito_turb pgen.  This block layout admits ONE refinement level (|z| < 2H to
# 8 cells/H); see inputs/shearing_box/gravito_turb_sc14_amr_overlay.athinput, which is
# also where the one other change is made and explained (a tolerance on the multigrid
# defect in place of the fixed 6 V-cycles).
#
# Submit FROM a NEW run directory in $SCRATCH; outputs land there:
#     mkdir -p $SCRATCH/gt_sc14_amr/b10_restart_L2 && cd $SCRATCH/gt_sc14_amr/b10_restart_L2
#     sbatch $HOME/athenak-multigrid/scripts/cluster/gt_sc14_b10_amr_restart.slurm
# NOT from the archived run's directory: this job writes dumps with the archived run's
# basename and numbering, and would overwrite them.  The script refuses to run there.
#
# The job CONTINUES ITSELF: the first submission starts from RST0 with the overlay; once
# rst/*.rst exists in the submit directory it restarts from the newest of those, which
# already carry the refinement settings.  Force a new start from RST0 with
#     sbatch --export=ALL,FRESH=1 ...
# Other knobs, all through --export:
#     RST0=/path/file.rst   the dump to start from
#     CAP=12                <mesh_refinement>/max_nmb_per_rank (arrays are sized by it)
#     ATHENAK=/tree         source tree (needs the gravito_turb Jeans criterion)

set -u
ATHENAK=${ATHENAK:-$HOME/athenak-multigrid}
OVERLAY=$ATHENAK/inputs/shearing_box/gravito_turb_sc14_amr_overlay.athinput
RST0=${RST0:-/scratch/08184/jhlim/gt_sc14_hse/gt_sc14_stndrd_b10_hse/GravitoTurb.00000.rst}
FRESH=${FRESH:-0}
CAP=${CAP:-12}

ATH=$ATHENAK/build-cluster/src/athena
[ -x "$ATH" ] || { echo "no athena binary at $ATH -- build it first" >&2; exit 1; }
[ -r "$OVERLAY" ] || { echo "no overlay input at $OVERLAY" >&2; exit 1; }

module load gcc openmpi fftw3

cd "$SLURM_SUBMIT_DIR"
if [ "$(cd "$(dirname "$RST0")" && pwd -P)" = "$(pwd -P)" ]; then
  echo "refusing to run in the directory of the archived run ($(pwd -P))" >&2; exit 1
fi

# continue if there is anything to continue from
RST=""
if [ "$FRESH" != "1" ]; then RST=$(ls rst/*.rst 2>/dev/null | tail -1); fi
if [ -n "$RST" ]; then
  START=(-r "$RST")
  MODE="continuing from $RST"
else
  [ -r "$RST0" ] || { echo "no restart file at $RST0" >&2; exit 1; }
  START=(-r "$RST0" -i "$OVERLAY")
  MODE="start from $RST0 with $(basename "$OVERLAY")"
fi
OVR=(mesh_refinement/max_nmb_per_rank=$CAP)

{
  echo "# job ${SLURM_JOB_ID} on ${SLURM_JOB_NUM_NODES} node(s): ${SLURM_JOB_NODELIST}, ${SLURM_NTASKS} ranks"
  echo "# started $(date); tree $ATHENAK; commit $(git -C "$ATHENAK" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "# $MODE"
  echo "# overrides: ${OVR[*]}"
} >> run.log

# ibrun takes rank count and placement from -N/-n.  athena's -t guard stops 15 min before
# the Slurm limit so the final restart dump is written and the chain can continue.
ibrun "$ATH" "${START[@]}" -d ./ -t 47:45:00 "${OVR[@]}" >> run.log 2>&1
