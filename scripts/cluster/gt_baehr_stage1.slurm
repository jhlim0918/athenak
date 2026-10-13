#!/bin/bash
#SBATCH -J gtb_stage1
#SBATCH -p gg                    # Vista Grace-Grace CPU nodes (2 x 72 cores, no GPU)
#SBATCH -N 16                    # nodes: 512^2x256 = 16384 blocks of 16^3
#SBATCH -n 2048                  # 8 MeshBlocks per rank (memory-bandwidth-bound:
                                 #   the same decomposition as the NAS run.  128 of the
                                 #   144 cores per node: the code is memory-bandwidth-
                                 #   bound, so the idle cores cost nothing (README sec. 2)
#SBATCH -t 48:00:00              # gg queue maximum
#SBATCH -A AST25015              # allocation to charge
#SBATCH -o slurm.%j.out          # stdout+stderr (no -e => merged, like PBS -j oe)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jay.lim1@unlv.edu

# STAGE 1 of the Baehr+22 dust runs: the gas alone to t = 50 (saturation).  Twin of
# gt_sc14_full.slurm at 21x the cells.  Cost: after the slab-plane rewrite (README
# sec. 3) expect well under 1 s per cycle on 16 nodes; the step is set by the
# free-falling floor halo, dt ~ 3e-3 (the SC14 box runs at 4e-3): ~17,000 cycles to
# t = 50 -> a few hours.  Stage 2 (particles, PC2 coupling, dust exchanges) ~1.3x.  Submit from the RUN directory in
# $SCRATCH -- outputs land there and Slurm starts the job in the submit directory:
#     cd $SCRATCH/gt_sc14_stndrd/b10 && sbatch $HOME/athenak-multigrid/scripts/cluster/gt_sc14_full.slurm
# To continue a run (walltime kill, or to run past tlim), resubmit from the same
# directory with CONT=1 (and TLIM if the target time changes); see below.
# Per-run parameters go on the ibrun line, e.g.  hydro_srcterms/bcool_beta=5
# Hi-res twin: INPUT=.../gravito_turb_sc14_hi.athinput with -N 6 -n 768 (README sec. 2).

ATHENAK=$HOME/athenak-multigrid
INPUT=$ATHENAK/inputs/shearing_box/gravito_turb_baehr.athinput

# CONTINUE an existing stage-1 run instead of starting one: set RST (or CONT=1 to pick
# the run's own last dump) and, to run past the input's tlim, TLIM.  Every key overridden
# here is present in the tracked input, so the command line can set it:
#     cd $SCRATCH/gtb/stage1 && CONT=1 TLIM=100 sbatch .../gt_baehr_stage1.slurm
RST=${RST:-}
if [ -n "${CONT:-}" ] && [ -z "$RST" ]; then RST=$(ls rst/*.rst | tail -1); fi
if [ -n "$RST" ]; then START=(-r "$RST"); else START=(-i "$INPUT"); fi
if [ -n "${TLIM:-}" ]; then OVR=(time/tlim=$TLIM); else OVR=(); fi

# the modules the binary was built with -- mirror `module list` from the cmake shell
module load gcc openmpi fftw3

cd $SLURM_SUBMIT_DIR
{
  echo "# job $SLURM_JOB_ID on $SLURM_JOB_NUM_NODES node(s): $SLURM_JOB_NODELIST, $SLURM_NTASKS ranks"
  echo "# started $(date); commit $(git -C $ATHENAK rev-parse --short HEAD)"
  echo "# start ${START[*]} ${OVR[*]}"
} >> run.log

# ibrun is TACC's launcher: rank count and placement come from -N/-n above.
# athena's -t guard stops 15 min before the Slurm limit so the final rst is written.
ibrun $ATHENAK/build-cluster/src/athena "${START[@]}" -d ./ -t 47:45:00 "${OVR[@]}" >> run.log 2>&1
