#!/bin/bash
#SBATCH -J si_As
#SBATCH -p gg                    # Vista Grace-Grace CPU nodes (2 x 72 cores, no GPU)
#SBATCH -N 1
#SBATCH -n 128                   # 256 MeshBlocks of 16^2 -> 2 per rank
#SBATCH -t 48:00:00              # gg queue maximum
#SBATCH -A AST25015
#SBATCH -o slurm.%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jay.lim1@unlv.edu

# Yang & Zhu (2021) Model As, single species, UNIFORM 256^2, seeded with the (6,6)
# linear eigenmode -- the exact twin of the Athena (C) run
# tst/particle/athinput.streaming2d_modelAs.  See inputs/dust/streaming_modelAs.athinput.
#
# COST.  The box is 0.2 H at 256^2, so dt = 2.34e-4 and t = 1000 P = 6283/Omega needs
# 2.7e7 cycles.  Measured 0.072 s/cycle on 4 Apple-M cores => ~5e2 core-hours; on 128
# Vista cores expect 20-40 h, i.e. ONE job if it scales well and TWO if it does not.
# THIS RUN IS DESIGNED TO BE CHAINED: restarts are written every 50 P, and the
# resubmission line is at the bottom.  Check the rate in run.log after the first hour
# (ndiag = 10000 cycles) before trusting the estimate.
#
# A NOTE ON WHAT TO EXPECT: 256^2 gives ~43 cells per wavelength of the seeded mode,
# which is marginal for a second-order scheme.  Athena measured s = 0.0026-0.0029 here
# against the analytic 0.0071294 (36-40%), with saturation only near t = 1000 P; 512^2
# recovered 95%.  So a slow rate is the expected agreement between the two codes, not a
# defect.  For the converged number, resubmit with mesh/nx1=512 mesh/nx2=512 appended to
# the ibrun line (4x the cells and 4x the particles: budget ~8x the time, and raise -n).
#
# SUBMIT FROM THE RUN DIRECTORY in $SCRATCH:
#     mkdir -p $SCRATCH/si/modelAs && cd $SCRATCH/si/modelAs
#     sbatch $HOME/athenak-multigrid/scripts/cluster/si_modelAs.slurm

ATHENAK=$HOME/athenak-multigrid
INPUT=$ATHENAK/inputs/dust/streaming_modelAs.athinput

module load gcc openmpi fftw3

cd $SLURM_SUBMIT_DIR
mkdir -p rst
{
  echo "# job $SLURM_JOB_ID on $SLURM_JOB_NUM_NODES node(s): $SLURM_JOB_NODELIST, $SLURM_NTASKS ranks"
  echo "# started $(date); commit $(git -C $ATHENAK rev-parse --short HEAD)"
} >> run.log

# first job: start from the input.  For every later job, comment this line out and use
# the restart line below instead (or just run resubmit.sh, written at the end).
ibrun $ATHENAK/build-cluster/src/athena -i $INPUT -d ./ -t 47:45:00 >> run.log 2>&1

# TO CONTINUE (every job after the first):
#   ibrun $ATHENAK/build-cluster/src/athena -r $(ls rst/*.rst | tail -1) -t 47:45:00 >> run.log 2>&1
#
# Convenience: after the first job completes, this one-liner resubmits from the last
# restart without editing the script --
#   sed 's|-i $INPUT -d ./|-r $(ls rst/*.rst \| tail -1)|' \
#       $HOME/athenak-multigrid/scripts/cluster/si_modelAs.slurm > resubmit.slurm \
#     && sbatch resubmit.slurm
