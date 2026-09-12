#!/bin/bash
#SBATCH -J si_BA
#SBATCH -p gg                    # Vista Grace-Grace CPU nodes (2 x 72 cores, no GPU)
#SBATCH -N 1
#SBATCH -n 64                    # 256 MeshBlocks of 16^2 -> 4 per rank
#SBATCH -t 12:00:00
#SBATCH -A AST25015
#SBATCH -o slurm.%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jay.lim1@unlv.edu

# Johansen & Youdin (2007) run BA at their published configuration, UNIFORM 256^2:
# tau_s = 1, eps = 0.2, 40 eta r box, 1.638e6 particles (25 per cell), t = 500/Omega.
# See inputs/dust/streaming_BA_uniform.athinput for the physics.
#
# COST (measured on 4 Apple-M cores: 0.237 s/cycle, dt = 2.3e-3, so 2.15e5 cycles
# = 14 core-hours-ish):  on 64 Vista cores expect 1-3 h.  12 h is deliberate margin --
# dt falls somewhat once the filaments form.  If it does hit the wall, restart with the
# line at the bottom; rst/ is written every 25/Omega.
#
# SUBMIT FROM THE RUN DIRECTORY in $SCRATCH -- outputs land in the submit directory:
#     mkdir -p $SCRATCH/si/BA && cd $SCRATCH/si/BA
#     sbatch $HOME/athenak-multigrid/scripts/cluster/si_BA_uniform.slurm

ATHENAK=$HOME/athenak-multigrid
INPUT=$ATHENAK/inputs/dust/streaming_BA_uniform.athinput

# the modules the binary was built with -- mirror `module list` from the cmake shell
module load gcc openmpi fftw3

cd $SLURM_SUBMIT_DIR
mkdir -p rst
{
  echo "# job $SLURM_JOB_ID on $SLURM_JOB_NUM_NODES node(s): $SLURM_JOB_NODELIST, $SLURM_NTASKS ranks"
  echo "# started $(date); commit $(git -C $ATHENAK rev-parse --short HEAD)"
} >> run.log

# -t stops the code 15 min before the Slurm limit so the final restart is written
ibrun $ATHENAK/build-cluster/src/athena -i $INPUT -d ./ -t 11:45:00 >> run.log 2>&1

# TO CONTINUE after a walltime kill, replace the ibrun line above with:
#   ibrun $ATHENAK/build-cluster/src/athena -r $(ls rst/*.rst | tail -1) -t 11:45:00 >> run.log 2>&1
