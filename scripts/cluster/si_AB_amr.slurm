#!/bin/bash
#SBATCH -J si_AB_amr
#SBATCH -p gg                    # Vista Grace-Grace CPU nodes (2 x 72 cores, no GPU)
#SBATCH -N 1
#SBATCH -n 64                    # 256 root MeshBlocks of 16^2 -> 4 per rank; ~10 once refined
#SBATCH -t 24:00:00
#SBATCH -A AST25015
#SBATCH -o slurm.%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jay.lim1@unlv.edu

# Johansen & Youdin (2007) run AB with ADAPTIVE refinement: root 256^2 = JY07's resolution,
# one refined level on the dense rims, particles split on refinement, t = 30.
# See inputs/dust/streaming_AB_amr256.athinput for the physics and the criterion.
#
# NEEDS BRANCH dust-smr (the AMR particle code is not on dust-multigrid): build it once as
#     cd $HOME && git clone -b dust-smr https://github.com/jhlim0918/athenak.git athenak-smr
#     (or: git -C $HOME/athenak-multigrid worktree add $HOME/athenak-smr dust-smr)
#     cd $HOME/athenak-smr && git submodule update --init kokkos
#     module load gcc openmpi fftw3          # BEFORE cmake: the default environment is nvhpc
#     mkdir build-cluster && cd build-cluster
#     cmake -DAthena_ENABLE_MPI=ON -DCMAKE_CXX_COMPILER=mpicxx .. && make -j 16
# and point ATHENAK below at it.  A bare `make -j` on the 144-core login node exhausts the
# process limit ("nvc++-Fatal-Could not fork", "/bin/sh: Operation not permitted"): always
# give -j a number.  Match the compiler of the build that ran the uniform jobs:
#     grep CMAKE_CXX_COMPILER: $HOME/athenak-multigrid/build-cluster/CMakeCache.txt
#
# COST: the laptop run at these settings did ~0.7 s per cycle on 8 Apple-M ranks with
# dt = 5.6e-5 once level 1 appears, i.e. ~400,000 cycles to t = 30 = ~80 rank-hours; on
# 64 Vista cores expect 3-6 h if it scales, so 24 h is margin.  rst/ is written every 5;
# the run chains itself: resubmit this script unchanged and it restarts from the last file.
#
# SUBMIT FROM THE RUN DIRECTORY in $SCRATCH -- outputs land in the submit directory:
#     mkdir -p $SCRATCH/si/AB_amr && cd $SCRATCH/si/AB_amr
#     sbatch $HOME/athenak-smr/scripts/cluster/si_AB_amr.slurm

ATHENAK=$HOME/athenak-smr
INPUT=$ATHENAK/inputs/dust/streaming_AB_amr256.athinput

# the modules the binary was built with -- mirror `module list` from the cmake shell
module load gcc openmpi fftw3

cd $SLURM_SUBMIT_DIR
mkdir -p rst
{
  echo "# job $SLURM_JOB_ID on $SLURM_JOB_NUM_NODES node(s): $SLURM_JOB_NODELIST, $SLURM_NTASKS ranks"
  echo "# started $(date); commit $(git -C $ATHENAK rev-parse --short HEAD)"
} >> run.log

# fresh start, or continue from the newest restart file (same rank count required: the
# restart header stores one particle count per rank).  -t stops 15 min before the limit.
LAST=$(ls rst/*.rst 2>/dev/null | tail -1)
if [ -z "$LAST" ]; then
  ibrun $ATHENAK/build-cluster/src/athena -i $INPUT -d ./ -t 23:45:00 >> run.log 2>&1
else
  echo "# restarting from $LAST" >> run.log
  ibrun $ATHENAK/build-cluster/src/athena -r $LAST -d ./ -t 23:45:00 >> run.log 2>&1
fi
