#!/bin/bash
#SBATCH -J build_growth
#SBATCH -p astro
#SBATCH -w h-2
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --mem=32G
#SBATCH -o build.%j.out
# MPI CPU build of the dust-growth source on h-2 (no FFT; the R1 runs need no gravity).
# The source is a tar copy of the local worktree athenak-worktrees/dust-growth (branch
# dust-growth off 9aaa5639 plus the uncommitted growth code), staged in ATHENAK.
# Submit from ATHENAK:  cd /data/limjay/athenak_growth && sbatch scripts/cluster/build_growth_h2.slurm
ATHENAK=${ATHENAK:-/data/limjay/athenak_growth}
CMAKE=/data/limjay/tools/cmake-3.30.5-linux-x86_64/bin/cmake
cd $ATHENAK
mkdir -p build-mpi && cd build-mpi
$CMAKE -DAthena_ENABLE_MPI=ON -DPROBLEM=built_in_pgens .. > cmake.log 2>&1 || exit 1
make -j 32 > make.log 2>&1 || exit 1
ls -l src/athena
