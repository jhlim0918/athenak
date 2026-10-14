#!/bin/bash
#SBATCH -J s24_stage1
#SBATCH -p gg                    # Vista Grace-Grace CPU nodes (2 x 72 cores, no GPU)
#SBATCH -N 1
#SBATCH -n 64                    # one rank per root MeshBlock (8 x 8 x 1 of 16^3)
#SBATCH -t 04:00:00
#SBATCH -A AST25015
#SBATCH -o slurm.%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jay.lim1@unlv.edu

# Stage 1 of the Schafer+24 run_1.6_320 box at 80/H
# (inputs/dust/streaming_strat_schafer24_L16_80.athinput): no self-gravity, uniform grid, to
# tlim (50 P; TLIM=<code time> to change) with a restart dump every 0.5 P.  Pick the restart
# for the self-gravity runs from the history afterwards: max rho_p is column 12 of s24.phst,
# to compare with the Hill density 9/gamma = 28.3 (gamma = 4 pi G rho0/Omega^2 = 0.318).
#
# The self-gravity restarts (uniform, SMR or AMR) must use the SAME number of ranks as this
# job (64): a particle restart reads one particle count per writing rank.
#
# NEEDS BRANCH si-strat-amr (mesh refinement with particles; dust-multigrid has none), built
# for the CPU -- the dust AMR code has never been compiled with CUDA:
#     git -C $HOME/athenak-multigrid fetch origin
#     git -C $HOME/athenak-multigrid worktree add $HOME/athenak-si-strat-amr si-strat-amr
#     cd $HOME/athenak-si-strat-amr && git submodule update --init kokkos
#     module load gcc openmpi fftw3          # BEFORE cmake: the default environment is nvhpc
#     cmake -B build-cluster -D Athena_ENABLE_MPI=ON -D Athena_ENABLE_FFT=ON \
#           -D CMAKE_CXX_COMPILER=/usr/bin/c++ -D Kokkos_ARCH_ARMV9_GRACE=ON
#     cmake --build build-cluster -j 16      # always give -j a number on the login node
#
# Submit FROM THE RUN DIRECTORY in $SCRATCH (outputs land there):
#     mkdir -p $SCRATCH/s24/L16_80 && cd $SCRATCH/s24/L16_80 \
#       && sbatch $HOME/athenak-si-strat-amr/scripts/cluster/si_strat_s24_stage1.slurm
# Cost: 128^2 x 16 cells, 262k particles, dt ~ 3.7e-3 (~1700 cycles per P); the Mac did
# ~3 min/P on 8 ranks, so 64 Grace cores should take well under a minute per P.

ATHENAK=${ATHENAK:-$HOME/athenak-si-strat-amr}
INPUT=${INPUT:-$ATHENAK/inputs/dust/streaming_strat_schafer24_L16_80.athinput}
BIN=${BIN:-$ATHENAK/build-cluster/src/athena}
if [ -n "${TLIM:-}" ]; then OVR=(time/tlim=$TLIM); else OVR=(); fi

module load gcc openmpi fftw3
cd $SLURM_SUBMIT_DIR
{
  echo "# job $SLURM_JOB_ID on $SLURM_JOB_NODELIST, $SLURM_NTASKS ranks (CPU)"
  echo "# started $(date); commit $(git -C $ATHENAK rev-parse --short HEAD); ${OVR[*]}"
} >> run.log
cp "$INPUT" input_used.athinput

ibrun $BIN -i input_used.athinput -d ./ "${OVR[@]}" >> run.log 2>&1
echo "# athena exit status $? at $(date)" >> run.log
