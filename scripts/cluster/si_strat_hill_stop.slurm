#!/bin/bash
#SBATCH -J s24_stage1
#SBATCH -p gg                    # Vista Grace-Grace CPU nodes (2 x 72 cores, no GPU)
#SBATCH -N 1
#SBATCH -n 64                    # one rank per root MeshBlock (8 x 8 x 1 of 16^3)
#SBATCH -t 02:00:00
#SBATCH -A AST25015
#SBATCH -o slurm.%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jay.lim1@unlv.edu

# Stage 1 of the Schafer+24 run_1.6_320 box at 80/H
# (inputs/dust/streaming_strat_schafer24_L16_80.athinput): no self-gravity, uniform grid,
# STOPPED at the first 0.5 P restart dump written after max rho_p >= RHO_H (default 28.3 =
# 9/gamma, gamma = 4 pi G rho0/Omega^2 = 0.318 of Schafer+24).  run.log names that dump.
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
#       && sbatch $HOME/athenak-si-strat-amr/scripts/cluster/si_strat_hill_stop.slurm
# Cost: 128^2 x 16 cells, 262k particles, dt ~ 3.7e-3 (~1700 cycles per P); the Mac did
# ~3 min/P on 8 ranks, so 64 Grace cores should take well under a minute per P.

ATHENAK=${ATHENAK:-$HOME/athenak-si-strat-amr}
INPUT=${INPUT:-$ATHENAK/inputs/dust/streaming_strat_schafer24_L16_80.athinput}
BIN=${BIN:-$ATHENAK/build-cluster/src/athena}
RHO_H=${RHO_H:-28.3}
P=6.283185307179586

module load gcc openmpi fftw3
cd $SLURM_SUBMIT_DIR
{
  echo "# job $SLURM_JOB_ID on $SLURM_JOB_NODELIST, $SLURM_NTASKS ranks (CPU)"
  echo "# started $(date); commit $(git -C $ATHENAK rev-parse --short HEAD); stop at max rho_p >= $RHO_H"
} >> run.log
cp "$INPUT" input_used.athinput

ibrun $BIN -i input_used.athinput -d ./ >> athena.$SLURM_JOB_ID.log 2>&1 &
APID=$!

# 1. wait for the first phst row with dpm_max (column 12) >= RHO_H
until [ -f s24.phst ] && awk -v h=$RHO_H '!/^#/ && $12 >= h {f=1; exit} END {exit !f}' s24.phst; do
  kill -0 $APID 2>/dev/null || { echo "# athena ended before max rho_p reached $RHO_H ($(date))" >> run.log; exit 1; }
  sleep 10
done
TC=$(awk -v h=$RHO_H '!/^#/ && $12 >= h {print $1; exit}' s24.phst)
NC=$(awk -v t=$TC -v p=$P 'BEGIN {printf "%05d", int(t/(0.5*p)) + 1}')
echo "# max rho_p >= $RHO_H first at t = $(awk -v t=$TC -v p=$P 'BEGIN{printf "%.3f", t/p}') P; waiting for rst/s24.$NC.rst" >> run.log

# 2. wait for the next 0.5 P restart dump to be complete, then stop the run
until [ -f rst/s24.$NC.rst ] && [ $(( $(date +%s) - $(stat -c %Y rst/s24.$NC.rst) )) -ge 20 ]; do
  kill -0 $APID 2>/dev/null || break
  sleep 5
done
kill $APID 2>/dev/null; sleep 10; pkill -u $USER -f "$BIN -i input_used.athinput" 2>/dev/null
{
  echo "# STOPPED after rst/s24.$NC.rst (t = $(awk -v n=$NC 'BEGIN{printf "%.1f", (n+0)*0.5}') P) at $(date)"
  awk -v p=$P '!/^#/ {r = sprintf("# last phst row: t = %.3f P, max rho_p = %.2f", $1/p, $12)} END {print r}' s24.phst
  echo "# self-gravity restarts: ibrun with $SLURM_NTASKS ranks, -r rst/s24.$NC.rst -i <add-on>"
} >> run.log
