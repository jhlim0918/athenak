#!/bin/bash
#SBATCH -J gt16_stage1
#SBATCH -p gh                    # Vista Grace-Hopper nodes: one GH200 (H100 96 GB) each
#SBATCH -N 8
#SBATCH -n 8                     # one rank per GPU
#SBATCH -t 24:00:00
#SBATCH -A AST25015
#SBATCH -o slurm.%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jay.lim1@unlv.edu

# STAGE 1 of the 16H proof-of-concept box (inputs/shearing_box/gravito_turb_bc16.athinput:
# 1280^2 x 512 cells, 80 per H_g) on GPUs.  ~1e8 cells per GPU on 8 nodes (about 40 GB of
# the 96 GB); 16 nodes halves the wall time.  Submit from the run directory in $SCRATCH:
#     mkdir -p $SCRATCH/gt16/stage1 && cd $SCRATCH/gt16/stage1 && sbatch $HOME/athenak-multigrid/scripts/cluster/gt_bc16_stage1_gpu.slurm
# Continue (walltime kill, or to run past tlim): CONT=1 [TLIM=100] sbatch ... as in
# gt_baehr_stage1.slurm.  The binary is the GPU build, build-gpu (README section 4).
# UNTESTED as of 2026-09-15: the CUDA build of this tree (multigrid + kokkos-fft +
# particles) has not been compiled or timed yet -- do the one-node test first.

ATHENAK=$HOME/athenak-multigrid
INPUT=$ATHENAK/inputs/shearing_box/gravito_turb_bc16.athinput
BIN=${BIN:-$ATHENAK/build-gpu/src/athena}
RST=${RST:-}
if [ -n "${CONT:-}" ] && [ -z "$RST" ]; then RST=$(ls rst/*.rst | tail -1); fi
if [ -n "$RST" ]; then START=(-r "$RST"); else START=(-i "$INPUT"); fi
if [ -n "${TLIM:-}" ]; then OVR=(time/tlim=$TLIM); else OVR=(); fi
# extra overrides, e.g. the one-node timing test's smaller (4H x 4H) domain (README sec. 4)
OVR+=(${XOVR:-})

# the modules the GPU binary was built with (README section 4): the NVIDIA HPC SDK's
# CUDA-aware OpenMPI (AthenaK hands device buffers to MPI); the SDK carries CUDA
module load nvidia/24.7 openmpi/5.0.5_nvc249

cd $SLURM_SUBMIT_DIR
{
  echo "# job $SLURM_JOB_ID on $SLURM_JOB_NUM_NODES node(s): $SLURM_JOB_NODELIST, $SLURM_NTASKS ranks (GPU)"
  echo "# started $(date); commit $(git -C $ATHENAK rev-parse --short HEAD); start ${START[*]} ${OVR[*]}"
} >> run.log

ibrun $BIN "${START[@]}" -d ./ -t 23:45:00 "${OVR[@]}" >> run.log 2>&1
