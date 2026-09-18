#!/bin/bash
#SBATCH -J gt16_dust
#SBATCH -p gh
#SBATCH -N 8
#SBATCH -n 8
#SBATCH -t 24:00:00
#SBATCH -A AST25015
#SBATCH -o slurm.%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jay.lim1@unlv.edu

# STAGE 2 of the 16H box: restart from a stage-1 dump and insert the St = 0.3 layer with
# the radial pressure gradient on (inputs/shearing_box/gravito_turb_bc16_dust.athinput).
# tlim in the add-on is ABSOLUTE: pass the target time.  Submit from the run directory:
#     mkdir -p $SCRATCH/gt16/dust && cd $SCRATCH/gt16/dust && \
#       RST=$SCRATCH/gt16/stage1/rst/gt16.00005.rst TLIM=100 sbatch $HOME/athenak-multigrid/scripts/cluster/gt_bc16_stage2_gpu.slurm
# RST defaults to the stage-1 directory's last dump; pick the dump after the first burst
# from the stage-1 history (Q at the bottom of a cooling phase).  Extra overrides in OVR.
# To continue this run from its own dump: RST=<own dump> and OVR="particles/restart_insert=false"
# (and ADDON= empty).  UNTESTED on GPUs as of 2026-09-15 (see gt_bc16_stage1_gpu.slurm).

ATHENAK=$HOME/athenak-multigrid
STAGE1=${STAGE1:-$SCRATCH/gt16/stage1}
RST=${RST:-$(ls $STAGE1/rst/*.rst | tail -1)}
ADDON=${ADDON-$ATHENAK/inputs/shearing_box/gravito_turb_bc16_dust.athinput}
TLIM=${TLIM:-100}
OVR=${OVR:-}
BIN=${BIN:-$ATHENAK/build-gpu/src/athena}
if [ -n "$ADDON" ]; then ADD=(-i "$ADDON"); else ADD=(); fi

module load nvidia/24.7 openmpi/5.0.5_nvc249

# see gt_bc16_stage1_gpu.slurm: this MPI is not CUDA-aware, so the binary must be the
# unified-memory build; UCX must not cache the memory type of a managed pointer
export UCX_MEMTYPE_CACHE=n


cd $SLURM_SUBMIT_DIR
{
  echo "# job $SLURM_JOB_ID on $SLURM_JOB_NUM_NODES node(s): $SLURM_JOB_NODELIST, $SLURM_NTASKS ranks (GPU)"
  echo "# started $(date); commit $(git -C $ATHENAK rev-parse --short HEAD); restart $RST; addon $ADDON; tlim $TLIM $OVR"
} >> run.log

ibrun $BIN -r $RST "${ADD[@]}" -d ./ -t 23:45:00 time/tlim=$TLIM $OVR >> run.log 2>&1
