#!/bin/bash
#SBATCH -J gt16_stage1
#SBATCH -p gh                    # Vista Grace-Hopper nodes: one GH200 (H100 96 GB) each
#SBATCH -N 16                    # 25600 blocks / 16 = 1600 per rank, evenly balanced
#SBATCH -n 16                    #   (8 also divides evenly, but halving the per-rank cell
                                 #   count halves the wall time and leaves headroom for the
                                 #   dust fields stage 2 adds on the same geometry)
#SBATCH -t 24:00:00
#SBATCH -A AST25015
#SBATCH -o slurm.%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jay.lim1@unlv.edu

# STAGE 1 of the 16H proof-of-concept box (inputs/shearing_box/gravito_turb_bc16.athinput:
# 1280^2 x 512 cells, 80 per H_g) on GPUs.  ~1e8 cells per GPU on 8 nodes (about 40 GB of
# the 96 GB); 16 nodes halves that.  Submit from the run directory in $SCRATCH:
#     mkdir -p $SCRATCH/gt16/stage1 && cd $SCRATCH/gt16/stage1 && sbatch $HOME/athenak-multigrid/scripts/cluster/gt_bc16_stage1_gpu.slurm
# Continue (walltime kill, or to run past tlim): CONT=1 [TLIM=100] sbatch ... as in
# gt_baehr_stage1.slurm.  The binary is the GPU build, build-gpu (README section 4).
# UNTESTED as of 2026-09-15: the CUDA build of this tree (multigrid + kokkos-fft +
# particles) has not been compiled or timed yet -- do the one-node test first.

ATHENAK=$HOME/athenak-multigrid
# INPUT is overridable so the same script runs the pilot, which has the dust in from
# t = 0 and must be a COMPLETE input rather than an add-on (a second -i replaces the
# first on a fresh start; only -r restart -i addon merges).  Give it its own -J too:
#   mkdir -p $SCRATCH/gt16/pilot && cd $SCRATCH/gt16/pilot \
#     && INPUT=$HOME/athenak-multigrid/inputs/shearing_box/gravito_turb_bc16_pilot.athinput \
#        sbatch -J gt16_pilot -t 48:00:00 $HOME/athenak-multigrid/scripts/cluster/gt_bc16_stage1_gpu.slurm
INPUT=${INPUT:-$ATHENAK/inputs/shearing_box/gravito_turb_bc16.athinput}
BIN=${BIN:-$ATHENAK/build-gpu/src/athena}
RST=${RST:-}
if [ -n "${CONT:-}" ] && [ -z "$RST" ]; then RST=$(ls rst/*.rst | tail -1); fi
# NB a restart takes every parameter from the .rst and never reads $INPUT, so an input
# fix made after a run started does NOT reach its continuations.  That is how
# gt16_gpu_stage1b inherited bcool_cs2_floor = 0 from a dump of the floorless run: the
# disk had already overshot to Q = 0.6, burst, and lost 16 per cent of its mass through
# the z faces, and by t = 58 it was at Q = 3.75 and still climbing, with alpha_grav ~ 0.
# Continuing a run whose input has since changed needs the change repeated in XOVR, e.g.
#   XOVR="hydro_srcterms/bcool_cs2_floor=4.70361"
if [ -n "$RST" ]; then START=(-r "$RST"); else START=(-i "$INPUT"); fi
if [ -n "${TLIM:-}" ]; then OVR=(time/tlim=$TLIM); else OVR=(); fi
# extra overrides, e.g. the one-node timing test's smaller (4H x 4H) domain (README sec. 4)
OVR+=(${XOVR:-})

# the modules the GPU binary was built with (README section 4): the NVIDIA HPC SDK's
# CUDA-aware OpenMPI (AthenaK hands device buffers to MPI); the SDK carries CUDA
module load nvidia/24.7 openmpi/5.0.5_nvc249

# Vista's openmpi/5.0.5_nvc249 is built WITHOUT CUDA support
# (opal_built_with_cuda_support = false, read-only; the only accelerator component is
# "null"), so it treats every pointer as host memory.  AthenaK posts its sends straight
# from Kokkos views, so MPI memcpys GPU memory on the CPU and every rank takes SIGSEGV,
# "invalid permissions for mapped object", inside libmpi at the first exchange.  No
# environment variable reaches this: the binary must be built with
# Kokkos_ENABLE_IMPL_CUDA_UNIFIED_MEMORY=ON (README section 4), which allocates with
# cudaMallocManaged and advises the pages into HBM -- host-accessible, so the memcpy is
# legal, and a memory type UCX does support ("cuda-managed" in ucx_info -d).
# UCX still caches the memory type of an address range, and that cache has a history of
# answering wrongly for managed pointers, so it stays off.
export UCX_MEMTYPE_CACHE=n


cd $SLURM_SUBMIT_DIR
{
  echo "# job $SLURM_JOB_ID on $SLURM_JOB_NUM_NODES node(s): $SLURM_JOB_NODELIST, $SLURM_NTASKS ranks (GPU)"
  echo "# started $(date); commit $(git -C $ATHENAK rev-parse --short HEAD); start ${START[*]} ${OVR[*]}"
} >> run.log

# Stop the binary ~15 min before SLURM kills the job, at whatever -t the job actually got.
# This used to be hardcoded at 23:45:00, so `sbatch -t 48:00:00` still stopped at 23:45 and
# the run died mid-flight without its own restart dump.
tleft() {
  local end="${SLURM_JOB_END_TIME:-}"
  if [ -z "$end" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    end=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
          | tr ' ' '\n' | sed -n 's/^EndTime=//p' | head -1)
    [ -n "$end" ] && end=$(date -d "$end" +%s 2>/dev/null)
  fi
  [ -z "$end" ] && return 1
  local left=$(( end - $(date +%s) - 900 ))
  [ "$left" -le 0 ] && return 1
  printf '%d:%02d:%02d' $((left/3600)) $(((left%3600)/60)) $((left%60))
}
ATIME=$(tleft) || ATIME=23:45:00
echo "# athena wall-clock limit: $ATIME" >> run.log

ibrun $BIN "${START[@]}" -d ./ -t $ATIME "${OVR[@]}" >> run.log 2>&1
