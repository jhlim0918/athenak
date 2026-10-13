#!/bin/bash
#SBATCH -J si3d_prof
#SBATCH -p gh                    # Vista Grace-Hopper nodes: one GH200 (H100 96 GB) each
#SBATCH -N 2                     # submit with -N 1 -n 1 for the one-GPU reference
#SBATCH -n 2
#SBATCH -t 00:30:00
#SBATCH -A AST25015
#SBATCH -o slurm.%j.out

# Nsight Systems profile of the stratified SI benchmark at the point where the scaling scan
# (si_strat_scaling_gpu.slurm, validation/run/scaling/si_3d) first loses efficiency:
# 256^3 in 64^3 MeshBlocks, 0.131 s/cycle on 1 GPU but 0.114 s/cycle on 2 (efficiency
# 0.57), and per-GPU throughput 4.5x lower on any multi-GPU run.  Suspect: Kokkos unified
# (managed) memory + Vista's CUDA-unaware OpenMPI, i.e. MPI touching managed pages on the
# host and page faults/migrations between host and HBM around every exchange.  The
# profile records CUDA kernels and memcpys, CPU- and GPU-side unified-memory page faults,
# MPI calls and OS runtime, one report per rank.  Submit both, from a directory in $SCRATCH:
#
#   mkdir -p $SCRATCH/si3d_prof && cd $SCRATCH/si3d_prof
#   sbatch -N 1 -n 1 $HOME/athenak-multigrid/scripts/cluster/si_strat_profile_gpu.slurm
#   sbatch -N 2 -n 2 $HOME/athenak-multigrid/scripts/cluster/si_strat_profile_gpu.slurm
#
# Each job writes prof_g<GPUs>/: run.log (environment, MPI build info, timing), the reports
# prof_r<rank>.nsys-rep, and stats_r<rank>.txt (nsys stats summaries, small: bring these
# back first; the .nsys-rep files open in the Nsight Systems GUI).  NLIM = 200 cycles; the
# tracing slows the run, so compare the 1- and 2-GPU reports with each other, not with the
# scan's timings.

ATHENAK=$HOME/athenak-multigrid
INPUT=${INPUT:-$ATHENAK/inputs/dust/streaming_strat_godines10au.athinput}
BIN=${BIN:-$ATHENAK/build-gpu/src/athena}
N=${N:-256}
MB=${MB:-64}
NLIM=${NLIM:-200}

module load nvidia/24.7 openmpi/5.0.5_nvc249
export UCX_MEMTYPE_CACHE=n       # as in the production scripts
command -v nsys > /dev/null || module load nsight-systems 2> /dev/null

cd $SLURM_SUBMIT_DIR
D=prof_g$SLURM_NTASKS
mkdir -p $D && cd $D
awk '/^<output[3-9]>/ {skip = 1; next} /^</ {skip = 0} !skip' $INPUT > bench.athinput
{
  echo "# job $SLURM_JOB_ID on $SLURM_JOB_NUM_NODES node(s): $SLURM_JOB_NODELIST, $SLURM_NTASKS ranks"
  echo "# started $(date); commit $(git -C $ATHENAK rev-parse --short HEAD); ${N}^3, MB=$MB, NLIM=$NLIM"
  echo "# nsys: $(command -v nsys) $(nsys --version 2>&1 | head -1)"
  echo "# MPI CUDA support (ompi_info):"
  ompi_info --parsable --all 2>/dev/null | grep -i -E "cuda_support|accelerator" | head -5
  echo "# UCX memory types:"
  ucx_info -d 2>/dev/null | grep -i "memory types" | sort -u | head -5
  echo "# MPI modules installed (look for a CUDA-aware build):"
  module -t spider openmpi 2>&1 | head -20
  module -t spider mvapich 2>&1 | head -10
  module -t spider hpcx 2>&1 | head -10
} >> run.log

# one report per rank; %q{} expands the rank from OpenMPI's environment
ibrun nsys profile -o prof_r%q{OMPI_COMM_WORLD_RANK} --force-overwrite=true \
  -t cuda,mpi,osrt --mpi-impl=openmpi \
  --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true \
  $BIN -i bench.athinput -d ./ -t 00:20:00 \
  mesh/nx1=$N mesh/nx2=$N mesh/nx3=$N \
  meshblock/nx1=$MB meshblock/nx2=$MB meshblock/nx3=$MB \
  time/nlim=$NLIM time/ndiag=10 > athena.$SLURM_JOB_ID.log 2>&1
echo "# athena exit status $? at $(date)" >> run.log
grep -E "^elapsed=.*cycle=(100|200) " athena.$SLURM_JOB_ID.log >> run.log

# text summaries: top kernels, CUDA API (incl. synchronisation), memcpys, unified-memory
# traffic and page faults, MPI time; a report a given nsys does not know is skipped
for r in prof_r*.nsys-rep; do
  k=${r#prof_r}; k=${k%.nsys-rep}
  for rep in cuda_gpu_kern_sum cuda_api_sum cuda_gpu_mem_time_sum cuda_gpu_mem_size_sum \
             um_sum um_total_sum um_cpu_page_faults_sum mpi_event_sum osrt_sum; do
    echo "===== $rep"
    nsys stats -q -r $rep -f column $r 2>&1 | head -40
  done > stats_r$k.txt
done
ls -la >> run.log
