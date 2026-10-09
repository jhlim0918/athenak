#!/bin/bash
# Submits the GPU scaling scan of the stratified SI box (si_strat_scaling_gpu.slurm) from
# the current directory, one job per point; every point writes into its own subdirectory
# N<N>_mb<MB>_g<GPUs>.  Run from a scan directory in $SCRATCH:
#
#   mkdir -p $SCRATCH/si3d_scaling && cd $SCRATCH/si3d_scaling
#   bash $HOME/athenak-multigrid/scripts/cluster/si_strat_scaling_submit.sh check  # 1 job first
#   bash $HOME/athenak-multigrid/scripts/cluster/si_strat_scaling_submit.sh        # the scan
#   bash $HOME/athenak-multigrid/scripts/cluster/si_strat_scaling_submit.sh big    # + 1024^3 on 64
#
# Matrix (GH200 = one GPU per node; cells per GPU in brackets):
#   strong, 256^3 in 64^3 blocks:   1, 2, 4, 8, 16 GPUs   (16.8M ... 1.0M)
#   strong, 512^3 in 128^3 blocks:  4, 8, 16, 32 GPUs     (33.6M ... 4.2M)
#   weak,   16.8M cells per GPU in 128^3 blocks: 256^3 on 1, 512^3 on 8 (from the strong
#           scan), and with "big" 1024^3 on 64 (1.07e9 particles)
# ~45 GPU-hours for the default scan, ~20 more for "big".
S=$HOME/athenak-multigrid/scripts/cluster/si_strat_scaling_gpu.slurm

sub() {   # sub N MB GPUs [extra sbatch args]
  local n=$1 mb=$2 g=$3; shift 3
  N=$n MB=$mb sbatch -J si3d_N${n}_g${g} -N $g -n $g "$@" $S
}

case "${1:-scan}" in
  check)
    CHECK=1 sbatch -J si3d_check -N 1 -n 1 -t 00:20:00 $S ;;
  scan)
    for g in 1 2 4 8 16; do sub 256 64 $g; done
    for g in 4 8 16 32;  do sub 512 128 $g; done
    sub 256 128 1 ;;
  big)
    sub 1024 128 64 -t 00:45:00 ;;
  *) echo "usage: $0 [check|scan|big]"; exit 1 ;;
esac
