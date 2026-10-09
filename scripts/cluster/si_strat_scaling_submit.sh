#!/bin/bash
# Submits the GPU scaling scan of the stratified SI box (si_strat_scaling_gpu.slurm) from
# the current directory, one job per point; every point writes into its own subdirectory
# N<N>_mb<MB>_g<GPUs>.  Run from a scan directory in $SCRATCH:
#
#   mkdir -p $SCRATCH/si3d_scaling && cd $SCRATCH/si3d_scaling
#   bash $HOME/athenak-multigrid/scripts/cluster/si_strat_scaling_submit.sh check  # 1 job first
#   bash $HOME/athenak-multigrid/scripts/cluster/si_strat_scaling_submit.sh        # the scan
#   bash $HOME/athenak-multigrid/scripts/cluster/si_strat_scaling_submit.sh big    # + 1024^3 on 64
#   bash $HOME/athenak-multigrid/scripts/cluster/si_strat_scaling_submit.sh proposal
#
# Matrix (GH200 = one GPU per node; cells per GPU in brackets):
#   strong, 256^3 in 64^3 blocks:   1, 2, 4, 8, 16 GPUs   (16.8M ... 1.0M)
#   strong, 512^3 in 128^3 blocks:  4, 8, 16, 32 GPUs     (33.6M ... 4.2M)
#   weak,   16.8M cells per GPU in 128^3 blocks: 256^3 on 1, 512^3 on 8 (from the strong
#           scan), and with "big" 1024^3 on 64 (1.07e9 particles)
#   proposal, the NHFP run (1.6H)^2 x 0.2H at 2560/H = 4096^2 x 512 shrunk in x and y at the
#           same resolution and per-GPU load (16.8M cells, 8 blocks of 128^3 per GPU, 4 block
#           layers in z): (0.4H)^2 on 32 GPUs and (0.8H)^2 on 128; with 512^3 = (0.2H)^2 on 8
#           from the scan they form a weak-scaling series 8 -> 32 -> 128 GPUs, leaving a 4x
#           step to the 512 GPUs of the full box.  Check the gh node limit (qlimits) first.
# ~45 GPU-hours for the default scan, ~20 more for "big", ~40 for "proposal".
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
  proposal)
    NXY=1024 NZ=512 MB=128 sbatch -J si3d_prop_g32 -N 32 -n 32 $S
    # 2048^2 x 512 = 2^31 cells: at 1 particle per cell the count would overflow AthenaK's
    # 32-bit particle total and tags (max 2^31 - 1), hence 0.99 per cell for this point
    NXY=2048 NZ=512 MB=128 OVR="particles/ppc=0.99" sbatch -J si3d_prop_g128 -N 128 -n 128 $S ;;
  *) echo "usage: $0 [check|scan|big|proposal]"; exit 1 ;;
esac
