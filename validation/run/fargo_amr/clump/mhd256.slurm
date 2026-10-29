#!/bin/bash
#SBATCH -J swclump_mhd256
#SBATCH -p astro
#SBATCH -w h-gpu
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o /data/limjay/runs/swing_clump_fargoamr/mhd256.%j.out
# Uniform 256^2 x 32 MHD twin of the B0 = 0.5 collapse runs (no refinement), to t = 12
export PATH=/usr/local/cuda/bin:$PATH
ATH=/data/limjay/athenak_fargoamr_gpu/build-cuda/src/athena
O=/data/limjay/runs/swing_clump_fargoamr
n=M6_uni256_b0.5; mkdir -p $O/$n; t0=$(date +%s); echo "start $(date)"
$ATH -i $O/mhd_clump.athinput -d $O/$n mesh_refinement/refinement=none mesh/nx1=256 mesh/nx2=256 mesh/nx3=32 time/tlim=12.0 > $O/$n/log 2>&1
echo "$n exit=$? $(( $(date +%s)-t0 ))s fatal=$(grep -c FATAL $O/$n/log) $(grep cycle= $O/$n/log | tail -1)"
echo "end $(date)"
