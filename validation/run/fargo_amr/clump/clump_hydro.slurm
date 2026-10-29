#!/bin/bash
#SBATCH -J swclump_hydro
#SBATCH -p astro
#SBATCH -w h-gpu
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 06:00:00
#SBATCH -o /data/limjay/runs/swing_clump_fargoamr/hydro.%j.out
# Nonlinear swing / clump collapse (4 pi G = 6, amp 0.2), hydro, Jeans criterion
# lambda_J >= 16 cells (num_levels 4): uniform, ring (oa_annular), patch, 256^2, marginal
export PATH=/usr/local/cuda/bin:$PATH
ATH=/data/limjay/athenak_fargoamr_gpu/build-cuda/src/athena
O=/data/limjay/runs/swing_clump_fargoamr
run() { local n=$1; shift; mkdir -p $O/$n; local t0=$(date +%s)
  $ATH -i $O/hydro_clump.athinput -d $O/$n "$@" > $O/$n/log 2>&1
  echo "$n exit=$? $(( $(date +%s)-t0 ))s fatal=$(grep -c FATAL $O/$n/log) $(grep 'cycle=' $O/$n/log | tail -1) $(grep -i created $O/$n/log | tail -1)"; }
echo "start $(date)"
run H1_uniform  mesh_refinement/refinement=none
run H2_ring     shearing_box/oa_annular=true
run H3_patch
run H4_uni256   mesh_refinement/refinement=none mesh/nx1=256 mesh/nx2=256 mesh/nx3=32
run H5_marginal gravity/four_pi_G=5.0 problem/amp=0.05
echo "end $(date)"
