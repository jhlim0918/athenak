#!/bin/bash
#SBATCH -J swclump_mhd
#SBATCH -p astro
#SBATCH -w h-gpu
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 06:00:00
#SBATCH -o /data/limjay/runs/swing_clump_fargoamr/mhd.%j.out
# Nonlinear swing / clump collapse with a field (isothermal HLLE, B along x2)
# Jeans criterion lambda_J >= 16 cells: uniform / ring / patch at b0 = 0.5, uniform / patch at 0.25
export PATH=/usr/local/cuda/bin:$PATH
ATH=/data/limjay/athenak_fargoamr_gpu/build-cuda/src/athena
O=/data/limjay/runs/swing_clump_fargoamr
run() { local n=$1; shift; mkdir -p $O/$n; local t0=$(date +%s)
  $ATH -i $O/mhd_clump.athinput -d $O/$n "$@" > $O/$n/log 2>&1
  echo "$n exit=$? $(( $(date +%s)-t0 ))s fatal=$(grep -c FATAL $O/$n/log) $(grep 'cycle=' $O/$n/log | tail -1) $(grep -i created $O/$n/log | tail -1)"; }
echo "start $(date)"
run M1_uni_b0.5    mesh_refinement/refinement=none
run M2_ring_b0.5   shearing_box/oa_annular=true
run M3_patch_b0.5
run M4_uni_b0.25   mesh_refinement/refinement=none problem/b0=0.25
run M5_patch_b0.25 problem/b0=0.25
echo "end $(date)"
