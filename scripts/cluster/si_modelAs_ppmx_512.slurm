#!/bin/bash
#SBATCH -J si_As_px512
#SBATCH -p gg                    # Vista Grace-Grace CPU nodes (2 x 72 cores, no GPU)
#SBATCH -N 2
#SBATCH -n 256                   # 256 MeshBlocks of 32^2 -> 1 per rank, 128 per node
#SBATCH -t 48:00:00              # gg queue maximum
#SBATCH -A AST25015
#SBATCH -o slurm.%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jay.lim1@unlv.edu

# YZ21 Model As at 512^2, 1 ppc, as si_modelAs_wenoz_512.slurm but with ppmx reconstruction
# instead of WENO-Z -- the one change: HLLE + IMEX2+, CFL 0.25, float64 dumps, 1000 orbits all
# as before.  The derived input is input_ppmx.athinput (the tracked input stays WENO-Z).
#
# COST: as the WENO-Z run (ppmx and WENO-Z measured equally fast on the linC benchmark).
#
# SUBMIT FROM ITS OWN RUN DIRECTORY in $SCRATCH (recon.txt and grid.txt guard against
# continuing another run's restarts):
#     mkdir -p $SCRATCH/si/modelAs_ppmx_512 && cd $SCRATCH/si/modelAs_ppmx_512
#     sbatch $HOME/athenak-multigrid/scripts/cluster/si_modelAs_ppmx_512.slurm

export NX=512 MB=32 RECON=ppmx
export SCRIPT=$HOME/athenak-multigrid/scripts/cluster/si_modelAs_ppmx_512.slurm   # chain resubmits this
exec bash $HOME/athenak-multigrid/scripts/cluster/si_modelAs_wenoz.slurm
