#!/bin/bash
#SBATCH -J si_As_wz512
#SBATCH -p gg                    # Vista Grace-Grace CPU nodes (2 x 72 cores, no GPU)
#SBATCH -N 2
#SBATCH -n 256                   # 256 MeshBlocks of 32^2 -> 1 per rank, 128 per node
#SBATCH -t 48:00:00              # gg queue maximum
#SBATCH -A AST25015
#SBATCH -o slurm.%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jay.lim1@unlv.edu

# YZ21 Model As at 512^2 (0.2 H box, ~85 cells per wavelength of the seeded (6,6) mode),
# 1 ppc, WENO-Z + HLLE + IMEX2+, CFL 0.25, float64 dumps, to 1000 orbits -- everything as
# si_modelAs_wenoz.slurm (which this runs), with mesh 512^2 and 32^2 MeshBlocks.
#
# COST.  dt = 9.76e-5 at 512^2 and CFL 0.25, so 1000 P is 6.4e7 cycles on 4x the cells of
# the 256^2 run.  At 1 ppc with 32^2 MeshBlocks h-3 measured ~8e5 zone-cycles/s per rank,
# i.e. ~1.3 ms/cycle here on 256 ranks -> ~1 day: ONE job if Grace is comparable; the
# chain (at most 4 jobs) covers a slower machine.  Check run.log after the first hour.
# Output: dust dumps every 1 P (2 MB each in float64) and gas every 5 P (8 MB): ~4 GB.
#
# SUBMIT FROM ITS OWN RUN DIRECTORY in $SCRATCH (never the 256^2 one -- grid.txt guards it):
#     mkdir -p $SCRATCH/si/modelAs_wenoz_512 && cd $SCRATCH/si/modelAs_wenoz_512
#     sbatch $HOME/athenak-multigrid/scripts/cluster/si_modelAs_wenoz_512.slurm

export NX=512 MB=32
export SCRIPT=$HOME/athenak-multigrid/scripts/cluster/si_modelAs_wenoz_512.slurm   # chain resubmits this
exec bash $HOME/athenak-multigrid/scripts/cluster/si_modelAs_wenoz.slurm
