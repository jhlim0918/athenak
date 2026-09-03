#!/bin/bash
#SBATCH -J gt_sc14_b10
#SBATCH -p gg                    # Vista Grace-Grace CPU nodes (2 x 72 cores, no GPU)
#SBATCH -N 2                     # nodes
#SBATCH -n 256                   # total MPI ranks: 768 MeshBlocks / 256 = 3 per rank,
                                 #   the same decomposition as the NAS run.  128 of the
                                 #   144 cores per node: the code is memory-bandwidth-
                                 #   bound, so the idle cores cost nothing (README sec. 2)
#SBATCH -t 48:00:00              # gg queue maximum
#SBATCH -A AST25015              # allocation to charge
#SBATCH -o slurm.%j.out          # stdout+stderr (no -e => merged, like PBS -j oe)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jay.lim1@unlv.edu

# Slurm/Vista twin of gt_sc14_full.pbs (NAS).  Submit from the RUN directory in
# $SCRATCH -- outputs land there and Slurm starts the job in the submit directory:
#     cd $SCRATCH/gt_sc14_stndrd/b10 && sbatch $HOME/athenak-multigrid/scripts/cluster/gt_sc14_full.slurm
# To continue a run after a walltime kill, replace the -i line with:
#     -r $(ls rst/*.rst | tail -1) \
# Per-run parameters go on the ibrun line, e.g.  hydro_srcterms/bcool_beta=5
# Hi-res twin: INPUT=.../gravito_turb_sc14_hi.athinput with -N 6 -n 768 (README sec. 2).

ATHENAK=$HOME/athenak-multigrid
INPUT=$ATHENAK/inputs/shearing_box/gravito_turb_sc14.athinput

# the modules the binary was built with -- mirror `module list` from the cmake shell
module load gcc openmpi fftw3

cd $SLURM_SUBMIT_DIR
{
  echo "# job $SLURM_JOB_ID on $SLURM_JOB_NUM_NODES node(s): $SLURM_JOB_NODELIST, $SLURM_NTASKS ranks"
  echo "# started $(date); commit $(git -C $ATHENAK rev-parse --short HEAD)"
} >> run.log

# ibrun is TACC's launcher: rank count and placement come from -N/-n above.
# athena's -t guard stops 15 min before the Slurm limit so the final rst is written.
ibrun $ATHENAK/build-cluster/src/athena -i $INPUT -d ./ -t 47:45:00 >> run.log 2>&1
