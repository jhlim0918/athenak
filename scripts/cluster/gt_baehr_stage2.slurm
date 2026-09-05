#!/bin/bash
#SBATCH -J gtb_BR_L_10
#SBATCH -p gg                    # Vista Grace-Grace CPU nodes
#SBATCH -N 16
#SBATCH -n 2048
#SBATCH -t 48:00:00
#SBATCH -A AST25015
#SBATCH -o slurm.%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jay.lim1@unlv.edu

# STAGE 2 of the Baehr+22 dust runs: restart from the saturated gas state of stage 1 and
# INSERT the particles (inputs/shearing_box/gravito_turb_baehr_dust.athinput).  Submit
# from the stage-2 run directory in $SCRATCH, one per model:
#     cd $SCRATCH/gtb/BR_L_10   && BR=true  sbatch $HOME/athenak-multigrid/scripts/cluster/gt_baehr_stage2.slurm
#     cd $SCRATCH/gtb/noBR_L_10 && BR=false sbatch $HOME/athenak-multigrid/scripts/cluster/gt_baehr_stage2.slurm
# STAGE1 is the stage-1 run directory; RST picks its t = 50 dump.  To continue a stage-2
# run after a walltime kill, restart from its OWN last dump with the particles it carries:
#     -r $(ls rst/*.rst | tail -1) particles/restart_insert=false   (and drop the -i line)

ATHENAK=$HOME/athenak-multigrid
STAGE1=${STAGE1:-$SCRATCH/gtb/stage1}
RST=${RST:-$(ls $STAGE1/rst/*.rst | tail -1)}
BR=${BR:-true}
ADDON=$ATHENAK/inputs/shearing_box/gravito_turb_baehr_dust.athinput

module load gcc openmpi fftw3

cd $SLURM_SUBMIT_DIR
{
  echo "# job $SLURM_JOB_ID on $SLURM_JOB_NUM_NODES node(s): $SLURM_JOB_NODELIST, $SLURM_NTASKS ranks"
  echo "# started $(date); commit $(git -C $ATHENAK rev-parse --short HEAD); restart $RST; back_reaction=$BR"
} >> run.log

ibrun $ATHENAK/build-cluster/src/athena -r $RST -i $ADDON -d ./ -t 47:45:00 \
      dust/back_reaction=$BR >> run.log 2>&1
