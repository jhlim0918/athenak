#!/bin/bash
#SBATCH -J gtb_scal
#SBATCH -p gg                    # Vista Grace-Grace CPU nodes
#SBATCH -N 16                    # override on the command line: sbatch -N 4 -n 512 ...
#SBATCH -n 2048
#SBATCH -t 00:25:00
#SBATCH -A AST25015
#SBATCH -o slurm.%j.out
#
# A SHORT diagnostic of the stage-2 cost: the same restart and the same physics as
# gt_baehr_stage2.slurm, all file output suppressed, the multigrid's per-phase wall
# clock switched on, stopped after ~12 minutes of cycles.  Two of these answer the two
# questions a 34-hour production job cannot be interrupted to ask -- where the time
# goes, and whether the run still scales at the rank count being used:
#
#     mkdir -p $SCRATCH/gtb/scal_2048 && cd $SCRATCH/gtb/scal_2048 \
#       && sbatch          $HOME/athenak-multigrid/scripts/cluster/gt_baehr_scaling.slurm
#     mkdir -p $SCRATCH/gtb/scal_512  && cd $SCRATCH/gtb/scal_512 \
#       && sbatch -N 4 -n 512 $HOME/athenak-multigrid/scripts/cluster/gt_baehr_scaling.slurm
#
# Then compare the s/cycle of the two (4x the ranks should be 4x the rate; the cost of
# a 16^3 block with 3 ghost cells is 2.6x its interior, so it will not be) and read the
# mg_phase_times lines, which break a solve into slab / load / setup / cycles /
# retrieve.  Both jobs together are ~5 node-hours against the ~1100 of the campaign.
#
# Same environment variables as stage 2: STAGE1, RST, BR, ADDON, OVR.

ATHENAK=$HOME/athenak-multigrid
STAGE1=${STAGE1:-$SCRATCH/gtb/stage1}
RST=${RST:-$(ls $STAGE1/rst/*.rst | tail -1)}
BR=${BR:-true}
ADDON=${ADDON:-$ATHENAK/inputs/shearing_box/gravito_turb_baehr_dust.athinput}
OVR=${OVR:-}

module load gcc openmpi fftw3

cd $SLURM_SUBMIT_DIR
{
  echo "# job $SLURM_JOB_ID on $SLURM_JOB_NUM_NODES node(s): $SLURM_JOB_NODELIST, $SLURM_NTASKS ranks"
  echo "# started $(date); commit $(git -C $ATHENAK rev-parse --short HEAD); restart $RST; SCALING DIAGNOSTIC"
} >> run.log

# every output off (dt beyond tlim), the multigrid phase timer on
NOOUT="output1/dt=1.0e9 output2/dt=1.0e9 output3/dt=1.0e9 output4/dt=1.0e9"
NOOUT="$NOOUT output5/dt=1.0e9 output6/dt=1.0e9 output7/dt=1.0e9"

ibrun $ATHENAK/build-cluster/src/athena -r $RST -i $ADDON -d ./ -t 00:12:00 \
      dust/back_reaction=$BR gravity/show_defect=1 $NOOUT $OVR >> run.log 2>&1

{
  echo "# --- s/cycle from the elapsed= lines:"
  grep "cycle=" run.log | tail -20 \
    | awk '{for(i=1;i<=NF;i++){if($i~/^elapsed=/){split($i,a,"=");e=a[2]}
            if($i~/^cycle=/){split($i,c,"=");n=c[2]}}
            if(p!=""){printf "  cycles %s-%s: %.3f s/cycle\n",p,n,(e-q)/(n-p)} p=n;q=e}'
  echo "# --- the last multigrid phase breakdown:"
  grep "mg_phase_times" run.log | tail -3
} >> run.log
