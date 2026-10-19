#!/bin/bash
#SBATCH -J si_growth_r1
#SBATCH -p astro                 # UNLV Physics, Zhu group partition (no time limit)
#SBATCH -w h-2                   # AMD EPYC 9825, 2 x 144 cores, 754 GB
#SBATCH -N 1
#SBATCH -n 64                    # 64 MeshBlocks of 32 x 4 x 32
#SBATCH --hint=nomultithread     # one rank per physical core
#SBATCH --mem=32G
#SBATCH -o slurm.%j.out

# R1: stratified streaming instability with live dust growth in a thin 3D box
# (inputs/dust/streaming_strat_growth_r1.athinput: 0.2 H x 4 cells x 0.2 H at 1280/H,
# FARGO, periodic z, Pi 0.05, Z 0.02, S0 1e-3, alpha_t 1e-4, v_frag 0.01 c_s).
# CASE selects the run (the overrides only replace parameters present in the input):
#   fid       growth + back-reaction
#   nobr      growth, no back-reaction (no SI)
#   nogrowth  back-reaction, no growth (S stays 1e-3)
# Submit FROM the run directory:
#   R=/data/limjay/runs/si_growth_r1
#   mkdir -p $R/fid && cd $R/fid && sbatch --export=ALL,CASE=fid $ATHENAK/scripts/cluster/si_growth_r1_h2.slurm
# If rst/*.rst exists the job continues from the newest one (TLIM extends the run).
# Optional: ATHENAK, BUILD (default build-mpi), TLIM (default 300 = 47.7 P).

set -u
ATHENAK=${ATHENAK:-/data/limjay/athenak_growth}
BUILD=${BUILD:-build-mpi}
CASE=${CASE:-fid}
TLIM=${TLIM:-300}
INPUT=$ATHENAK/inputs/dust/streaming_strat_growth_r1.athinput
case $CASE in
  fid)      OVR="" ;;
  nobr)     OVR="dust/back_reaction=false" ;;
  nogrowth) OVR="dust/growth=none dust/stopping_time_mode=species_fixed" ;;
  *) echo "unknown CASE=$CASE"; exit 2 ;;
esac

cd $SLURM_SUBMIT_DIR
cp $ATHENAK/$BUILD/src/athena ./athena_growth     # the binary this run used
echo "job $SLURM_JOB_ID host=$(hostname) ntasks=$SLURM_NTASKS case=$CASE tlim=$TLIM" >> cmd.txt
echo "source $ATHENAK (tar copy of worktree dust-growth, uncommitted growth code)" >> cmd.txt
echo "overrides: $OVR" >> cmd.txt
echo "start $(date)" >> status.log
last=$(ls -1 rst/*.rst 2>/dev/null | sort | tail -1)
if [ -n "$last" ]; then
  echo "restart from $last" >> status.log
  mpirun -np $SLURM_NTASKS --bind-to core ./athena_growth -r $last time/tlim=$TLIM \
      >> run.log 2>&1
  rc=$?
else
  cp $INPUT input_used.athinput
  mpirun -np $SLURM_NTASKS --bind-to core ./athena_growth -i input_used.athinput \
      time/tlim=$TLIM $OVR > run.log 2>&1
  rc=$?
fi
echo "end $(date) exit $rc" >> status.log
