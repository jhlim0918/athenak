#!/bin/bash
#SBATCH -J si3d_scale
#SBATCH -p gh                    # Vista Grace-Hopper nodes: one GH200 (H100 96 GB) each
#SBATCH -N 1                     # one rank per node = one GPU per rank; the submit helper
#SBATCH -n 1                     #   overrides -N/-n for every point of the scan
#SBATCH -t 00:30:00
#SBATCH -A AST25015
#SBATCH -o slurm.%j.out

# GPU scaling test of the stratified streaming-instability box (streaming_strat pgen,
# inputs/dust/streaming_strat_godines10au.athinput: the Godines+26 10 au model, (0.2H)^3,
# four species, 1 particle per cell, Gaussian layer, Roe/WENO-Z/imex2+, no self-gravity)
# at N^3 cells with N^3/MB^3 MeshBlocks -- or, for the proposal geometry, NXY x NXY x NZ
# cells in an Lxy x Lxy x 0.2 H box with Lxy = NXY*(0.2 H/NZ) (cubic cells, the resolution
# set by NZ; NXY = 4096, NZ = 512 is (1.6H)^2 x 0.2H at 2560/H).  One job = one point; run
# scripts/cluster/si_strat_scaling_submit.sh from the scan directory to submit the matrix,
# and si_strat_scaling_collect.sh to tabulate it.
#
# The job runs a copy of the input with every mesh/particle/restart output removed (only
# hst and phst stay), for NLIM cycles from the t = 0 Gaussian layer, and appends to
# run.log a "# bench" line: s/cycle and cell-updates/s (per GPU) over the steady window
# from cycle STEADY0 on (start-up and the managed-memory page migration of the first
# cycles excluded), the peak GPU memory, and the projected wall time and GPU-hours to
# t = 100 P at the measured dt.  mem.log has host/GPU memory every 5 s per node.
#
#   N=256 MB=64 sbatch -N 4 -n 4 $HOME/athenak-multigrid/scripts/cluster/si_strat_scaling_gpu.slurm
#
# CHECK=1 instead runs the 64^3 Mac run (active_projects/si_fillfactor_resolution/runs/
# g10au_N64, 8 MeshBlocks of 32^3) to t = 0.1 P on one GPU and keeps its phst/hst for a
# CPU-vs-GPU comparison (the initial state is decomposition independent, so both start
# identical; differences should stay at round-off for these few hundred cycles).
#
# The binary is the GPU build (README section 4: unified memory is required on Vista).

ATHENAK=$HOME/athenak-multigrid
INPUT=${INPUT:-$ATHENAK/inputs/dust/streaming_strat_godines10au.athinput}
BIN=${BIN:-$ATHENAK/build-gpu/src/athena}
N=${N:-256}
NXY=${NXY:-$N}
NZ=${NZ:-$N}
MB=${MB:-64}
NLIM=${NLIM:-400}
STEADY0=${STEADY0:-100}
OVR=(${OVR:-})
if [ -n "${CHECK:-}" ]; then
  NXY=64; NZ=64; MB=32; NLIM=-1; OVR+=(time/tlim=0.6283185307179586)
fi
HXY=$(awk -v a=$NXY -v b=$NZ 'BEGIN {printf "%.17g", 0.1*a/b}')   # half width, cubic cells
if [ "$NXY" = "$NZ" ]; then NTAG=N$NZ; else NTAG=N${NXY}x${NXY}x${NZ}; fi

module load nvidia/24.7 openmpi/5.0.5_nvc249
export UCX_MEMTYPE_CACHE=n       # see gt_bc16_stage1_gpu.slurm

cd $SLURM_SUBMIT_DIR
TAG=${CHECK:+check_}${NTAG}_mb${MB}_g${SLURM_NTASKS}
mkdir -p $TAG && cd $TAG
# bench input: the production input minus output blocks 3 and up (dust_dpm, pvtk,
# hydro_w, rst), so no dump enters the timing; hst and phst (outputs 1-2) stay
awk '/^<output[3-9]>/ {skip = 1; next} /^</ {skip = 0} !skip' $INPUT > bench.athinput
{
  echo "# job $SLURM_JOB_ID on $SLURM_JOB_NUM_NODES node(s): $SLURM_JOB_NODELIST, $SLURM_NTASKS ranks, GPU (GH200)"
  echo "# started $(date); commit $(git -C $ATHENAK rev-parse --short HEAD); ${NXY}x${NXY}x${NZ} cells, box ${HXY}x2 H wide, MB=$MB NLIM=$NLIM ${OVR[*]}"
} >> run.log

# memory monitor, one per node (as in gtb_dust_bench.slurm)
MON='while :; do
  printf "%s %s %s %s %s\n" "$(hostname -s)" "$(date +%s)" \
    "$(free -m | awk "/^Mem:/{print \$3, \$7}")" \
    "$(ps -C athena -o rss= | awk "{s+=\$1} END {print int(s/1024)}")" \
    "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr "\n" " ")"
  sleep 5; done'
echo "# host epoch host_used_MB host_avail_MB athena_rss_MB gpu_used_MB" > mem.log
MONPIDS=()
for h in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
  ssh -o StrictHostKeyChecking=no "$h" "$MON" >> mem.log 2>/dev/null &
  MONPIDS+=($!)
done

LOG=athena.$SLURM_JOB_ID.log
ibrun $BIN -i bench.athinput -d ./ -t 00:20:00 \
  mesh/nx1=$NXY mesh/nx2=$NXY mesh/nx3=$NZ \
  mesh/x1min=-$HXY mesh/x1max=$HXY mesh/x2min=-$HXY mesh/x2max=$HXY \
  meshblock/nx1=$MB meshblock/nx2=$MB meshblock/nx3=$MB \
  time/nlim=$NLIM time/ndiag=10 "${OVR[@]}" > $LOG 2>&1
STATUS=$?
cat $LOG >> run.log
echo "# athena exit status $STATUS at $(date)" >> run.log
kill "${MONPIDS[@]}" 2>/dev/null
sleep 2

# steady window: first elapsed= line >= STEADY0 cycles, to the last one; GPU memory peak
GPUMAX=$(awk 'NR > 1 && NF >= 6 {for (i = 6; i <= NF; i++) if ($i > m) m = $i} END {print m + 0}' mem.log)
awk -v nxy=$NXY -v nz=$NZ -v mb=$MB -v s0=$STEADY0 -v ng=$SLURM_NTASKS -v gmax=$GPUMAX '
  /^elapsed=/ { split($1, a, "="); split($2, b, "="); split($4, d, "=")
                t = a[2] + 0; c = b[2] + 0; dt = d[2] + 0
                if (c >= s0 && t1 == "") { t1 = t; k1 = c }
                t2 = t; k2 = c }
  END { if (t1 == "" || k2 <= k1) { print "# bench: too few cycles for a steady window"; exit }
        spc = (t2 - t1)/(k2 - k1); ncell = nxy*nxy*nz
        cyc = 628.3185307179586/dt; hrs = cyc*spc/3600
        printf "# bench N=%dx%dx%d MB=%d GPUs=%d: cycles %d-%d %.4f s/cycle, %.3e cell-updates/s (%.3e per GPU), dt %.3e, 100 P = %.2e cycles = %.1f h wall = %.0f GPU-h, peak GPU mem %d MB\n",
               nxy, nxy, nz, mb, ng, k1, k2, spc, ncell/spc, ncell/spc/ng, dt, cyc, hrs, hrs*ng, gmax }' $LOG >> run.log
tail -1 run.log
sacct -j "$SLURM_JOB_ID" --format=JobID,State,ExitCode,MaxRSS,Elapsed >> run.log 2>&1
