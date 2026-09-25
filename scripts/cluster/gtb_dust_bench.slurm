#!/bin/bash
#SBATCH -J gtb_dust_bench
#SBATCH -p gh                    # GPU: Grace-Hopper, one GH200 per node.  CPU twin:
#SBATCH -N 1                     #   sbatch -p gg -n 128 ... (Grace-Grace, 128 ranks =
#SBATCH -n 1                     #   the per-node layout of the production CPU dust runs)
#SBATCH -t 01:00:00
#SBATCH -A AST25015
#SBATCH -o slurm.%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jay.lim1@unlv.edu

# The dust stage on ONE node, GPU or CPU, for two purposes:
#
# 1. The GPU memory failure.  Both GPU dust jobs so far inserted their particles and
#    were then SIGKILLed (exit 137 -- the Linux OOM killer, i.e. HOST memory; an HBM
#    allocation failure would be a Kokkos abort with a message): gt16_gpu_s2test
#    (8 ranks, restart + insert, killed before cycle 0 finished) and gt16_gpu_pilot
#    (8 ranks, dust from t = 0, killed after cycle 0).  With the unified-memory build
#    (README section 4) managed pages that do not fit in HBM, or are first touched by
#    the CPU, live in host memory, so which memory ran out is open.  gtb's 16^3
#    MeshBlocks carry 2.6x their cells in ghost-padded arrays, so one GPU here holds
#    1.74e8 padded cells -- the same as one rank of the 16H box on 8 (1.76e8).  mem.log
#    records host used/available, athena RSS and GPU memory every 5 s.  If one GPU
#    survives, -N 2 -n 2 adds the inter-rank particle exchange, never yet run on GPUs.
#
# 2. The GPU-vs-CPU throughput, same problem on both: restart the 20/H Baehr gas at
#    t = 150 (gtb stage 1, 512^2 x 256, 16384 MeshBlocks) and insert the three-species
#    layer of gtb_BR_3sp (gravito_turb_baehr_dust3.athinput, 4.5e6 particles), run to
#    t = 150.5 (~350 cycles).  run.log ends with AthenaK's own zone-cycles/cpu_second
#    and particle-updates/cpu_second (whole loop, first outputs included) and a
#    "# bench" line: the steady rate from the elapsed= lines after cycle STEADY0 past the
#    restart, with the start-up and the t = 150 dumps excluded.  ADDON= (empty) gives the
#    gas-only rate of the same box.
#
#   GPU:  mkdir -p $SCRATCH/gtb/bench_gh && cd $SCRATCH/gtb/bench_gh \
#           && sbatch $HOME/athenak-multigrid/scripts/cluster/gtb_dust_bench.slurm
#   CPU:  mkdir -p $SCRATCH/gtb/bench_gg && cd $SCRATCH/gtb/bench_gg \
#           && sbatch -p gg -n 128 -t 02:00:00 $HOME/athenak-multigrid/scripts/cluster/gtb_dust_bench.slurm
#   gas only: ADDON= sbatch ...      two GPUs: sbatch -N 2 -n 2 ...
#
# RST is the t = 150 dump the CPU dust tests (gtb_test_fix512*) restarted from; TLIM is
# ABSOLUTE.  The partition picks the modules and the binary.

ATHENAK=$HOME/athenak-multigrid
RST=${RST:-$SCRATCH/gtb/stage1/rst/gtb.00030.rst}
ADDON=${ADDON-$ATHENAK/inputs/shearing_box/gravito_turb_baehr_dust3.athinput}
TLIM=${TLIM:-150.5}
OVR=${OVR:-}
NCELL=${NCELL:-67108864}         # 512 x 512 x 256 active cells
STEADY0=${STEADY0:-100}          # cycles past the restart before the steady window opens
if [ -n "$ADDON" ]; then ADD=(-i "$ADDON"); else ADD=(); fi

if [ "${SLURM_JOB_PARTITION:-gh}" = "gg" ]; then
  module load gcc openmpi fftw3
  BIN=${BIN:-$ATHENAK/build-cluster/src/athena}
  ARCH="CPU (Grace-Grace)"
else
  module load nvidia/24.7 openmpi/5.0.5_nvc249
  export UCX_MEMTYPE_CACHE=n     # see gt_bc16_stage1_gpu.slurm
  BIN=${BIN:-$ATHENAK/build-gpu/src/athena}
  ARCH="GPU (GH200)"
fi

cd $SLURM_SUBMIT_DIR
{
  echo "# job $SLURM_JOB_ID on $SLURM_JOB_NUM_NODES node(s): $SLURM_JOB_NODELIST, $SLURM_NTASKS ranks, $ARCH"
  echo "# started $(date); commit $(git -C $ATHENAK rev-parse --short HEAD); restart $RST; addon ${ADDON:-none}; tlim $TLIM $OVR"
} >> run.log

# memory monitor, one per node: host used / available (MB), summed athena RSS (MB),
# GPU memory used (MB; empty on gg).  ssh reaches the job's own nodes on Vista.
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

# this job's own output, so the parse below never mixes in an earlier job's lines
LOG=athena.$SLURM_JOB_ID.log
ibrun $BIN -r "$RST" "${ADD[@]}" -d ./ -t 00:45:00 time/tlim=$TLIM $OVR > $LOG 2>&1
STATUS=$?
cat $LOG >> run.log
echo "# athena exit status $STATUS at $(date)" >> run.log

kill "${MONPIDS[@]}" 2>/dev/null
sleep 2

# steady rate: first elapsed= line at least STEADY0 cycles past the restart cycle, to the last
NP=$(sed -n 's/.*inserted \([0-9]*\) dust particles.*/\1/p' $LOG | head -1)
awk -v ncell=$NCELL -v np=${NP:-0} -v s0=$STEADY0 -v nn=$SLURM_JOB_NUM_NODES -v arch="$ARCH" '
  /^elapsed=/ { split($1,a,"="); split($2,b,"="); t=a[2]+0; c=b[2]+0
                if (c0 == "") c0 = c
                if (c - c0 >= s0 && t1 == "") { t1 = t; k1 = c }
                t2 = t; k2 = c }
  END { if (t1 == "" || k2 <= k1) { print "# bench: too few cycles for a steady window"; exit }
        r = (k2-k1)/(t2-t1)
        printf "# bench %s, %d node(s): cycles %d-%d in %.1f s = %.3f s/cycle; %.3e zone-cycles/s (%.3e per node); %.3e particle-updates/s (%d particles)\n",
               arch, nn, k1, k2, t2-t1, 1/r, ncell*r, ncell*r/nn, np*r, np }' $LOG >> run.log
sacct -j "$SLURM_JOB_ID" --format=JobID,State,ExitCode,MaxRSS,Elapsed >> run.log 2>&1
