#!/bin/bash
#SBATCH -J si3d
#SBATCH -p gh                    # Vista Grace-Hopper nodes: one GH200 (H100 96 GB) each
#SBATCH -N 4                     # one rank per GPU; 64 MeshBlocks / 4 = 16 per GPU at 320/H
#SBATCH -n 4
#SBATCH -t 24:00:00
#SBATCH -A AST25015
#SBATCH -o slurm.%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jay.lim1@unlv.edu

# Production runner for the stratified streaming-instability box (streaming_strat pgen) on
# Vista GPUs.  Default input: the NHFP pilot, inputs/dust/streaming_strat_nhfp_L16.athinput
# ((1.6H)^2 x 0.2H at 320/H, 16.8M cells and particles, four species, Z = 0.04, Pi = 0.05,
# 100 P).  Submit FROM THE RUN DIRECTORY in $SCRATCH (outputs land there):
#     mkdir -p $SCRATCH/nhfp/L16_320 && cd $SCRATCH/nhfp/L16_320 \
#       && sbatch $HOME/athenak-multigrid/scripts/cluster/si_strat_gpu.slurm
# Continue after a walltime stop (the binary stops itself ~15 min before the job end and
# writes rst/), or to run past tlim:   CONT=1 [TLIM=<code time>] sbatch ... (same directory).
# Other inputs/resolutions: INPUT=<athinput> and/or XOVR="mesh/nx1=... meshblock/nx3=...".
# NB a restart takes every parameter from the .rst; changes made to the input after the
# start must be repeated in XOVR.  A particle restart needs the SAME number of ranks
# (-N/-n) as the job that wrote it: the file stores one particle count per writing rank
# and the reader assumes its own rank count (a mismatch dies with a misleading "particle
# array widths ... differ" message).  $SCRATCH is purged after 10 days without access: copy
# finished runs to $WORK or Ranch (scripts/cluster/archive_runs.slurm).
#
# Cost (pre-measurement): dt = 8.9e-4 at 320/H, so 100 P = 7.1e5 cycles; at 1e8
# cell-updates/s per GH200 that is ~33 GPU-h, ~8 h on 4 GPUs.  The 256^3 points of the
# scaling scan (si_strat_scaling_*) have the same 16.8M cells and give the real rate.

ATHENAK=$HOME/athenak-multigrid
INPUT=${INPUT:-$ATHENAK/inputs/dust/streaming_strat_nhfp_L16.athinput}
BIN=${BIN:-$ATHENAK/build-gpu/src/athena}
RST=${RST:-}
if [ -n "${CONT:-}" ] && [ -z "$RST" ]; then RST=$(ls rst/*.rst | tail -1); fi
if [ -n "$RST" ]; then START=(-r "$RST"); else START=(-i "$INPUT"); fi
if [ -n "${TLIM:-}" ]; then OVR=(time/tlim=$TLIM); else OVR=(); fi
OVR+=(${XOVR:-})

module load nvidia/24.7 openmpi/5.0.5_nvc249
export UCX_MEMTYPE_CACHE=n       # see gt_bc16_stage1_gpu.slurm (unified-memory build)

cd $SLURM_SUBMIT_DIR
{
  echo "# job $SLURM_JOB_ID on $SLURM_JOB_NUM_NODES node(s): $SLURM_JOB_NODELIST, $SLURM_NTASKS ranks (GPU)"
  echo "# started $(date); commit $(git -C $ATHENAK rev-parse --short HEAD); start ${START[*]} ${OVR[*]}"
} >> run.log
[ -z "$RST" ] && cp "$INPUT" input_used.athinput

# stop the binary ~15 min before the job ends, at whatever -t the job got (as in
# gt_bc16_stage1_gpu.slurm), so it writes its own restart dump
tleft() {
  local end="${SLURM_JOB_END_TIME:-}"
  if [ -z "$end" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    end=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
          | tr ' ' '\n' | sed -n 's/^EndTime=//p' | head -1)
    [ -n "$end" ] && end=$(date -d "$end" +%s 2>/dev/null)
  fi
  [ -z "$end" ] && return 1
  local left=$(( end - $(date +%s) - 900 ))
  [ "$left" -le 0 ] && return 1
  printf '%d:%02d:%02d' $((left/3600)) $(((left%3600)/60)) $((left%60))
}
ATIME=$(tleft) || ATIME=23:45:00
echo "# athena wall-clock limit: $ATIME" >> run.log

ibrun $BIN "${START[@]}" -d ./ -t $ATIME "${OVR[@]}" >> run.log 2>&1
echo "# athena exit status $? at $(date)" >> run.log
