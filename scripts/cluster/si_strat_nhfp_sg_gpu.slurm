#!/bin/bash
#SBATCH -J nhfp_sg
#SBATCH -p gh                    # Vista Grace-Hopper nodes: one GH200 each
#SBATCH -N 4                     # MUST equal the rank count of the stage-1 pilot (4 GPUs)
#SBATCH -n 4
#SBATCH -t 08:00:00                # 5 P is ~1-2 h if dt holds; collapse shrinks dt
#SBATCH -A AST25015
#SBATCH -o slurm.%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jay.lim1@unlv.edu

# Stage 2 of the NHFP pilot ((1.6H)^2 x 0.2H, 320/H, four species; si_strat_gpu.slurm): restart
# the stage-1 run from its LAST restart dump with dust self-gravity on (uniform grid), using
# inputs/dust/streaming_strat_nhfp_sg_addon.athinput (4 pi G rho0/Omega^2 = 0.0723, multigrid,
# dust_dpm and particles every 0.01 P, restarts every 1 P).  Runs to TLIM (default: restart
# time + 5 P); athena stops itself 15 min before the walltime with a restart dump and the job
# resubmits itself (MAX_JOBS).  Same binary as stage 1 (build-gpu of dust-multigrid).
#
# Submit FROM A NEW, EMPTY RUN DIRECTORY (outputs land there; the parent is never written):
#     mkdir -p $SCRATCH/nhfp/L16_320_sg && cd $SCRATCH/nhfp/L16_320_sg \
#       && sbatch $HOME/athenak-multigrid/scripts/cluster/si_strat_nhfp_sg_gpu.slurm
# Options (export before sbatch, or sbatch --export=ALL,VAR=...):
#     PARENT=<stage-1 dir>     default $SCRATCH/nhfp/L16_320
#     RST=<file>               a particular dump instead of the last one in $PARENT/rst
#     TLIM=<code time>         end time (P = 2 pi);  ADDON=<file>  another add-on
#     XOVR="block/par=value"   extra overrides, e.g. XOVR="gravity/four_pi_G=0.0745" (45 au)
#     MAX_JOBS=<n>             resubmissions allowed (default 4)
# A resubmitted job (or any job in a directory that already has rst/) continues from its own
# last dump; the first job copies the binary to ./athena so the chain never mixes builds.
# Output: 500 pvtk (~0.6 GB) + 500 dust_dpm (67 MB) files in 5 P, ~335 GB; keep $SCRATCH in mind.
# Cost: stage 1 ran 100 P on these 4 GPUs; the multigrid adds 4 V-cycles per stage, and dt
# falls as clumps collapse, so check run.log after the first hours.

ATHENAK=${ATHENAK:-$HOME/athenak-multigrid}
PARENT=${PARENT:-$SCRATCH/nhfp/L16_320}
ADDON=${ADDON:-$ATHENAK/inputs/dust/streaming_strat_nhfp_sg_addon.athinput}
SRC_EXE=${SRC_EXE:-$ATHENAK/build-gpu/src/athena}
MAX_JOBS=${MAX_JOBS:-4}

module load nvidia/24.7 openmpi/5.0.5_nvc249
export UCX_MEMTYPE_CACHE=n       # see gt_bc16_stage1_gpu.slurm (unified-memory build)
cd $SLURM_SUBMIT_DIR
die() { echo "ERROR: $*" | tee -a run.log; exit 1; }

# stop the binary ~15 min before the job ends, at whatever -t the job got (si_strat_gpu.slurm)
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
ATIME=$(tleft) || ATIME=7:45:00
echo "# job $SLURM_JOB_ID on $SLURM_JOB_NUM_NODES node(s): $SLURM_JOB_NODELIST, $SLURM_NTASKS ranks (GPU); started $(date); athena -t $ATIME" >> run.log

# time and writer rank count of a restart file (the particle block stores one count per rank)
rstinfo() { python3 - "$1" <<'EOF'
import sys, struct
b = open(sys.argv[1], "rb").read()
k = b.index(b"\n", b.index(b"<par_end>")) + 1
nmb, = struct.unpack_from("<i", b, k)
time, = struct.unpack_from("<d", b, k + 8 + 72 + 76 + 76)
ng, n1, n2, n3 = struct.unpack_from("<4i", b, k + 8 + 72 + 76)
o = k + 8 + 72 + 76 + 76 + 16 + 4 + 16*nmb + 4*nmb
cells = (n1 + 2*ng)*(n2 + 2*ng)*(n3 + 2*ng)
for nr in range(1, 65537):
    nrd, nid = struct.unpack_from("<ii", b, o + 4*nr); ds, = struct.unpack_from("<Q", b, o + 4*nr + 8)
    if nrd == 17 and nid in (3, 4) and ds % (8*cells) == 0 and ds//(8*cells) in (4, 5):
        print(f"{time:.15g} {nr}"); break
else:
    print(f"{time:.15g} 0")
EOF
}

if ls rst/*.rst > /dev/null 2>&1; then                  # continuation of this chain
  RST=$(ls rst/*.rst | tail -1)
  echo "# continuing from $RST" >> run.log
  ibrun ./athena -r $RST -d ./ -t $ATIME >> run.log 2>&1
else                                                    # first job: from the parent
  RST=${RST:-$(ls $PARENT/rst/*.rst 2>/dev/null | tail -1)}
  [ -f "$RST" ] || die "no restart file in $PARENT/rst"
  read T0 NW <<< "$(rstinfo $RST)"
  [ "$NW" = "$SLURM_NTASKS" ] || die "$RST was written by $NW ranks, this job has $SLURM_NTASKS"
  TLIM=${TLIM:-$(python3 -c "import math; print(f'{$T0 + 10*math.pi:.15g}')")}
  OVR=(time/tlim=$TLIM ${XOVR:-})
  cp $SRC_EXE ./athena && cp "$ADDON" addon_used.athinput
  {
    echo "# parent $RST (t = $T0 = $(python3 -c "import math; print(f'{$T0/(2*math.pi):.3f}')") P, $NW ranks)"
    echo "# binary $SRC_EXE, commit $(git -C $ATHENAK rev-parse --short HEAD); add-on $ADDON; ${OVR[*]}"
  } >> run.log
  ibrun ./athena -r $RST -i addon_used.athinput -d ./ -t $ATIME "${OVR[@]}" >> run.log 2>&1
fi
echo "# athena exit status $? at $(date)" >> run.log
echo $SLURM_JOB_ID >> jobs.list
if tail -40 run.log | grep -q "^Terminating on wall clock limit"; then
  if [ $(wc -l < jobs.list) -lt $MAX_JOBS ]; then
    sbatch --export=ALL,ATHENAK=$ATHENAK $ATHENAK/scripts/cluster/si_strat_nhfp_sg_gpu.slurm >> run.log 2>&1
  else
    echo "# MAX_JOBS = $MAX_JOBS reached; resubmit by hand from this directory" >> run.log
  fi
fi
