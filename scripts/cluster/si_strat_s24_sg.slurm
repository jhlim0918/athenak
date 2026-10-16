#!/bin/bash
#SBATCH -J s24_sg
#SBATCH -p gg                    # Vista Grace-Grace CPU nodes (2 x 72 cores, no GPU)
#SBATCH -N 1
#SBATCH -n 64                    # MUST equal the rank count of the stage-1 run (64)
#SBATCH -t 12:00:00
#SBATCH -A AST25015
#SBATCH -o slurm.%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jay.lim1@unlv.edu

# Stage 2 of the Schafer+24 run_1.6_320 box: restart the stage-1 run (si_strat_s24_stage1.slurm)
# from its LAST restart dump with dust self-gravity on (uniform grid), using the add-on
# inputs/dust/streaming_strat_schafer24_sg_addon.athinput (gamma = 0.318, multigrid, dust_dpm
# and particles every 0.1 P).  Runs to TLIM (default: restart time + 20 P); athena stops itself
# 15 min before the walltime with a restart dump and the job resubmits itself (MAX_JOBS).
#
# Submit FROM A NEW, EMPTY RUN DIRECTORY (outputs land there; the parent is never written):
#     mkdir -p $SCRATCH/s24/L16_80_sg_uni && cd $SCRATCH/s24/L16_80_sg_uni \
#       && sbatch $HOME/athenak-si-strat-amr/scripts/cluster/si_strat_s24_sg.slurm
# Options (export before sbatch, or sbatch --export=ALL,VAR=...):
#     PARENT=<stage-1 dir>     default $SCRATCH/s24/L16_80
#     RST=<file>               a particular dump instead of the last one in $PARENT/rst
#     TLIM=<code time>         end time (P = 2 pi);  ADDON=<file>  another add-on (e.g. SMR/AMR)
#     MAX_JOBS=<n>             resubmissions allowed (default 4)
# A resubmitted job (or any job in a directory that already has rst/) continues from its own
# last dump; the first job copies the binary to ./athena so the chain never mixes builds.
#
# NEEDS si-strat-amr AT OR AFTER THE COMMIT THAT INITIALISES THE DUST BOUNDARY POINTERS
# (dust.hpp: pbval_qp/dm/us = nullptr): 7d96f3ef and older can crash at start-up in
# DustGasDrag::SetRefinementFactors (seen on the Mac, 2026-10-15).  Rebuild before submitting:
#     cd $HOME/athenak-si-strat-amr && git pull && module load gcc openmpi fftw3 \
#       && cmake --build build-cluster -j 16

ATHENAK=${ATHENAK:-$HOME/athenak-si-strat-amr}
PARENT=${PARENT:-$SCRATCH/s24/L16_80}
ADDON=${ADDON:-$ATHENAK/inputs/dust/streaming_strat_schafer24_sg_addon.athinput}
SRC_EXE=${SRC_EXE:-$ATHENAK/build-cluster/src/athena}
MAX_JOBS=${MAX_JOBS:-4}

module load gcc openmpi fftw3
cd $SLURM_SUBMIT_DIR
die() { echo "ERROR: $*" | tee -a run.log; exit 1; }
WS=$(squeue -h -j $SLURM_JOB_ID -o %l | awk -F'[-:]' '{s=0; for(i=1;i<=NF;i++) s=s*60+$i; print (NF==4? $1*86400+$2*3600+$3*60+$4 : s) - 900}')
WALL=$(printf '%d:%02d:%02d' $((WS/3600)) $(((WS%3600)/60)) $((WS%60)))
echo "# job $SLURM_JOB_ID on $SLURM_JOB_NODELIST, $SLURM_NTASKS ranks; started $(date); athena -t $WALL" >> run.log

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
  ibrun ./athena -r $RST -d ./ -t $WALL >> run.log 2>&1
else                                                    # first job: from the parent
  RST=${RST:-$(ls $PARENT/rst/*.rst 2>/dev/null | tail -1)}
  [ -f "$RST" ] || die "no restart file in $PARENT/rst"
  read T0 NW <<< "$(rstinfo $RST)"
  [ "$NW" = "$SLURM_NTASKS" ] || die "$RST was written by $NW ranks, this job has $SLURM_NTASKS"
  TLIM=${TLIM:-$(python3 -c "import math; print(f'{$T0 + 40*math.pi:.15g}')")}
  cp $SRC_EXE ./athena && cp "$ADDON" addon_used.athinput
  {
    echo "# parent $RST (t = $T0 = $(python3 -c "import math; print(f'{$T0/(2*math.pi):.3f}')") P, $NW ranks)"
    echo "# binary $SRC_EXE, commit $(git -C $ATHENAK rev-parse --short HEAD); add-on $ADDON; tlim $TLIM"
  } >> run.log
  ibrun ./athena -r $RST -i addon_used.athinput -d ./ -t $WALL time/tlim=$TLIM >> run.log 2>&1
fi
echo "# athena exit status $? at $(date)" >> run.log
echo $SLURM_JOB_ID >> jobs.list
if tail -40 run.log | grep -q "^Terminating on wall clock limit"; then
  if [ $(wc -l < jobs.list) -lt $MAX_JOBS ]; then
    sbatch --export=ALL,ATHENAK=$ATHENAK $ATHENAK/scripts/cluster/si_strat_s24_sg.slurm >> run.log 2>&1
  else
    echo "# MAX_JOBS = $MAX_JOBS reached; resubmit by hand from this directory" >> run.log
  fi
fi
