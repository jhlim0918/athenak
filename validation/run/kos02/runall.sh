#!/bin/bash
# Run every KOS02 tier-1 case: four at a time, 4 MPI ranks each; s0 = 0.01 last.
cd "$(dirname "$0")"
ATH=$HOME/athenak-worktrees/fargo-amr/build-fft/src/athena
list=$(ls inputs/*.athinput | grep -v "s0_0.01_"; ls inputs/s0_0.01_*.athinput)
run1() { n=$(basename $1 .athinput); mkdir -p runs/$n; cd runs/$n
  [ -f done ] && return; t0=$(date +%s)
  mpirun -np 4 $ATH -i ../../$1 > log 2>&1; echo "$n exit=$? $(( $(date +%s)-t0 ))s" | tee done; }
export -f run1; export ATH
echo "$list" | xargs -P 4 -I{} bash -c 'run1 {}'
echo ALLDONE
