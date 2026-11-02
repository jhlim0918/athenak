#!/bin/bash
# s0 = 25 cases (KOS02 Table 1 "spiral arm"), alongside the main queue: 2 at a time, 2 ranks.
cd "$(dirname "$0")"
ATH=$HOME/athenak-worktrees/fargo-amr/build-fft/src/athena
run1() { n=$(basename $1 .athinput); mkdir -p runs/$n; cd runs/$n
  [ -f done ] && return; t0=$(date +%s)
  mpirun -np 2 $ATH -i ../../$1 > log 2>&1; echo "$n exit=$? $(( $(date +%s)-t0 ))s" > done; }
export -f run1; export ATH
ls inputs/s0_25_*.athinput | xargs -P 2 -I{} bash -c 'run1 {}'
echo DONE25
