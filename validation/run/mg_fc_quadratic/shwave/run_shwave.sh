#!/bin/bash
# usage: run_shwave.sh BIN TAG INPUT [overrides]
BIN=$1; TAG=$2; INP=$3; shift 3
for N in 32 64 128; do
  MB=$((N/4))
  for T0 in 0.0 0.37; do
    mpirun -np 4 $BIN -i $INP mesh/nx1=$N mesh/nx2=$N mesh/nx3=$N meshblock/nx1=$MB \
      meshblock/nx2=$MB meshblock/nx3=$MB problem/profile=shwave problem/time0=$T0 \
      gravity/threshold=-1 gravity/niteration=30 "$@" > log_${TAG}_${N}_t$T0 2>&1
    echo "N=$N time0=$T0 $(grep 'SHWAVE ERROR' log_${TAG}_${N}_t$T0)"
  done
done
