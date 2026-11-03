#!/bin/bash
# usage: ladder.sh BIN TAG NP MB [extra overrides...]
# MB = 4 (4^3 blocks, nx3=4) or 8 (8^3 blocks, nx3=8)
BIN=$1; TAG=$2; NP=$3; MB=$4; shift 4
INP=${INP:-../../../inputs/tests/mg_interface_wave.athinput}
if [ "$MB" = 4 ]; then NS="16 32 64 128 256"; else NS="32 64 128 256"; fi
for N in $NS; do
  X3=$(python3 -c "print($MB/$N)")
  mpirun -np $NP $BIN -i $INP mesh/nx1=$N mesh/nx2=$N mesh/nx3=$MB mesh/x3max=$X3 \
     refined_region1/x3max=$X3 meshblock/nx1=$MB meshblock/nx2=$MB meshblock/nx3=$MB \
     gravity/threshold=-1 gravity/niteration=40 "$@" > log_${TAG}_mb${MB}_$N 2>&1
  echo "N=$N $(grep MG_INTERFACE_WAVE log_${TAG}_mb${MB}_$N)"
done
