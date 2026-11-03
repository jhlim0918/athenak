#!/bin/bash
# usage: run_cube.sh BIN TAG MB INTERP "N list" [overrides]
# refined cube [0.25,0.75]^3, wave (2,1,1), MB^3 MeshBlocks, 4 ranks, 30 V-cycles (FMG)
BIN=$1; TAG=$2; MB=$3; FCI=$4; NS=$5; shift 5
INP=../../../../inputs/tests/mg_interface_wave_cube.athinput
for N in $NS; do
  mpirun -np 4 $BIN -i $INP mesh/nx1=$N mesh/nx2=$N mesh/nx3=$N meshblock/nx1=$MB \
    meshblock/nx2=$MB meshblock/nx3=$MB gravity/mg_fc_interp=$FCI "$@" > log_${TAG}_mb${MB}_$N 2>&1
  echo "N=$N $(grep MG_INTERFACE_WAVE log_${TAG}_mb${MB}_$N)"
done
