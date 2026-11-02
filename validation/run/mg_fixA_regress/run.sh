#!/bin/bash
# Regression of the multigrid fine/coarse ghost fix (ProlongateFCMG transverse slopes): every case with the
# committed code (athena_before = 0583ec1a build) and the fix (athena_after), in <case>_<before|after>/.
I=$HOME/athenak-worktrees/fargo-amr/inputs/tests
F=output1/data_format=%.15e
run() { c=$1; v=$2; np=$3; shift 3; d=${c}_$v; mkdir -p $d; (cd $d && mpirun --bind-to none -np $np ../athena_${v%_np1} "$@" > log 2>&1; echo "exit=$?" > done); }
for v in before after; do
  run swing_unif $v 4 -i $I/swing_selfgrav_smr.athinput mesh_refinement/refinement=none $F &
  run swing_ring $v 4 -i $I/swing_selfgrav_smr.athinput $F &
  run swing_patch $v 4 -i $I/swing_selfgrav_smr_ypatch.athinput $F &
  wait
  run swing_amr $v 4 -i $I/swing_selfgrav_amr.athinput $F &
  run poisson_slab_smr $v 2 -i $I/mg_poisson_slab_smr.athinput &
  run poisson_shear_smr $v 2 -i $I/mg_poisson_shear_smr.athinput &
  run mhd_ring $v 3 -i ../../fargo_amr/mhd/swing/mesh_smr.athinput output2/dt=100 &
  wait
done
# MPI: the fixed code, serial vs 3 ranks on the refined swing ring
run swing_ring after_np1 1 -i $I/swing_selfgrav_smr.athinput $F
echo ALLDONE
