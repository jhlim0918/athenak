#!/bin/bash
# Dust growth (src/dust/dust_growth.cpp) test suite D3, run from the repository root:
#   scripts/analysis/dust_growth_tests.sh [a|b|c|d|e|all]
# Outputs go to validation/run/dust_growth/<test>/; scripts/analysis/dust_growth_tests.py
# checks them.  Test (a) needs a baseline binary of the parent commit in $BASE_ATHENA
# (default: the si-strat-amr worktree's build at 9aaa5639).
set -e
ROOT=$(pwd)
ATHENA=${ATHENA:-$ROOT/build/src/athena}
BASE_ATHENA=${BASE_ATHENA:-/Users/jaysmac/athenak-worktrees/si-strat-amr/build/src/athena}
NP=${NP:-4}
RUN=$ROOT/validation/run/dust_growth
IN=$ROOT/inputs/dust
which=${1:-all}

# AthenaK's command line can only replace parameters already in the input file, so every
# run gets a patched copy (in.athinput): block/name=value sets or adds the parameter
# (and the block)
patch_input() {   # patch_input <src> <dst> [block/name=value ...]
  python3 - "$@" <<'PY'
import re, sys
src, dst, sets = sys.argv[1], sys.argv[2], sys.argv[3:]
lines = open(src).read().splitlines()
for kv in sets:
    blk, rest = kv.split('/', 1)
    name, val = rest.split('=', 1)
    cur, ib, hit = None, None, False
    for i, l in enumerate(lines):
        m = re.match(r'\s*<(\S+)>', l)
        if m:
            cur = m.group(1)
            if cur == blk: ib = i
            continue
        if cur == blk and re.match(r'\s*' + re.escape(name) + r'\s*=', l):
            lines[i] = f'{name} = {val}'; hit = True
    if not hit:
        if ib is None:
            lines += ['', f'<{blk}>']; ib = len(lines) - 1
        lines.insert(ib + 1, f'{name} = {val}')
open(dst, 'w').write('\n'.join(lines) + '\n')
PY
}

run() {   # run <dir> <exe> <np> <input> [block/name=value ...]
  local dir=$1 exe=$2 np=$3 inp=$4; shift 4
  rm -rf "$dir"; mkdir -p "$dir"
  patch_input "$inp" "$dir/in.athinput" "$@"
  ( cd "$dir" && mpirun -np "$np" "$exe" -i in.athinput > log.txt 2>&1 )
}

GROW="dust/stopping_time_mode=dynamic dust/growth=monodisperse"

# (a) growth = none (the default) is bitwise identical to the parent commit
if [[ $which == a || $which == all ]]; then
  for v in base new; do
    exe=$ATHENA; [[ $v == base ]] && exe=$BASE_ATHENA
    run $RUN/a/$v/nsh2d $exe 1 $IN/dust_nsh.athinput time/tlim=2.0
    run $RUN/a/$v/amr3d $exe $NP $IN/dust_nsh_shear3d_amr.athinput
    run $RUN/a/$v/yj14 $exe $NP $IN/streaming_strat_yj14.athinput time/nlim=60 \
        output1/dt=0.01 output2/dt=0.01 output3/dt=0.05 output4/dt=0.05 \
        output5/dt=0.05 output5/vtk_dbl=true output6/dt=1e9
  done
fi

# (b) uniform growth vs the analytic S(t), three timesteps (first order)
if [[ $which == b || $which == all ]]; then
  for c in 0.45 0.225 0.1125; do
    run $RUN/b/cfl$c $ATHENA 1 $IN/dust_growth_box.athinput time/cfl_number=$c
  done
fi

# (c) the turbulent fragmentation cap S_frag = v_frag^2/(3 alpha_t c_s^2) = 0.0333
if [[ $which == c || $which == all ]]; then
  run $RUN/c $ATHENA 1 $IN/dust_growth_box.athinput dust/growth_vfrag=0.01
fi

# (d) drift term: first-cycle rate vs the three-species NSH relative velocities, in 2D
# r-z, 3D with orbital advection, and 3D without (gas in the full frame)
if [[ $which == d || $which == all ]]; then
  D="$GROW dust/growth_alpha=0 dust/growth_vfrag=10 dust/growth_smax=10 \
     problem/check_equilibrium=false time/nlim=1 output9/file_type=pvtk \
     output9/variable=prtcl_all output9/dcycle=1 output9/vtk_dbl=true"
  run $RUN/d/rz2d $ATHENA 1 $IN/dust_nsh.athinput $D
  run $RUN/d/sb3d $ATHENA $NP $IN/dust_nsh_shear3d.athinput $D
  run $RUN/d/sb3d_nofargo $ATHENA $NP $IN/dust_nsh_shear3d.athinput $D \
      shearing_box/orbital_advection=false
fi

# (e) the stopping times survive a restart (bitwise) and AMR refine/derefine with
# particle splitting (4 ranks)
if [[ $which == e || $which == all ]]; then
  E="problem/etavk=0.05 hydro_srcterms/const_accel_val=0.1 dust/back_reaction=true \
     time/tlim=4.0 output9/file_type=pvtk output9/variable=prtcl_all output9/dt=1.0 \
     output9/vtk_dbl=true"
  # the straight run writes the restart file itself (not a run stopped at tlim = 2, whose
  # shortened last step would change the timestep sequence)
  run $RUN/e/full $ATHENA 1 $IN/dust_growth_box.athinput $E output8/file_type=rst \
      output8/dt=2.0
  mkdir -p $RUN/e/rst
  ( cd $RUN/e/rst && mpirun -np 1 $ATHENA -r ../full/rst/dgrow_box.00001.rst \
      > log_restart.txt 2>&1 )
  # AMR: growth through the refine (t = 1) and derefine (t = 4) events on 4 ranks, writing
  # a restart at t ~ 0.5.  Split check: restart there with a zero growth rate, so every
  # t_s (grown, no longer the species value) must pass the particle split unchanged.
  # (An AMR-vs-uniform comparison is not a test here: refining this NSH lattice makes
  # the PM deposit non-uniform, dpm_max up to ~1.3, with or without growth.)
  A="$GROW dust/growth_alpha=1e-3 dust/growth_vfrag=10 dust/growth_smax=10 \
     problem/check_equilibrium=false"
  run $RUN/e/amr $ATHENA $NP $IN/dust_nsh_shear3d_amr.athinput $A output2/file_type=phst \
      output2/dt=0.05 output2/data_format=%.16e output8/file_type=rst output8/dt=0.5
  rm -rf $RUN/e/split; mkdir -p $RUN/e/split
  ( cd $RUN/e/split && mpirun -np $NP $ATHENA -r ../amr/rst/nsh3d_amr.00001.rst \
      dust/growth_alpha=0 dust/growth_drift=false time/tlim=1.5 > log.txt 2>&1 )
fi
