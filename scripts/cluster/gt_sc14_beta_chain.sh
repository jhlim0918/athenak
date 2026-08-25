#!/bin/bash
# SC14 beta-scan "morphing" chain (their sec. 2.3): each constant-beta run is
# initialized from the FINAL restart dump of an earlier run so the disk adjusts
# from an already gravito-turbulent state instead of re-suffering the cold
# collapse.  SC14 chain:  10 -> 20 -> 40 -> 80 -> 120,  and  10 -> {4, 5, 8}.
# Durations follow their Table 1 (standard resolution): the averaging window is
# the trailing 100-200/Omega of each run.
#
# Usage (repo root, after runs/gt_sc14_b10 has finished via gt_sc14_full.pbs):
#   scripts/cluster/gt_sc14_beta_chain.sh print          # show the commands only
#   scripts/cluster/gt_sc14_beta_chain.sh run 20 150     # one stage: beta, duration
#   scripts/cluster/gt_sc14_beta_chain.sh run 20 150 gt_sc14_b10   # explicit source
#
# Each stage restarts from the last rst of its source run with the cooling time
# overridden on the command line and tlim extended by the stage duration; outputs
# land in runs/gt_sc14_b<beta>.  Wrap the 'run' invocation in your PBS script (same
# resources as gt_sc14_full.pbs).  Written for bash 3.2 (macOS) and Linux.

set -eu

REPO=$(cd "$(dirname "$0")/../.." && pwd)
ATHENA=${ATHENA:-$REPO/build-cluster/src/athena}
NRANKS=${NRANKS:-64}
MPIEXEC=${MPIEXEC:-mpiexec -np $NRANKS}

# default source of each stage (SC14 morphing order)
chain_source() {
  case "$1" in
    4|5|8|20) echo gt_sc14_b10 ;;
    40)       echo gt_sc14_b20 ;;
    80)       echo gt_sc14_b40 ;;
    120)      echo gt_sc14_b80 ;;
    *)        echo gt_sc14_b10 ;;
  esac
}

# SC14 Table 1 durations (standard resolution), Omega^-1
chain_duration() {
  case "$1" in
    4|5|8) echo 200 ;;
    20)    echo 150 ;;
    40)    echo 300 ;;
    80)    echo 400 ;;
    120)   echo 500 ;;
    *)     echo 300 ;;
  esac
}

last_rst() {  # newest restart dump of a run dir
  ls "$REPO/runs/$1"/rst/*.rst 2>/dev/null | sort | tail -1
}

rst_time() {  # simulation time from the restart's binary header: after the
  # "<par_end>" sentinel come int nmb_total, int root_level, RegionSize (9 Real),
  # RegionIndcs x2 (19 int each), then Real time  (src/outputs/restart.cpp)
  python3 - "$1" << 'PYEOF'
import struct, sys
data = open(sys.argv[1], 'rb').read(1 << 22)
off = data.index(b'<par_end>')
off = data.index(b'\n', off) + 1
off += 4 + 4 + 9*8 + 2*19*4
print(struct.unpack_from('d', data, off)[0])
PYEOF
}

stage() {  # stage <beta> <duration> [source_run]
  beta=$1; dur=$2; src=${3:-$(chain_source "$beta")}
  rst=$(last_rst "$src")
  [ -n "$rst" ] || { echo "no rst found in runs/$src" >&2; exit 1; }
  t0=$(rst_time "$rst")
  tlim=$(python3 -c "print($t0 + $dur)")
  run=$REPO/runs/gt_sc14_b$beta
  mkdir -p "$run"
  {
    echo "restarted from: $rst (t = $t0)"
    echo "beta override:  hydro_srcterms/bcool_beta=$beta, tlim = $tlim"
    echo "date: $(date); commit: $(git -C "$REPO" rev-parse HEAD)"
  } > "$run/provenance.txt"
  echo "# beta=$beta: restart $rst (t=$t0) -> tlim=$tlim"
  echo $MPIEXEC "$ATHENA" -r "$rst" -d "$run" \
    hydro_srcterms/bcool_beta=$beta time/tlim=$tlim
}

case "${1:-print}" in
  print)
    echo "SC14 morphing chain (run each stage after its source finishes):"
    for b in 20 40 80 120 4 5 8; do
      printf "  %s run %-3s %-3s   # from runs/%s\n" \
        "$0" "$b" "$(chain_duration $b)" "$(chain_source $b)"
    done
    ;;
  run)
    beta=${2:?usage: run <beta> [duration] [source_run]}
    dur=${3:-$(chain_duration "$beta")}
    cmd=$(stage "$beta" "$dur" "${4:-}" | tail -1)
    echo "+ $cmd"
    eval "$cmd" > "$REPO/runs/gt_sc14_b$beta/run.log" 2>&1
    ;;
  *)
    echo "usage: $0 print | run <beta> [duration] [source_run]" >&2
    exit 1
    ;;
esac
