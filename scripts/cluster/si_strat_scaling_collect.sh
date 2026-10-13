#!/bin/bash
# Tabulates the "# bench" lines of a si_strat_scaling scan (run from the scan directory):
# per point the s/cycle, cell-updates/s per GPU, strong-scaling efficiency against the
# smallest GPU count of the same N and MB, peak GPU memory, and the projected cost of
# t = 100 P.  Points without a bench line (still queued, failed) are listed as such.
printf "%-14s %-5s %5s %10s %13s %7s %9s %9s %10s\n" "N" "MB" "GPUs" "s/cycle" \
  "cell-upd/s/GPU" "eff" "100P [h]" "GPU-h" "GPU mem MB"
for d in N*_mb*_g*; do
  [ -d "$d" ] || continue
  b=$(grep '^# bench N=' "$d/run.log" 2>/dev/null | tail -1)
  if [ -z "$b" ]; then
    st=$(grep '^# athena exit status' "$d/run.log" 2>/dev/null | tail -1)
    echo "$d ${st:-# no result yet}"
  else
    echo "$b"
  fi
done | awk '
  /^# bench/ {
    match($0, /N=[0-9x]+/);  n  = substr($0, RSTART+2, RLENGTH-2)
    match($0, /MB=[0-9]+/);  mb = substr($0, RSTART+3, RLENGTH-3)
    match($0, /GPUs=[0-9]+/); g = substr($0, RSTART+5, RLENGTH-5)
    split($0, f, " ")
    for (i in f) {
      if (f[i] == "s/cycle,") spc = f[i-1]
      if (f[i] == "GPU),")    pg  = f[i-2]
      if (f[i] == "wall")     h   = f[i-2]
      if (f[i] == "GPU-h,")   gh  = f[i-1]
      if (f[i] == "MB" && f[i-1] ~ /^[0-9]+$/) mem = f[i-1]
    }
    sub(/\(/, "", pg)
    key = n "_" mb; row[++nr] = key SUBSEP g SUBSEP spc SUBSEP pg SUBSEP h SUBSEP gh SUBSEP mem
    if (!(key in g0) || g + 0 < g0[key]) { g0[key] = g + 0; s0[key] = spc + 0 }
    next }
  { other[++no] = $0 }
  END {
    for (i = 1; i <= nr; i++) {
      split(row[i], r, SUBSEP); split(r[1], k, "_")
      eff = (s0[r[1]] * g0[r[1]]) / (r[3] * r[2])
      printf "%-14s %-5s %5d %10.4f %13s %7.2f %9s %9s %10s\n", k[1], k[2], r[2], r[3], r[4], eff, r[5], r[6], r[7]
    }
    for (i = 1; i <= no; i++) print other[i]
  }' | sort -k1,1V -k2,2n -k3,3n
