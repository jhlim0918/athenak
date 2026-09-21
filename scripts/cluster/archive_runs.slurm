#!/bin/bash
#SBATCH -J athenak-archive
#SBATCH -p gg                    # Vista Grace-Grace CPU nodes
#SBATCH -N 1
#SBATCH -n 1                     # tar is single-threaded; TACC allocates whole nodes
#SBATCH -t 04:00:00
#SBATCH -A AST25015
#SBATCH -o archive.%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jay.lim1@unlv.edu

# Tar a finished AthenaK run out of the purge-prone $SCRATCH and into $WORK,
# ready to ship to NAS Lou from there.
# The Vista counterpart of scripts/cluster/backup_run.sh (NAS/shiftc only).
#
# Run this as a BATCH JOB, not on a login node: a 20+ GB tar is exactly the kind of
# sustained I/O TACC kills login-node processes for.
#
# Usage -- submit FROM the run directory; the tars land in $WORK/athenak_archive:
#     cd $SCRATCH/gt_sc14_stndrd/b20
#     sbatch $HOME/athenak-multigrid/scripts/cluster/archive_runs.slurm
#
# Explicit directories also work, several runs per job:
#     sbatch $HOME/athenak-multigrid/scripts/cluster/archive_runs.slurm $SCRATCH/gt_sc14_stndrd/b{20,3_unif,3_amr}
#
# Environment overrides:
#     DEST=<dir>   where the tars go             (default $WORK/athenak_archive)
#     FORCE=1      overwrite existing tars       (default: skip, exit non-zero)
#     NOSUM=1      skip the sha256 manifest      (default: checksum everything)
#
# Two archives per run, mirroring backup_run.sh:
#     <name>_meta.tar   top-level regular files: .hst, logs, slurm scripts,
#                       provenance, frozen input, .npz analysis products (small --
#                       grab this alone when you only want the history)
#     <name>_data.tar   bin/ and rst/ (the bulk)
# Both store paths under <name>/, so `tar -xf <name>_data.tar` recreates the run
# directory rather than spilling bin/ into the current directory.
#
# The meta list excludes *.tar, *.sha256 and this job's own archive.*.out, so the
# archive never contains itself and never tars a file that is still being written.

set -u -o pipefail

FORCE=${FORCE:-0}
NOSUM=${NOSUM:-0}

# no arguments => archive the directory the job was submitted from
if [ "$#" -eq 0 ]; then
  set -- "${SLURM_SUBMIT_DIR:-$PWD}"
fi

echo "# job ${SLURM_JOB_ID:-interactive} on $(hostname), started $(date)"
echo

rc_all=0

# Free space in KB at $1, and a human label for it.  On a Lustre filesystem (TACC
# $WORK, $SCRATCH) `df` reports the whole shared filesystem -- 3.5 PB on Stockyard --
# which says nothing about what this user may write, so a df-based guard sails past a
# full quota and tar then dies mid-write with "Wrote only N of M bytes".  Ask the quota
# first and fall back to df only when there is no usable answer.
#
# `lfs quota -q -u <user> <path>` prints one line per filesystem:
#   <fs> <kbytes-used> <soft-quota> <hard-limit> <grace> <files> <fsoft> <fhard> <grace>
# A value of 0 means "no limit" for that column, and a trailing '*' marks a limit that
# is already exceeded.  Use the soft quota when it is set, otherwise the hard limit.
# Prints "<available-KB> <source>".  It runs in a command substitution, so it cannot
# set a variable in the caller -- the source is returned alongside the number.
avail_kb() {
  local d=$1 toks used lim soft hard
  if command -v lfs >/dev/null 2>&1; then
    # Flatten to a token stream: lfs wraps the numbers onto the next line when the
    # filesystem path is long (it does for $WORK on Vista), so line-based field
    # numbering is not reliable.  Drop the headers, drop the leading path token, and
    # the remainder is: used soft hard grace files fsoft fhard fgrace.
    toks=$(lfs quota -q -u "${USER:-$(id -un)}" "$d" 2>/dev/null \
             | tr -d '*' | grep -v -i '^ *Disk quotas' | grep -v -i 'Filesystem' | tr -s ' \n' ' ')
    set -- $toks
    case "${1:-}" in /*) shift ;; esac
    used=${1:-}; soft=${2:-}; hard=${3:-}
    # plain integers only: a unit suffix (lfs -h) or anything odd falls through to df
    case "$used$soft$hard" in
      ''|*[!0-9]*) ;;
      *) soft=$((soft)); hard=$((hard)); lim=$soft; [ "$lim" -gt 0 ] || lim=$hard
         if [ "$lim" -gt 0 ]; then
           if [ "$lim" -gt "$used" ]; then echo "$((lim - used)) quota"; else echo "0 quota"; fi
           return
         fi ;;
    esac
  fi
  echo "$(df -Pk "$d" 2>/dev/null | tail -1 | awk '{print $4+0}') df"
}

hr_kb() {  # KB -> human, without assuming GNU numfmt
  awk -v k="$1" 'BEGIN{ s="KMGTP"; i=1; while (k>=1024 && i<5) {k/=1024; i++}
                        printf "%.1f%s", k, substr(s,i,1) }'
}

archive_one() {          # archive_one <tarball> <parent dir> <file list file>
  local tarball="$1" parent="$2" listfile="$3"
  local n_expected n_actual
  n_expected=$(wc -l < "$listfile")
  if [ "$n_expected" -eq 0 ]; then
    echo "    $(basename "$tarball"): nothing to archive, skipped"
    return 0
  fi
  if [ -e "$tarball" ] && [ "$FORCE" != "1" ]; then
    echo "    REFUSING to overwrite $(basename "$tarball") -- set FORCE=1 to replace" >&2
    return 1
  fi
  # -b 2048 => 1 MB records, friendlier to Lustre than tar's 10 kB default
  if ! tar -C "$parent" -b 2048 -cf "$tarball" -T "$listfile"; then
    echo "    tar FAILED for $(basename "$tarball")" >&2
    rm -f "$tarball"
    return 1
  fi
  # verify: the archive must list back exactly the files that went in
  n_actual=$(tar -tf "$tarball" | grep -c -v '/$')
  if [ "$n_actual" -ne "$n_expected" ]; then
    echo "    VERIFY FAILED: $n_expected files in, $n_actual out" >&2
    return 1
  fi
  echo "    $(basename "$tarball"): $n_actual files, $(du -h "$tarball" | cut -f1), verified"
  return 0
}

for arg in "$@"; do
  run=$(cd "$arg" 2>/dev/null && pwd) || { echo "!! no such directory: $arg" >&2; rc_all=1; continue; }
  name=$(basename "$run")
  parent=$(dirname "$run")
  dest=${DEST:-$WORK/athenak_archive}
  mkdir -p "$dest" || { rc_all=1; continue; }

  need_kb=$(du -sk "$run" | cut -f1)
  read -r free_kb SPACE_SRC <<< "$(avail_kb "$dest")"
  echo "== $name  ($(du -sh "$run" | cut -f1) in $run)"
  echo "    tars -> $dest   (need ~$(hr_kb "$need_kb"), $(hr_kb "$free_kb") available by $SPACE_SRC)"
  if [ "${free_kb:-0}" -lt "$need_kb" ]; then
    echo "    NOT ENOUGH SPACE at $dest: the tars need about as much room as the run" >&2
    echo "    ($(hr_kb "$need_kb") needed, $(hr_kb "$free_kb") available by $SPACE_SRC)." >&2
    echo "    Check the quota with:  lfs quota -h -u \$USER $dest" >&2
    echo "    \$WORK is usually the tight one; stage on \$SCRATCH instead:" >&2
    echo "      sbatch --export=ALL,DEST=\$SCRATCH/athenak_archive $0 $run" >&2
    rc_all=1; echo; continue
  fi

  tmp=$(mktemp -d) || { rc_all=1; continue; }

  # metadata: top-level regular files, minus our own products and the live job log
  (cd "$parent" && find "$name" -maxdepth 1 -type f \
      ! -name '*.tar' ! -name '*.sha256' ! -name "archive.${SLURM_JOB_ID:-NONE}.out") \
    | sort > "$tmp/meta.list"
  archive_one "$dest/${name}_meta.tar" "$parent" "$tmp/meta.list" || rc_all=1

  # bulk: bin/ and rst/, whichever exist
  : > "$tmp/data.list"
  for sub in bin rst; do
    [ -d "$run/$sub" ] && (cd "$parent" && find "$name/$sub" -type f) >> "$tmp/data.list"
  done
  sort -o "$tmp/data.list" "$tmp/data.list"
  archive_one "$dest/${name}_data.tar" "$parent" "$tmp/data.list" || rc_all=1

  if [ "$NOSUM" != "1" ]; then
    ( cd "$dest" && for t in "${name}_meta.tar" "${name}_data.tar"; do
        [ -f "$t" ] && sha256sum "$t"
      done > "${name}.sha256" ) && echo "    ${name}.sha256 written"
  fi

  rm -rf "$tmp"
  echo
done

echo "# finished $(date), exit $rc_all"
cat <<NOTE
#
# Next step (from a login node -- Lou needs interactive auth, so it cannot run here):
#     cd ${DEST:-\$WORK/athenak_archive}
#     scp <name>_{meta,data}.tar <you>@lou.nas.nasa.gov:athenak_backups/
# Check integrity on arrival with:  sha256sum -c <name>.sha256
# Restore with:                     tar -C <dest> -xf <name>_data.tar
#
# \$WORK is not purged, but it is quota-limited and NOT backed up -- these are a
# staging area, not the archive.  The \$SCRATCH originals are purged after ~10 days.
NOTE
exit $rc_all
