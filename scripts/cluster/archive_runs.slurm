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
  free_kb=$(df -Pk "$dest" | tail -1 | awk '{print $4}')
  echo "== $name  ($(du -sh "$run" | cut -f1) in $run)"
  echo "    tars -> $dest   (free there: $(df -Ph "$dest" | tail -1 | awk '{print $4}'))"
  if [ "$free_kb" -lt "$need_kb" ]; then
    echo "    NOT ENOUGH SPACE at $dest: the tars need about as much room as the run." >&2
    echo "    Check your \$WORK quota, or re-run with DEST=<somewhere with room>." >&2
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
