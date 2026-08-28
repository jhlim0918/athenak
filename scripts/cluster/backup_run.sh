#!/bin/bash
# Archive one AthenaK run directory to Lou (NAS mass storage).
#
# Usage:  scripts/cluster/backup_run.sh <rundir> [lou_dir]
#         lou_dir defaults to athenak_backups (i.e. lou:~/athenak_backups)
#
# Two archives per run, made with shiftc (NAS's transfer tool: streams the tar
# straight to Lou -- no local copy -- with checksums and automatic retries):
#   <name>_data.tar     bin/ and rst/          (the bulk)
#   <name>_meta.tar     everything else: .hst, run.log, PBS scripts/output,
#                       provenance, frozen input  (small; grab this one first
#                       when you only need the history)
# Check progress/integrity afterwards with:  shiftc --status
#
# Restore:  shiftc --extract-tar lou:<lou_dir>/<name>_data.tar <dest>

set -eu

RUNDIR=$(cd "${1:?usage: backup_run.sh <rundir> [lou_dir]}" && pwd)
LOUDIR=${2:-athenak_backups}
NAME=$(basename "$RUNDIR")

command -v shiftc >/dev/null || {
  echo "shiftc not found -- on NAS load it with: module load shift" >&2; exit 1; }

cd "$RUNDIR"

# the bulk: bin/ and rst/ (whichever exist)
DATA=""
[ -d bin ] && DATA="bin"
[ -d rst ] && DATA="$DATA rst"
if [ -n "$DATA" ]; then
  echo "archiving $DATA -> lou:$LOUDIR/${NAME}_data.tar"
  shiftc --create-tar $DATA "lou:$LOUDIR/${NAME}_data.tar"
fi

# the metadata: every regular file at the top level (hst, logs, pbs, provenance)
META=$(find . -maxdepth 1 -type f | sed 's|^\./||')
if [ -n "$META" ]; then
  echo "archiving metadata -> lou:$LOUDIR/${NAME}_meta.tar"
  shiftc --create-tar $META "lou:$LOUDIR/${NAME}_meta.tar"
fi

echo "queued; monitor with: shiftc --status"
