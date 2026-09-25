#!/usr/bin/env bash
#
# Remove generated output from every case folder, and Python bytecode caches.
#
# A case folder keeps its inputs -- case.toml and the measured record (*.txt)
# -- and everything else in it is generated: dump.npz, bar_identified.npz,
# figures, .dat results. All of it is reproducible with simulate.py,
# identify_bar_*.py and the analysis scripts, run on that folder.
#
# Usage:
#     ./clean.sh           remove the files
#     ./clean.sh -n        dry run: list what would be removed, delete nothing
#
set -euo pipefail

DRY_RUN=0
case "${1:-}" in
    -n|--dry-run) DRY_RUN=1 ;;
    "")           ;;
    -h|--help)    sed -n '3,12p' "$0"; exit 0 ;;
    *)            echo "unknown option: $1 (try -h)" >&2; exit 2 ;;
esac

# Work relative to this script, so it can be called from anywhere.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Generated artefacts: every file in a case folder except its inputs. Listed
# by what is KEPT rather than by name, because the case folders are the only
# place scripts write to -- a stray file of your own belongs elsewhere.
is_input() {
    case "$(basename "$1")" in
        case.toml|*.txt) return 0 ;;
        *)               return 1 ;;
    esac
}

GENERATED=()
while IFS= read -r -d '' f; do
    is_input "$f" || GENERATED+=("${f#"$ROOT/"}")
done < <(find "$ROOT/cases" -type f -print0 | sort -z)

# Outputs older revisions wrote beside the code, before the case folders.
for legacy in dump.npz bar_identified.npz specimen.dat eps.npy force.npy \
              meta.npz meta.npy; do
    [ -e "$ROOT/$legacy" ] && GENERATED+=("$legacy")
done

removed=0

for rel in "${GENERATED[@]+"${GENERATED[@]}"}"; do
    path="$ROOT/$rel"
    [ -e "$path" ] || continue
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "would remove  $rel"
    else
        rm -f -- "$path"
        echo "removed  $rel"
    fi
    removed=$((removed + 1))
done

# Bytecode caches, anywhere under this directory.
while IFS= read -r -d '' cache; do
    rel="${cache#"$ROOT/"}"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "would remove  $rel/"
    else
        rm -rf -- "$cache"
        echo "removed  $rel/"
    fi
    removed=$((removed + 1))
done < <(find "$ROOT" -type d -name '__pycache__' -print0)

if [ "$removed" -eq 0 ]; then
    echo "already clean"
elif [ "$DRY_RUN" -eq 1 ]; then
    echo "-- dry run: $removed item(s) would be removed, nothing deleted"
else
    echo "-- $removed item(s) removed"
fi
