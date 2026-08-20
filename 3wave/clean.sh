#!/usr/bin/env bash
#
# Remove generated simulation output and Python bytecode caches.
#
# Everything deleted here is reproducible with:
#     python3 drive_compression.py && python3 reduce_specimen.py
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
    -h|--help)    sed -n '3,11p' "$0"; exit 0 ;;
    *)            echo "unknown option: $1 (try -h)" >&2; exit 2 ;;
esac

# Work relative to this script, so it can be called from anywhere.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Generated artefacts, listed explicitly rather than by wildcard so that a
# stray *.npy or *.png of your own is never caught by accident.
GENERATED=(
    # recorded output written by drive_compression.py / drive_tension.py / drive_calibration_tension.py
    "dump.npz"
    # ground truth written by simulate_compression.py
    "specimen.dat"
    # superseded by dump.npz; listed so old runs get cleaned up too
    "eps.npy"
    "force.npy"
    "meta.npz"
    "meta.npy"
    # results written by reduce_specimen.py
    "specimen_reconstructed.dat"
    "specimen_reconstruction.png"
    # figure written by plot_forces.py
    "gauge_forces.png"
    # figure written by identify_bar_tension.py
    "bar_identification_tension.png"
    # figure written by identify_bar_compression.py
    "bar_identification_compression.png"
    # same script run on a MEASURED shot: --experiment <case>
    "bar_identification_experiment_pc_bar.png"
    # the identified numbers, handed from identify_bar_compression.py to
    # reconstruct_interface.py the way dump.npz is handed from a driver
    "bar_identified.npz"
    # written by reconstruct_interface.py, with and without --no-attenuation
    "interface_force.png"
    "interface_force.dat"
    "interface_force_lossless.png"
    "interface_force_lossless.dat"
    # same script with --case <another shot>, e.g. the specimen record
    "interface_force_experiment_pc_specimen.png"
    "interface_force_experiment_pc_specimen.dat"
    "interface_force_experiment_pc_specimen_lossless.png"
    "interface_force_experiment_pc_specimen_lossless.dat"
    # figure written by lagrange_diagram.py
    "lagrange_diagram.png"
    # legacy outputs of the unmodified simulate_compression.py, in case an older copy is run
    "eps_vel.dat"
    "u_vel.dat"
    "linescan_analysis.dat"
    "Symmpact_time_force.txt"
)

removed=0

for rel in "${GENERATED[@]}"; do
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
