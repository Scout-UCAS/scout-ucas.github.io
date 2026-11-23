#!/usr/bin/env bash
# Run from the project root in the activated Python environment.
set -euo pipefail
config_path="${1:-configs/sd35.json}"
study_dir="${2:-runs/study}"
dataset_dir="${3:-data}"
cache_dir="${4:-cache/refs}"
if [[ ! -f "$dataset_dir/dataset.json" ]]; then
    python -m responseguard data --out "$dataset_dir"
fi
python -m responseguard study --config "$config_path" --out "$study_dir" --data "$dataset_dir" --cache "$cache_dir" "${@:5}"
