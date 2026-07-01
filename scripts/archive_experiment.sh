#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <experiment_dir_name> <command...>" >&2
  exit 1
fi

exp_name="$1"
shift

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exp_dir="$root_dir/experiments/$exp_name"
art_dir="$exp_dir/artifacts"

mkdir -p "$art_dir"

{
  printf '%s\n' "# Experiment: $exp_name"
  printf '\n%s\n' "## Command"
  printf '```bash\n%s\n```\n' "$*"
  printf '\n%s\n' "## Timestamp"
  date
} > "$exp_dir/README.md"

printf '%s\n' "$*" > "$exp_dir/command.txt"

if [ -f "$root_dir/pusht_results.txt" ]; then
  mv "$root_dir/pusht_results.txt" "$exp_dir/"
fi

if [ -f "$root_dir/tworoom_results.txt" ]; then
  mv "$root_dir/tworoom_results.txt" "$exp_dir/"
fi

find "$root_dir" -maxdepth 1 -name '*.mp4' -exec mv -t "$art_dir" {} +

if [ -f "$root_dir/config/eval/pusht.yaml" ] && [[ "$exp_name" == *pusht* ]]; then
  cp "$root_dir/config/eval/pusht.yaml" "$exp_dir/config.yaml"
fi

if [ -f "$root_dir/config/eval/tworoom.yaml" ] && [[ "$exp_name" == *tworoom* ]]; then
  cp "$root_dir/config/eval/tworoom.yaml" "$exp_dir/config.yaml"
fi

echo "Archived to: $exp_dir"