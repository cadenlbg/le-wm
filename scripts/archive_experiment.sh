#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <experiment_dir_name> <command...>" >&2
  exit 1
fi

exp_name="$1"
shift

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
experiments_root="${EXPERIMENTS_DIR:-$(dirname "$root_dir")/experiments}"
exp_dir="$experiments_root/$exp_name"
art_dir="$exp_dir/artifacts"
hydra_dir="$exp_dir/hydra_outputs"

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

if [ -d "$root_dir/outputs" ]; then
  mkdir -p "$hydra_dir"
  find "$root_dir/outputs" -mindepth 1 -maxdepth 1 -exec mv -t "$hydra_dir" {} +
  rmdir "$root_dir/outputs" 2>/dev/null || true
fi

find "$root_dir" -maxdepth 1 -name '*.mp4' -exec mv -t "$art_dir" {} +

stablewm_home="${STABLEWM_HOME:-}"
if [ -n "$stablewm_home" ]; then
  if [ -d "$stablewm_home/pusht" ]; then
    find "$stablewm_home/pusht" -maxdepth 1 -name '*.mp4' -exec mv -t "$art_dir" {} +
  fi
  if [ -d "$stablewm_home/tworoom" ]; then
    find "$stablewm_home/tworoom" -maxdepth 1 -name '*.mp4' -exec mv -t "$art_dir" {} +
  fi
fi

if [ -f "$root_dir/config/eval/pusht.yaml" ] && [[ "$exp_name" == *pusht* ]]; then
  cp "$root_dir/config/eval/pusht.yaml" "$exp_dir/config.yaml"
fi

if [ -f "$root_dir/config/eval/tworoom.yaml" ] && [[ "$exp_name" == *tworoom* ]]; then
  cp "$root_dir/config/eval/tworoom.yaml" "$exp_dir/config.yaml"
fi

echo "Archived to: $exp_dir"
