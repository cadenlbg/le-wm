#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <experiment_dir_name> <command...>" >&2
  echo "Example: $0 2026-07-03_pusht_latent_bc \"python eval_latent_bc.py policy_ckpt=2026-07-03_pusht_latent_bc/policy.pt eval.num_eval=50\"" >&2
  exit 1
fi

exp_name="$1"
shift

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "${LEWM_EXPERIMENTS_DIR:-}" ]; then
  experiments_root="$LEWM_EXPERIMENTS_DIR"
elif [ -n "${EXPERIMENTS_DIR:-}" ]; then
  experiments_root="$EXPERIMENTS_DIR"
else
  experiments_root="$(dirname "$root_dir")/experiments"
fi

exp_dir="$experiments_root/$exp_name"
art_dir="$exp_dir/artifacts"
hydra_out="$exp_dir/hydra_outputs"
notes_dir="$exp_dir/notes"

mkdir -p "$exp_dir" "$art_dir" "$hydra_out" "$notes_dir"

{
  printf '%s\n' "# Latent BC Experiment: $exp_name"
  printf '\n%s\n' "## Command"
  printf '```bash\n%s\n```\n' "$*"
  printf '\n%s\n' "## Timestamp"
  date
  printf '\n%s\n' "## Paths"
  printf '%s\n' "- repository: $root_dir"
  printf '%s\n' "- experiments_root: $experiments_root"
  printf '%s\n' "- experiment_dir: $exp_dir"
  printf '\n%s\n' "## Expected Artifacts"
  printf '%s\n' "- policy.pt"
  printf '%s\n' "- config.yaml"
  printf '%s\n' "- metrics.jsonl"
  printf '%s\n' "- pusht_results.txt"
  printf '%s\n' "- artifacts/*.mp4"
  printf '%s\n' "- hydra_outputs/"
} > "$exp_dir/README.md"

printf '%s\n' "$*" > "$exp_dir/command.txt"

if [ -f "$root_dir/config/eval/pusht.yaml" ]; then
  cp "$root_dir/config/eval/pusht.yaml" "$exp_dir/eval_pusht.yaml"
fi

for note in \
  latent_bc_experiment_guide.md \
  latent_goal_conditioned_bc_plan.md \
  latent_bc_experiment_logic.md
do
  if [ -f "$root_dir/notes/$note" ]; then
    cp "$root_dir/notes/$note" "$notes_dir/"
  fi
done

find "$exp_dir" -maxdepth 1 -type f -name '*.mp4' -exec mv -t "$art_dir" {} + 2>/dev/null || true

copy_latest_hydra_run() {
  local job_name="$1"
  local job_root="$experiments_root/hydra/$job_name"
  local latest

  if [ ! -d "$job_root" ]; then
    return 0
  fi

  latest="$(find "$job_root" -mindepth 2 -maxdepth 2 -type d -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -n 1 | cut -d' ' -f2-)"
  if [ -n "$latest" ] && [ -d "$latest" ]; then
    mkdir -p "$hydra_out/$job_name"
    cp -a "$latest" "$hydra_out/$job_name/"
  fi
}

if [ -n "${HYDRA_RUNS:-}" ]; then
  IFS=':' read -r -a hydra_runs <<< "$HYDRA_RUNS"
  for run_dir in "${hydra_runs[@]}"; do
    if [ -d "$run_dir" ]; then
      cp -a "$run_dir" "$hydra_out/"
    fi
  done
else
  copy_latest_hydra_run build_latent_bc_dataset
  copy_latest_hydra_run train_latent_bc
  copy_latest_hydra_run eval_latent_bc
fi

{
  printf '%s\n' "experiment_dir=$exp_dir"
  printf '%s\n' "policy=$(test -f "$exp_dir/policy.pt" && echo present || echo missing)"
  printf '%s\n' "metrics=$(test -f "$exp_dir/metrics.jsonl" && echo present || echo missing)"
  printf '%s\n' "results=$(test -f "$exp_dir/pusht_results.txt" && echo present || echo missing)"
  printf '%s\n' "videos=$(find "$art_dir" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
  printf '%s\n' "hydra_runs=$(find "$hydra_out" -mindepth 1 -type d | wc -l)"
} > "$exp_dir/manifest.txt"

echo "Archived latent BC experiment to: $exp_dir"
