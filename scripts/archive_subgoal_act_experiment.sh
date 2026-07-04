#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <experiment_dir_name> <command...>" >&2
  echo "Example: $0 subgoal_act_train128k_wm01 \"python -m latent_subgoal_act.eval policy_ckpt=/data/zflin/lewm_re/experiments/subgoal_act_train128k_wm01/policy.pt\"" >&2
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
notes_dir="$exp_dir/notes"

mkdir -p "$exp_dir" "$art_dir" "$notes_dir"

{
  printf '%s\n' "# Latent Subgoal ACT Experiment: $exp_name"
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
} > "$exp_dir/README.md"

printf '%s\n' "$*" > "$exp_dir/command.txt"

if [ -f "$root_dir/config/eval/pusht.yaml" ]; then
  cp "$root_dir/config/eval/pusht.yaml" "$exp_dir/eval_pusht.yaml"
fi

for note in \
  latent_subgoal_act_guide.md \
  lewm_latent_act_planbook.md \
  lewm_act_downstream_experiment_plan.md
do
  if [ -f "$root_dir/notes/$note" ]; then
    cp "$root_dir/notes/$note" "$notes_dir/"
  fi
done

find "$exp_dir" -maxdepth 1 -type f -name '*.mp4' -exec mv -t "$art_dir" {} + 2>/dev/null || true

{
  printf '%s\n' "experiment_dir=$exp_dir"
  printf '%s\n' "policy=$(test -f "$exp_dir/policy.pt" && echo present || echo missing)"
  printf '%s\n' "config=$(test -f "$exp_dir/config.yaml" && echo present || echo missing)"
  printf '%s\n' "metrics=$(test -f "$exp_dir/metrics.jsonl" && echo present || echo missing)"
  printf '%s\n' "results=$(test -f "$exp_dir/pusht_results.txt" && echo present || echo missing)"
  printf '%s\n' "videos=$(find "$art_dir" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
} > "$exp_dir/manifest.txt"

echo "Archived latent subgoal ACT experiment to: $exp_dir"

