# Latent Subgoal ACT Guide

## Idea

This variant removes `delta_z` and decomposes control into:

```text
z_t, z_g -> z_hat_{t+5}
z_t, z_g, z_hat_{t+5} -> action chunk a_t ... a_{t+4}
```

At execution time it still uses receding horizon control:

```text
observe -> encode z_t/z_g -> predict subgoal/action chunk -> execute 1 action -> observe again
```

## Dataset

The dataset must contain a true short-horizon target:

```text
z_t
z_g
z_h = z_{min(t+5, goal_step)}
action
action_raw
episode
step
subgoal_step
goal_step
metadata
```

If `t+5` would pass the goal step, `z_h` is set to `z_g` by default.

Build a leakage-aware training dataset with episode-level train/test split:

```bash
python -m latent_subgoal_act.build_dataset \
  output_dataset=pusht_g25_k5_h5_train128k.pt \
  max_samples=128000 \
  split=train \
  split_seed=42 \
  test_fraction=0.1 \
  lewm_policy=pusht/lewm \
  device=cuda
```

To generate many samples for the same goal anchor, use `sample_mode=goal_anchored`:

```bash
python -m latent_subgoal_act.build_dataset \
  output_dataset=pusht_g25_k5_h5_anchor_train128k.pt \
  max_samples=128000 \
  sample_mode=goal_anchored \
  goal_stride=25 \
  split=train \
  split_seed=42 \
  test_fraction=0.1 \
  lewm_policy=pusht/lewm \
  device=cuda
```

For one anchor goal `z_g=z25`, this creates examples like:

```text
(z1, z25 -> z6)
(z2, z25 -> z7)
...
(z24, z25 -> z25)
```

Evaluation defaults to the held-out `test` episode split with the same seed.

Inspect the dataset before training:

```bash
python -m latent_subgoal_act.inspect_dataset \
  dataset=pusht_g25_k5_h5_train128k.pt \
  expected_split=train \
  split_seed=42 \
  test_fraction=0.1
```

## Loss

Base loss:

```text
L = L_action + lambda_subgoal * L_subgoal + lambda_smooth * L_smooth
```

where:

```text
L_action  = MSE(action_pred, action_expert)
L_subgoal = MSE(z_hat_{t+5}, z_true_{t+5})
```

Optional frozen LeWM predictor loss:

```text
z_rollout = frozen_LeWM(z_t, action_pred)

L += lambda_rollout * MSE(z_rollout, z_true_{t+5})
L += lambda_align   * MSE(z_rollout, z_hat_{t+5})
```

This uses the pretrained LeWM predictor as a latent dynamics verifier. LeWM parameters are frozen, but gradients can flow through `action_pred`, so the action head is encouraged to output chunks whose predicted consequence matches the short-horizon subgoal.

## Recommended First Runs

Base subgoal ACT:

```bash
python -m latent_subgoal_act.train \
  dataset=pusht_g25_k5_h5_train128k.pt \
  output=subgoal_act_train128k \
  max_samples=128000 \
  seed=42 \
  train.epochs=100 \
  loss.lambda_subgoal=1.0
```

With frozen LeWM rollout consistency:

```bash
python -m latent_subgoal_act.train \
  dataset=pusht_g25_k5_h5_train128k.pt \
  output=subgoal_act_train128k_wm01 \
  max_samples=128000 \
  seed=42 \
  train.epochs=100 \
  loss.lambda_subgoal=1.0 \
  wm.enabled=true \
  wm.policy=pusht/lewm \
  wm.history_size=1 \
  wm.lambda_rollout=0.1 \
  wm.lambda_align=0.1
```

Evaluation:

```bash
python -m latent_subgoal_act.eval \
  policy_ckpt=/data/zflin/lewm_re/experiments/subgoal_act_train128k_wm01/policy.pt \
  lewm_policy=pusht/lewm \
  device=cuda \
  eval.split=test \
  eval.split_seed=42 \
  eval.test_fraction=0.1
```

Archive an experiment after evaluation:

```bash
bash scripts/archive_subgoal_act_experiment.sh \
  subgoal_act_train128k_wm01 \
  "python -m latent_subgoal_act.eval policy_ckpt=/data/zflin/lewm_re/experiments/subgoal_act_train128k_wm01/policy.pt eval.split=test"
```

## Optional Inference Reranking

At evaluation time, local LeWM reranking is enabled by default. The policy generates noisy variants around its predicted action chunk and uses the frozen LeWM predictor to rerank them:

```text
base action chunk -> noisy candidates
z_t, candidate chunk -> frozen LeWM rollout -> z_candidate
choose candidate closest to z_hat_{t+5} or z_g
```

This is much cheaper than full CEM because candidates are local perturbations around the ACT chunk instead of broad random search.

```bash
python -m latent_subgoal_act.eval \
  policy_ckpt=/data/zflin/lewm_re/experiments/subgoal_act_train128k_wm01/policy.pt \
  lewm_policy=pusht/lewm \
  device=cuda \
  eval.split=test \
  eval.split_seed=42 \
  eval.test_fraction=0.1 \
  rerank.enabled=true \
  rerank.num_candidates=16 \
  rerank.noise_std=0.2 \
  rerank.target=subgoal
```

Disable reranking for an ablation:

```bash
python -m latent_subgoal_act.eval \
  policy_ckpt=/data/zflin/lewm_re/experiments/subgoal_act_train128k_wm01/policy.pt \
  lewm_policy=pusht/lewm \
  device=cuda \
  eval.split=test \
  rerank.enabled=false
```

## Optional CEM To Predicted Subgoal

You can also use CEM at inference. Unlike the original LeWM CEM that directly minimizes distance to the final goal latent, this variant minimizes the distance between the LeWM-predicted fifth-step latent and the policy-predicted short subgoal:

```text
subgoal ACT: z_t, z_g -> z_hat_{t+5}
CEM samples action chunks a_{t:t+4}
LeWM rollout: z_t, a_{t:t+4} -> z_rollout_{t+5}
objective: minimize ||z_rollout_{t+5} - z_hat_{t+5}||^2
```

Run CEM:

```bash
python -m latent_subgoal_act.eval \
  policy_ckpt=/data/zflin/lewm_re/experiments/subgoal_act_train128k_wm01/policy.pt \
  lewm_policy=pusht/lewm \
  device=cuda \
  eval.split=test \
  eval.split_seed=42 \
  eval.test_fraction=0.1 \
  cem.enabled=true \
  cem.num_iters=3 \
  cem.num_candidates=64 \
  cem.elite_frac=0.1 \
  cem.init_std=0.5 \
  cem.min_std=0.05 \
  rerank.target=subgoal
```

CEM uses the ACT chunk as the initial mean, so it is closer to policy-guided MPC than pure random CEM.

For the main experiment, run the same trained checkpoint with three evaluation modes and compare the result files in the experiment folder:

1. Direct policy only:

```bash
python -m latent_subgoal_act.eval \
  policy_ckpt=/data/zflin/lewm_re/experiments/subgoal_act_anchor_train128k_wm01/policy.pt \
  lewm_policy=pusht/lewm \
  device=cuda \
  eval.split=test \
  eval.split_seed=42 \
  eval.test_fraction=0.1 \
  rerank.enabled=false \
  cem.enabled=false
```

2. Local LeWM rerank to predicted subgoal:

```bash
python -m latent_subgoal_act.eval \
  policy_ckpt=/data/zflin/lewm_re/experiments/subgoal_act_anchor_train128k_wm01/policy.pt \
  lewm_policy=pusht/lewm \
  device=cuda \
  eval.split=test \
  eval.split_seed=42 \
  eval.test_fraction=0.1 \
  rerank.enabled=true \
  rerank.num_candidates=16 \
  rerank.noise_std=0.2 \
  rerank.target=subgoal \
  cem.enabled=false
```

3. CEM to predicted subgoal:

```bash
python -m latent_subgoal_act.eval \
  policy_ckpt=/data/zflin/lewm_re/experiments/subgoal_act_anchor_train128k_wm01/policy.pt \
  lewm_policy=pusht/lewm \
  device=cuda \
  eval.split=test \
  eval.split_seed=42 \
  eval.test_fraction=0.1 \
  rerank.target=subgoal \
  cem.enabled=true \
  cem.num_iters=3 \
  cem.num_candidates=64 \
  cem.elite_frac=0.1 \
  cem.init_std=0.5 \
  cem.min_std=0.05
```

By default, these write separate files:

```text
pusht_direct_results.txt
pusht_rerank_to_subgoal_results.txt
pusht_cem_to_subgoal_results.txt
```

If CEM is too slow, first reduce `cem.num_candidates=32`. If it is stable but weak, try increasing `cem.num_iters=5` or `cem.num_candidates=128`.

## Optional Temporal Ensemble

ACT-style temporal ensemble averages the action predicted for the current time from several recent chunks:

```text
current chunk:  a_t[0]
previous chunk: a_{t-1}[1]
older chunk:    a_{t-2}[2]
...
```

This can reduce action jitter when using receding-horizon execution. It is especially useful when the policy predicts `K=5` actions but execution only uses one step before replanning.

```bash
python -m latent_subgoal_act.eval \
  policy_ckpt=/data/zflin/lewm_re/experiments/subgoal_act_train128k_wm01/policy.pt \
  lewm_policy=pusht/lewm \
  device=cuda \
  eval.split=test \
  eval.split_seed=42 \
  eval.test_fraction=0.1 \
  temporal_ensemble.enabled=true \
  temporal_ensemble.decay=0.01
```

Temporal ensemble can also be combined with reranking:

```bash
python -m latent_subgoal_act.eval \
  policy_ckpt=/data/zflin/lewm_re/experiments/subgoal_act_train128k_wm01/policy.pt \
  lewm_policy=pusht/lewm \
  device=cuda \
  eval.split=test \
  eval.split_seed=42 \
  eval.test_fraction=0.1 \
  rerank.enabled=true \
  rerank.num_candidates=16 \
  rerank.noise_std=0.2 \
  rerank.target=subgoal \
  temporal_ensemble.enabled=true \
  temporal_ensemble.decay=0.01
```

## What To Add Next

If the base model still fails, the most useful next additions are:

1. Train base subgoal ACT and verify `subgoal_mse` improves.
2. Add frozen LeWM consistency if action chunks do not move toward the subgoal.
3. Add temporal ensemble if rollouts look jittery.
4. Add reranking if the policy is close but picks unstable local actions.
5. Full CEM distillation if reranking helps but local noise is not enough.
