# DP Latent Prior

This folder implements a Diffusion-Policy-style latent action prior:

```text
p(action_chunk | z_t, z_g)
```

It keeps more Diffusion Policy details than `latent_subgoal_act/action_priors`:

```text
Temporal U-Net denoiser
global conditioning injected into every residual block by FiLM
DDPM cosine noise schedule
epsilon prediction loss
EMA checkpoint for eval
separate prediction_horizon and execution_horizon
optional action normalizer
```

The dataset action is already standardized by `build_dataset.py`, so `normalizer.enabled=False` is the default to avoid double normalization.

## Train

```bash
CUDA_VISIBLE_DEVICES=2 python -B -m latent_subgoal_act.dp_latent_prior.train \
  dataset=pusht_fixed_g25_k25_t25_ms_128k_train.pt \
  output=dp_latent_prior_g25_T25_ms128k \
  prediction_horizon=25 \
  train.epochs=400 \
  loader.batch_size=256 \
  device=cuda
```

Smaller smoke:

```bash
CUDA_VISIBLE_DEVICES=2 python -B -m latent_subgoal_act.dp_latent_prior.train \
  dataset=pusht_fixed_g25_k25_t25_ms_128k_train.pt \
  output=dp_latent_prior_g25_T25_smoke \
  prediction_horizon=25 \
  max_samples=4096 \
  train.epochs=5 \
  loader.batch_size=128 \
  device=cuda
```

## Eval: Diffusion Rerank

```bash
CUDA_VISIBLE_DEVICES=2 python -B -m latent_subgoal_act.dp_latent_prior.eval \
  policy_ckpt=dp_latent_prior_g25_T25_ms128k/policy.pt \
  eval.num_eval=10 \
  eval.goal_offset_steps=25 \
  eval.eval_budget=50 \
  plan_config.execution_horizon=1 \
  world.num_envs=10 \
  sample.num_candidates=32 \
  cem.enabled=False \
  device=cuda
```

## Eval: Diffusion + CEM

```bash
CUDA_VISIBLE_DEVICES=2 python -B -m latent_subgoal_act.dp_latent_prior.eval \
  policy_ckpt=dp_latent_prior_g25_T25_ms128k/policy.pt \
  eval.num_eval=10 \
  eval.goal_offset_steps=25 \
  eval.eval_budget=50 \
  plan_config.execution_horizon=1 \
  world.num_envs=10 \
  sample.num_candidates=64 \
  cem.enabled=True \
  cem.diffusion_topk=8 \
  cem.num_iters=3 \
  cem.num_candidates=32 \
  cem.elite_frac=0.25 \
  cem.min_std=0.05 \
  cem.std_scale=1.0 \
  device=cuda
```

