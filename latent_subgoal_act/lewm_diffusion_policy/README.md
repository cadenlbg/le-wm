# LE-WM Diffusion Policy

This is a Diffusion-Policy-style latent action policy for LE-WM datasets.
It follows the official `real-stanford/diffusion_policy` structure more closely than the earlier simplified prior:

```text
LeWMLatentDiffusionPolicy.compute_loss(batch)
ConditionalUnet1D with FiLM global condition
DDPM scheduler with squaredcos_cap_v2-style schedule
EMA with Diffusion Policy warmup schedule
AdamW betas=(0.95, 0.999)
cosine LR scheduler with warmup
checkpoints/best.pt and checkpoints/latest.pt
non-overwritten checkpoints/epoch=XXXX-val_loss=*.pt snapshots
```

The official image observation encoder is replaced by latent condition:

```text
official: obs history -> obs_encoder -> global_cond
ours:     concat(z_{t-1}, z_t, z_g) -> global_cond
```

Defaults now follow the official Push-T setup more closely:

```text
horizon = 16
n_action_steps = 8
history_size = 2
goal_condition = True
```

## Train

Official-scale run:

```bash
CUDA_VISIBLE_DEVICES=4 python -B -m latent_subgoal_act.lewm_diffusion_policy.train \
  dataset=pusht_fixed_g25_k25_t25_ms_128k_train.pt \
  output=lewm_dp_g25_h16_a8_hist2_ms128k \
  horizon=16 \
  n_action_steps=8 \
  history_size=2 \
  training.num_epochs=3050 \
  dataloader.batch_size=64 \
  val_dataloader.batch_size=64 \
  device=cuda
```

Practical short run:

```bash
CUDA_VISIBLE_DEVICES=4 python -B -m latent_subgoal_act.lewm_diffusion_policy.train \
  dataset=pusht_fixed_g25_k25_t25_ms_128k_train.pt \
  output=lewm_dp_g25_h16_a8_hist2_ms128k_short \
  horizon=16 \
  n_action_steps=8 \
  history_size=2 \
  policy.down_dims=[128,256,512] \
  training.num_epochs=100 \
  training.checkpoint_every=100 \
  dataloader.batch_size=256 \
  val_dataloader.batch_size=256 \
  device=cuda
```

Smoke:

```bash
CUDA_VISIBLE_DEVICES=4 python -B -m latent_subgoal_act.lewm_diffusion_policy.train \
  dataset=pusht_fixed_g25_k25_t25_ms_128k_train.pt \
  output=lewm_dp_smoke \
  horizon=16 \
  n_action_steps=8 \
  history_size=2 \
  max_samples=4096 \
  policy.down_dims=[64,128,256] \
  training.num_epochs=2 \
  training.max_train_steps=3 \
  training.max_val_steps=2 \
  training.checkpoint_every=1 \
  dataloader.batch_size=128 \
  val_dataloader.batch_size=128 \
  device=cuda
```

## Eval: Diffusion Rerank

```bash
CUDA_VISIBLE_DEVICES=4 python -B -m latent_subgoal_act.lewm_diffusion_policy.eval \
  policy_ckpt=lewm_dp_g25_h16_a8_hist2_ms128k_short/checkpoints/best.pt \
  eval.num_eval=10 \
  eval.goal_offset_steps=25 \
  eval.eval_budget=50 \
  plan_config.execution_horizon=8 \
  world.num_envs=10 \
  sample.num_candidates=32 \
  cem.enabled=False \
  output.filename=lewm_dp_rerank_n10.txt \
  device=cuda
```

## Eval: Diffusion + CEM

```bash
CUDA_VISIBLE_DEVICES=4 python -B -m latent_subgoal_act.lewm_diffusion_policy.eval \
  policy_ckpt=lewm_dp_g25_h16_a8_hist2_ms128k_short/checkpoints/best.pt \
  eval.num_eval=10 \
  eval.goal_offset_steps=25 \
  eval.eval_budget=50 \
  plan_config.execution_horizon=8 \
  world.num_envs=10 \
  sample.num_candidates=64 \
  cem.enabled=True \
  cem.diffusion_topk=8 \
  cem.num_iters=3 \
  cem.num_candidates=32 \
  cem.elite_frac=0.25 \
  cem.min_std=0.05 \
  cem.std_scale=1.0 \
  output.filename=lewm_dp_diffusion_cem_n10.txt \
  device=cuda
```

## Train-Time WandB Videos

Enable rollout inside training to upload videos like official Diffusion Policy:

```bash
CUDA_VISIBLE_DEVICES=4 python -B -m latent_subgoal_act.lewm_diffusion_policy.train \
  dataset=pusht_fixed_g25_k25_t25_ms_128k_train.pt \
  output=lewm_dp_official_dims_video_smoke \
  horizon=16 \
  n_action_steps=8 \
  history_size=2 \
  policy.down_dims=[512,1024,2048] \
  training.num_epochs=3 \
  training.max_train_steps=20 \
  training.max_val_steps=10 \
  training.rollout_every=1 \
  training.checkpoint_every=1 \
  training.sample_every=1 \
  dataloader.batch_size=16 \
  val_dataloader.batch_size=16 \
  logging.mode=online \
  logging.project=lewm_diffusion_policy \
  logging.name=lewm_dp_official_dims_video_smoke \
  rollout.enabled=True \
  rollout.num_eval=3 \
  rollout.num_vis=3 \
  rollout.sample_num_candidates=4 \
  rollout.execution_horizon=8 \
  device=cuda
```

WandB keys:

```text
test/sim_video_0
test/sim_video_1
test/sim_video_2
```
