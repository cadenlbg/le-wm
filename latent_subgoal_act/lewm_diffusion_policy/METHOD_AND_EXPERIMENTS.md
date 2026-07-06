# LE-WM Latent Diffusion Policy

## 1. Goal

This experiment adapts official Diffusion Policy Push-T training to LE-WM latent space.

Official Diffusion Policy learns:

```text
recent observations -> future action chunk
```

This version learns:

```text
[z_{t-1}, z_t, z_g] -> future action chunk
```

The default setup follows official Push-T more closely:

```text
horizon = 16
n_action_steps = 8
history_size = 2
goal_condition = True
```

So the policy predicts 16 future actions and executes the first 8 before replanning.

## 2. Dataset

The code uses the existing latent dataset, for example:

```text
pusht_fixed_g25_k25_t25_ms_128k_train.pt
```

Required fields:

```text
z_t
z_g
action
episode
step
metadata
```

Training truncates the stored action chunk:

```text
action_gt = action[:, :16]
```

No rebuild is required for latent history. The dataset wrapper reconstructs:

```text
z_history = [z_{t-1}, z_t]
```

by looking up `(episode, step-1)`. If the previous step is missing, it repeats current `z_t`.

## 3. Model

The denoiser is an official-Diffusion-Policy-style `ConditionalUnet1D`.

Input:

```text
noisy_action: [B, 16, action_dim]
diffusion_step: [B]
global_cond: concat(z_{t-1}, z_t, z_g)
```

If latent dimension is `D`, then:

```text
global_cond_dim = 3D
```

The architecture is:

```text
noisy action sequence
  -> Conv1d / GroupNorm / Mish residual blocks
  -> temporal downsampling
  -> bottleneck
  -> temporal upsampling with skip connections
  -> predicted noise sequence
```

Each residual block receives the same global condition through FiLM:

```text
scale, bias = Linear(condition)
feature = scale * feature + bias
```

This is closer to official Diffusion Policy than the earlier simplified transformer/diffusion prior.

## 4. Loss

The loss is standard epsilon-prediction DDPM loss.

For each batch:

```text
action = normalize(action_gt)
k ~ Uniform(0, 99)
epsilon ~ N(0, I)
noisy_action = sqrt(alpha_bar_k) * action + sqrt(1-alpha_bar_k) * epsilon
epsilon_hat = model(noisy_action, k, z_history, z_g)
loss = MSE(epsilon_hat, epsilon)
```

Training uses:

```text
DDPM train timesteps = 100
beta_schedule = squaredcos_cap_v2
AdamW lr = 1e-4
betas = (0.95, 0.999)
weight_decay = 1e-6
cosine LR schedule
warmup = 500 steps
EMA warmup schedule
```

Checkpoint files:

```text
checkpoints/best.pt                    lowest val_loss checkpoint
checkpoints/latest.pt                  latest periodic checkpoint
checkpoints/epoch=0100-val_loss=*.pt   non-overwritten periodic snapshots
```

## 5. Eval

Evaluation is closed-loop.

At each replanning step:

```text
1. Encode current image with frozen LE-WM -> z_t
2. Encode goal image with frozen LE-WM -> z_g
3. Maintain latent history [z_{t-1}, z_t]
4. Diffusion samples N action chunks of length 16
5. LE-WM rolls out each candidate in latent space
6. Cost = || z_terminal - z_g ||^2
7. Pick the best candidate
8. Execute first 8 actions
9. Re-observe and replan
```

Optional Diffusion + CEM:

```text
diffusion samples candidates
top-k candidates initialize CEM mean/std
CEM refines in action space using LE-WM rollout cost
```

## 6. Recommended Experiments

### 6.1 Smoke With WandB

Use this to verify logging, checkpointing, and data loading.

```bash
CUDA_VISIBLE_DEVICES=4 python -B -m latent_subgoal_act.lewm_diffusion_policy.train \
  dataset=pusht_fixed_g25_k25_t25_ms_128k_train.pt \
  output=lewm_dp_h16_a8_hist2_smoke \
  horizon=16 \
  n_action_steps=8 \
  history_size=2 \
  max_samples=4096 \
  policy.down_dims=[64,128,256] \
  training.num_epochs=5 \
  training.max_train_steps=20 \
  training.max_val_steps=10 \
  training.checkpoint_every=1 \
  dataloader.batch_size=128 \
  val_dataloader.batch_size=128 \
  logging.mode=online \
  logging.project=lewm_diffusion_policy \
  logging.name=lewm_dp_h16_a8_hist2_smoke \
  device=cuda
```

Smoke eval:

```bash
CUDA_VISIBLE_DEVICES=4 python -B -m latent_subgoal_act.lewm_diffusion_policy.eval \
  policy_ckpt=lewm_dp_h16_a8_hist2_smoke/checkpoints/best.pt \
  eval.num_eval=3 \
  eval.goal_offset_steps=25 \
  eval.eval_budget=50 \
  plan_config.execution_horizon=8 \
  world.num_envs=3 \
  sample.num_candidates=8 \
  cem.enabled=False \
  output.filename=lewm_dp_smoke_eval.txt \
  device=cuda
```

### 6.2 Practical Training

This is the first serious run. It keeps the official DP horizon/action-step logic but uses a smaller U-Net.

Recommended epochs:

```text
200 epochs first
```

Reason: official DP trains much longer, but our dataset has 128k samples and larger batch size. Start with 200 to check whether train/val loss both decrease; extend to 500 if val loss is still improving.

```bash
CUDA_VISIBLE_DEVICES=4 python -B -m latent_subgoal_act.lewm_diffusion_policy.train \
  dataset=pusht_fixed_g25_k25_t25_ms_128k_train.pt \
  output=lewm_dp_g25_h16_a8_hist2_ms128k_200ep \
  horizon=16 \
  n_action_steps=8 \
  history_size=2 \
  policy.down_dims=[128,256,512] \
  training.num_epochs=200 \
  training.checkpoint_every=100 \
  dataloader.batch_size=256 \
  val_dataloader.batch_size=256 \
  logging.mode=online \
  logging.project=lewm_diffusion_policy \
  logging.name=lewm_dp_g25_h16_a8_hist2_ms128k_200ep \
  device=cuda
```

Eval rerank:

```bash
CUDA_VISIBLE_DEVICES=4 python -B -m latent_subgoal_act.lewm_diffusion_policy.eval \
  policy_ckpt=lewm_dp_g25_h16_a8_hist2_ms128k_200ep/checkpoints/best.pt \
  eval.num_eval=100 \
  eval.goal_offset_steps=25 \
  eval.eval_budget=50 \
  plan_config.execution_horizon=8 \
  world.num_envs=100 \
  sample.num_candidates=32 \
  cem.enabled=False \
  output.filename=lewm_dp_rerank_n100_c32_exec8.txt \
  device=cuda
```

Eval Diffusion + CEM:

```bash
CUDA_VISIBLE_DEVICES=4 python -B -m latent_subgoal_act.lewm_diffusion_policy.eval \
  policy_ckpt=lewm_dp_g25_h16_a8_hist2_ms128k_200ep/checkpoints/best.pt \
  eval.num_eval=100 \
  eval.goal_offset_steps=25 \
  eval.eval_budget=50 \
  plan_config.execution_horizon=8 \
  world.num_envs=100 \
  sample.num_candidates=64 \
  cem.enabled=True \
  cem.diffusion_topk=8 \
  cem.num_iters=3 \
  cem.num_candidates=32 \
  cem.elite_frac=0.25 \
  cem.min_std=0.05 \
  cem.std_scale=1.0 \
  output.filename=lewm_dp_diffusion_cem_n100_c64_exec8.txt \
  device=cuda
```

### 6.3 Longer Training

If validation loss is still decreasing at 200 epochs, continue with a 500 epoch run:

```bash
CUDA_VISIBLE_DEVICES=4 python -B -m latent_subgoal_act.lewm_diffusion_policy.train \
  dataset=pusht_fixed_g25_k25_t25_ms_128k_train.pt \
  output=lewm_dp_g25_h16_a8_hist2_ms128k_500ep \
  horizon=16 \
  n_action_steps=8 \
  history_size=2 \
  policy.down_dims=[128,256,512] \
  training.num_epochs=500 \
  training.checkpoint_every=100 \
  dataloader.batch_size=256 \
  val_dataloader.batch_size=256 \
  logging.mode=online \
  logging.project=lewm_diffusion_policy \
  logging.name=lewm_dp_g25_h16_a8_hist2_ms128k_500ep \
  device=cuda
```

## 7. What To Watch

In WandB, watch:

```text
train_loss
val_loss
lr
epoch
global_step
```

Healthy behavior:

```text
train_loss decreases
val_loss decreases or plateaus slowly
best.pt updates during the run
```

Warning signs:

```text
train_loss decreases but val_loss rises early -> overfitting or condition mismatch
val_loss stays near 1.0 -> denoiser is not learning much
eval improves with CEM but not rerank -> diffusion samples are useful but not precise enough
```
