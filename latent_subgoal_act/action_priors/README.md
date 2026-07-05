# Action Priors

这个文件夹放两个独立 baseline，不改 `latent_subgoal_act` 主线代码。

## 1. Deterministic Gaussian Action Prior

目标：

```text
z_t, z_g -> action_mean
```

训练时用专家动作做 MSE：

```text
L = MSE(action_mean, action_gt)
```

评估时把 `action_mean` 当作高斯搜索分布的均值：

```text
candidate = action_mean + noise * std
```

然后用 LEWM rollout 评估：

```text
cost = || LEWM(z_t, candidate)_terminal - z_g ||^2
```

支持三种模式：

```text
direct: 直接执行 action_mean
rerank: 在 action_mean 附近采样候选，一轮筛选
CEM:    以 action_mean 为初始 mean，多轮采样/elite/update
```

训练：

```bash
python -B -m latent_subgoal_act.action_priors.train_deterministic \
  dataset=pusht_fixed_g25_k25_t25_ms_128k_train.pt \
  output=goal_action_prior_g25_K5_ms128k \
  action_horizon=5 \
  train.epochs=200 \
  loader.batch_size=256
```

rerank eval：

```bash
python -B -m latent_subgoal_act.action_priors.eval_deterministic \
  policy_ckpt=goal_action_prior_g25_K5_ms128k/policy.pt \
  eval.num_eval=10 \
  eval.goal_offset_steps=25 \
  eval.eval_budget=50 \
  plan_config.receding_horizon=1 \
  world.num_envs=10 \
  rerank.enabled=True \
  rerank.num_candidates=32 \
  rerank.noise_std=1.0 \
  cem.enabled=False
```

CEM eval：

```bash
python -B -m latent_subgoal_act.action_priors.eval_deterministic \
  policy_ckpt=goal_action_prior_g25_K5_ms128k/policy.pt \
  eval.num_eval=10 \
  eval.goal_offset_steps=25 \
  eval.eval_budget=50 \
  plan_config.receding_horizon=1 \
  world.num_envs=10 \
  rerank.enabled=False \
  cem.enabled=True \
  cem.num_iters=3 \
  cem.num_candidates=32 \
  cem.elite_frac=0.25 \
  cem.init_std=1.0 \
  cem.min_std=0.05
```

## 2. Simplified Diffusion Action Prior

目标：

```text
p(action_chunk | z_t, z_g)
```

第一版只 condition `z_t,z_g`，不使用 `z_h_seq`。

训练时加噪动作并预测噪声：

```text
noisy_action = sqrt(alpha_t) * action + sqrt(1-alpha_t) * noise
denoiser(noisy_action, t, z_t, z_g) -> pred_noise
L = MSE(pred_noise, noise)
```

评估时：

```text
1. diffusion sample 多个 action chunks
2. LEWM rollout 每个 candidate
3. 选择 terminal latent 最接近 z_g 的 action
4. 执行 receding_horizon 步
```

训练：

```bash
python -B -m latent_subgoal_act.action_priors.train_diffusion \
  dataset=pusht_fixed_g25_k25_t25_ms_128k_train.pt \
  output=goal_diffusion_prior_g25_K5_ms128k \
  action_horizon=5 \
  diffusion.num_steps=50 \
  train.epochs=200 \
  loader.batch_size=256
```

eval：

```bash
python -B -m latent_subgoal_act.action_priors.eval_diffusion \
  policy_ckpt=goal_diffusion_prior_g25_K5_ms128k/policy.pt \
  eval.num_eval=10 \
  eval.goal_offset_steps=25 \
  eval.eval_budget=50 \
  plan_config.receding_horizon=1 \
  world.num_envs=10 \
  sample.num_candidates=32
```

Diffusion + CEM eval：

```bash
python -B -m latent_subgoal_act.action_priors.eval_diffusion_cem \
  policy_ckpt=goal_diffusion_prior_g25_K5_ms128k/policy.pt \
  eval.num_eval=10 \
  eval.goal_offset_steps=25 \
  eval.eval_budget=50 \
  plan_config.receding_horizon=1 \
  world.num_envs=10 \
  diffusion.num_candidates=64 \
  diffusion.topk=8 \
  cem.num_iters=3 \
  cem.num_candidates=32 \
  cem.elite_frac=0.25 \
  cem.min_std=0.05 \
  cem.std_scale=1.0
```

Diffusion + CEM 的流程：

```text
1. diffusion 先采样 diffusion.num_candidates 条 action chunks
2. LEWM rollout 到 z_g，选 cost 最低的 diffusion.topk 条
3. 用 top-k 的 mean/std 初始化 CEM
4. CEM 再迭代采样、筛 elite、更新 mean/std
5. 执行最终 best action chunk 的前 receding_horizon 步
```

## 对比意义

这两个 baseline 都测试同一个问题：

```text
只用 z_t,z_g 作为 action conditioning，能不能比 z_hat_seq-conditioned policy 更快、更少蠕动？
```

区别：

```text
Deterministic prior:
  学一个 action mean，再用 Gaussian noise / CEM 探索。

Diffusion prior:
  直接学习 action chunk 分布，多样化采样后用 LEWM rerank。
```
