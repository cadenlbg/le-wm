# Latent Subgoal ACT 实验日志

日志创建时间：2026-07-05 16:59:49 +08:00

## 快速跳转

- [实验目标](#实验目标)
- [1. 生成 128k 数据集](#1-生成-128k-数据集)
- [2. 检查数据集](#2-检查数据集)
- [3. K=5, T=5 小实验训练](#3-k5-t5-小实验训练)
- [4. K=5, T=5 CEM eval 100 cases](#4-k5-t5-cem-eval-100-cases)
- [5. K=5, T=5 轻量 CEM eval](#5-k5-t5-轻量-cem-eval)
- [下一步](#下一步)

## 实验目标

先使用 128k 样本的数据集跑一个小规模正式实验，验证当前 `latent_subgoal_act` 链路：

```text
build dataset -> inspect dataset -> train policy -> direct / LE rerank / CEM eval
```

本轮数据集采用：

```text
goal = t + 25
action chunk = a_{t:t+24}
future latent sequence = z_{t+1:t+25}
sample_mode = fixed_offset
```

## 1. 生成 128k 数据集

状态：已完成

命令记录时间：2026-07-05 16:59:49 +08:00

完成信息：

```text
saved 128000 samples to /data/zflin/lewm_re/stablewm_data/latent_subgoal_act_datasets/pusht_fixed_g25_k25_t25_ms_128k_train.pt
```

命令：

```bash
python -B -m latent_subgoal_act.build_dataset \
  output_dataset=pusht_fixed_g25_k25_t25_ms_128k_train.pt \
  sample_mode=fixed_offset \
  split=train \
  max_samples=128000 \
  eval.goal_offset_steps=25 \
  plan_config.action_block=25 \
  plan_config.subgoal_horizon=25 \
  encode_batch_size=128
```

关键参数：

| 参数 | 值 |
| --- | --- |
| `output_dataset` | `pusht_fixed_g25_k25_t25_ms_128k_train.pt` |
| `sample_mode` | `fixed_offset` |
| `split` | `train` |
| `max_samples` | `128000` |
| `eval.goal_offset_steps` | `25` |
| `plan_config.action_block` | `25` |
| `plan_config.subgoal_horizon` | `25` |
| `encode_batch_size` | `128` |

## 2. 检查数据集

状态：待运行

命令记录时间：2026-07-05 16:59:49 +08:00

命令：

```bash
python -B -m latent_subgoal_act.inspect_dataset \
  dataset=pusht_fixed_g25_k25_t25_ms_128k_train.pt \
  expected_split=train
```

需要重点检查：

```text
num_samples = 128000
missing_keys = []
z_h_seq_shape = [128000, 25, latent_dim]
action_shape = [128000, 25, action_dim]
split_ok = true
```

关键参数：

| 参数 | 值 |
| --- | --- |
| `dataset` | `pusht_fixed_g25_k25_t25_ms_128k_train.pt` |
| `expected_split` | `train` |

## 3. K=5, T=5 小实验训练

状态：已运行，结果待补充

命令记录时间：2026-07-05 17:02:41 +08:00

实验内容：

```text
使用 128k fixed_offset 数据集训练一个 K=5, T=5 的 latent subgoal ACT policy。
```

命令：

```bash
python -B -m latent_subgoal_act.train \
  dataset=pusht_fixed_g25_k25_t25_ms_128k_train.pt \
  output=subgoal_act_fixed_g25_K5_T5_ms128k \
  action_horizon=5 \
  subgoal_horizon=5 \
  train.epochs=200 \
  loader.batch_size=256
```

关键参数：

| 参数 | 值 |
| --- | --- |
| `dataset` | `pusht_fixed_g25_k25_t25_ms_128k_train.pt` |
| `output` | `subgoal_act_fixed_g25_K5_T5_ms128k` |
| `action_horizon` | `5` |
| `subgoal_horizon` | `5` |
| `train.epochs` | `200` |
| `loader.batch_size` | `256` |

待补充结果：

```text
训练是否完成：
最佳 epoch：
val_score：
policy.pt 路径：
备注：
```

## 4. K=5, T=5 CEM eval 100 cases

状态：未完成，参数过重，运行后长时间无新输出

命令记录时间：2026-07-05 20:17:23 +08:00

实验内容：

```text
使用 K=5, T=5 的 policy checkpoint 跑 CEM eval。
评估 case 数设为 100。
CEM 主要参数尽量对齐原始 LeWM CEM baseline。
```

与原始 CEM baseline 的参数映射：

| 原始 baseline 参数 | 本实验参数 |
| --- | --- |
| `num_samples=300` | `cem.num_candidates=300` |
| `n_steps=30` | `cem.num_iters=30` |
| `topk=30` | `cem.elite_frac=0.1` |
| `receding_horizon=5` | `plan_config.receding_horizon=5` |
| `goal_offset_steps=25` | `eval.goal_offset_steps=25` |
| `eval_budget=50` | `eval.eval_budget=50` |

命令：

```bash
python -B -m latent_subgoal_act.eval \
  policy_ckpt=subgoal_act_fixed_g25_K5_T5_ms128k/policy.pt \
  eval.num_eval=100 \
  eval.goal_offset_steps=25 \
  eval.eval_budget=50 \
  plan_config.receding_horizon=5 \
  world.num_envs=100 \
  cem.enabled=True \
  cem.num_iters=30 \
  cem.num_candidates=300 \
  cem.elite_frac=0.1 \
  cem.init_std=1.0 \
  cem.min_std=0.05 \
  rerank.target=subgoal
```

关键说明：

```text
当前 checkpoint 是 K=5, T=5。
因此 CEM target 是 z_hat_{t+5}，不是原始 baseline 的 z_g = z_{t+25}。
这个实验更像局部 subgoal CEM。
后续如需严格对齐 25-step CEM baseline，应使用 K=25, T=25 checkpoint。
```

待补充结果：

```text
是否完成：否
结果文件：
metrics：
evaluation_time：
备注：运行后出现 Gymnasium warning，随后长时间无新输出。该 warning 本身通常不是致命错误；更可能的原因是参数规模过大：
      eval.num_eval=100, world.num_envs=100, cem.num_candidates=300, cem.num_iters=30。
      每次 replanning 约需 100 * 300 * 30 = 900000 candidate rollout，
      eval_budget=50 且 receding_horizon=5 时约有 10 次 replanning，总量约 900 万 candidate rollout。
```

## 5. K=5, T=5 轻量 CEM eval

状态：待运行

命令记录时间：2026-07-05 20:28:08 +08:00

实验内容：

```text
将 CEM eval 参数缩小，用于确认评估链路可以稳定完成。
```

命令：

```bash
python -B -m latent_subgoal_act.eval \
  policy_ckpt=subgoal_act_fixed_g25_K5_T5_ms128k/policy.pt \
  eval.num_eval=10 \
  eval.goal_offset_steps=25 \
  eval.eval_budget=20 \
  plan_config.receding_horizon=5 \
  world.num_envs=10 \
  cem.enabled=True \
  cem.num_iters=3 \
  cem.num_candidates=32 \
  cem.elite_frac=0.25 \
  cem.init_std=0.3 \
  cem.min_std=0.05 \
  rerank.target=subgoal
```

关键参数：

| 参数 | 值 |
| --- | --- |
| `eval.num_eval` | `10` |
| `eval.eval_budget` | `20` |
| `world.num_envs` | `10` |
| `plan_config.receding_horizon` | `5` |
| `cem.num_iters` | `3` |
| `cem.num_candidates` | `32` |
| `cem.elite_frac` | `0.25` |
| `cem.init_std` | `0.3` |
| `cem.min_std` | `0.05` |
| `rerank.target` | `subgoal` |

待补充结果：

```text
是否完成：
结果文件：
metrics：
evaluation_time：
备注：
```

## 下一步

如果 K=5, T=5 训练完成，先运行 direct eval：

```bash
python -B -m latent_subgoal_act.eval \
  policy_ckpt=subgoal_act_fixed_g25_K5_T5_ms128k/policy.pt \
  eval.num_eval=50 \
  rerank.enabled=False \
  cem.enabled=False
```

## 6. Diffusion Action Prior, K=25

状态：待运行  
命令记录时间：2026-07-06 03:08:38 +08:00

实验内容：
```text
训练 goal-conditioned diffusion action prior：输入 z_t 和 z_g，输出 25-step action chunk 的动作分布。
这次对齐 128k dataset 中的最大动作 chunk：goal_offset_steps=25, action_horizon=25。
训练使用 2 号 GPU，epoch 从 200 提高到 400。
```

训练命令：
```bash
CUDA_VISIBLE_DEVICES=2 python -B -m latent_subgoal_act.action_priors.train_diffusion \
  dataset=pusht_fixed_g25_k25_t25_ms_128k_train.pt \
  output=goal_diffusion_prior_g25_K25_ms128k \
  action_horizon=25 \
  diffusion.num_steps=50 \
  train.epochs=400 \
  loader.batch_size=256 \
  device=cuda
```

关键参数：
| 参数 | 值 |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | `2` |
| `dataset` | `pusht_fixed_g25_k25_t25_ms_128k_train.pt` |
| `output` | `goal_diffusion_prior_g25_K25_ms128k` |
| `action_horizon` | `25` |
| `diffusion.num_steps` | `50` |
| `train.epochs` | `400` |
| `loader.batch_size` | `256` |
| `device` | `cuda` |

后续 Diffusion + CEM eval 命令：
```bash
CUDA_VISIBLE_DEVICES=2 python -B -m latent_subgoal_act.action_priors.eval_diffusion_cem \
  policy_ckpt=goal_diffusion_prior_g25_K25_ms128k/policy.pt \
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
  cem.std_scale=1.0 \
  device=cuda
```
