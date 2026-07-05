# Latent Subgoal ACT 实验计划

这份文档说明当前 `latent_subgoal_act` 实验的核心原理、正式实验流程，以及后续要扩展的 latent action prior + CEM refinement 方向。

## 1. 实验核心思想

原始 LeWM CEM baseline 是一个纯 planning 方法。它不训练新的 action policy，而是在评估时在线搜索动作序列：

```text
当前观测 o_t
目标观测 o_g = o_{t+25}
        ↓
LEWM encoder 得到 z_t, z_g
        ↓
CEM 在 action space 搜索候选动作块
        ↓
LEWM rollout 预测每个动作块的 terminal latent
        ↓
选择最接近 z_g 的动作块
        ↓
执行 receding_horizon 步，再重新规划
```

它的目标函数是：

```text
min || z_rollout_terminal - z_g ||^2
```

PushT baseline 默认：

```text
horizon = 5
action_block = 5
goal_offset_steps = 25
```

所以原始 CEM 每次大约规划未来 25 个环境 step。

我们的实验是在这个基础上加入一个 learned policy。policy 不直接只预测 action，而是先预测未来 latent trajectory：

```text
z_t, z_g
  -> z_hat_{t+1}, z_hat_{t+2}, ..., z_hat_{t+T}
  -> action chunk a_t, ..., a_{t+K-1}
```

这有一点 VLA-JEPA 风格：模型不仅学习动作，还学习“未来状态应该怎么演化”。当前实现里：

```text
Subgoal Transformer:
  z_t, z_g -> z_hat_{t+1:t+T}

Action Transformer:
  z_t, z_g, z_hat_{t+1:t+T} -> action chunk
```

训练监督来自专家轨迹：

```text
L_action  = MSE(action_pred, action_gt)
L_future  = MSE(z_hat_{t+1:t+T}, z_true_{t+1:t+T})
```

可选 world model 辅助 loss：

```text
L_rollout = MSE(LEWM(z_t, action_pred)_terminal, z_true_{tT})
L_align   = MSE(LEWM(z_t, action_pred)_terminal, z_hat_{tT})
```

当前默认 `wm.enabled=True`，但 `wm.lambda_rollout=0.0`、`wm.lambda_align=0.0`，所以 world model loss 默认不影响训练。正式实验中可以后续单独打开它做 ablation。

## 2. 数据集设计

正式实验先构建一个大数据集：

```text
goal = t + 25
z_h_seq = z_{t+1:t+25}
action = a_{t:t+24}
```

也就是：

```text
eval.goal_offset_steps = 25
plan_config.subgoal_horizon = 25
plan_config.action_block = 25
sample_mode = fixed_offset
```

这样和原始 CEM baseline 的 25-step goal 对齐。

训练时可以从这个大数据集中截断：

```text
subgoal_horizon=5,  action_horizon=5
subgoal_horizon=10, action_horizon=10
subgoal_horizon=25, action_horizon=25
```

因此不需要为每个 horizon 重新 build dataset。

数据集 `.pt` 中主要字段：

| 字段 | shape | 含义 |
| --- | --- | --- |
| `z_t` | `[N, D]` | 当前图像 latent。 |
| `z_g` | `[N, D]` | `t+25` goal 图像 latent。 |
| `z_h_seq` | `[N, 25, D]` | 未来 latent 序列 `z_{t+1:t+25}`。 |
| `z_h` | `[N, D]` | terminal latent，即 `z_h_seq[:, -1]`。 |
| `action` | `[N, 25, A]` | 标准化动作块 `a_{t:t+24}`。 |
| `action_raw` | `[N, 25, A]` | 未标准化动作块。 |
| `episode`, `step` | `[N]` | 样本来源 episode 和起点 step。 |
| `subgoal_steps` | `[N, 25]` | 每个 future latent 对应的 step。 |
| `goal_step` | `[N]` | goal 对应 step。 |
| `metadata` | dict | 构建参数、维度、action scaler 等。 |

## 3. 当前三种评估方式

### 3.1 Direct

Direct 是最简单的闭环 policy：

```text
z_t, z_g -> policy -> action chunk
执行前 receding_horizon 步
重新观测
```

它不使用 LEWM 对动作做筛选。

命令里关闭 rerank 和 CEM：

```bash
rerank.enabled=False cem.enabled=False
```

Direct 是重要 ablation，用来判断 learned policy 本身是否有效。

### 3.2 LE rerank

LE rerank 是一次性候选筛选：

```text
policy 输出 action_chunk
复制成多个候选
除第 0 个外，其余加 Gaussian noise
LEWM rollout 每个候选
选择 terminal latent 最接近 z_hat_{tT} 的候选
```

目标函数：

```text
min || LEWM(z_t, action_candidate)_terminal - z_hat_{tT} ||^2
```

这里 target 不是 `z_g`，而是 policy 预测的 terminal future latent：

```text
z_hat_{tT} = pred_z_h_seq[:, -1]
```

LE rerank 比 CEM 快，因为它只采样一轮，不迭代更新分布。

### 3.3 CEM

CEM 是多轮优化：

```text
mean = policy 输出的 action_chunk
std = cem.init_std

for iter in cem.num_iters:
  从 N(mean, std) 采样 action candidates
  用 LEWM rollout 计算 cost
  选 cost 最低的 elite candidates
  用 elite 更新 mean/std

返回所有迭代中 cost 最低的 action chunk
```

目标函数同样是：

```text
min || LEWM(z_t, action_candidate)_terminal - z_hat_{tT} ||^2
```

所以当前 CEM 和原始 CEM baseline 的主要区别是：

```text
原始 CEM:
  搜索目标是 z_g
  action 分布不来自 learned policy

当前 CEM:
  action mean 来自 learned policy
  搜索目标是 z_hat_{tT}
```

这使 CEM 更像一个局部 refinement，而不是完全从零搜索。

## 4. 正式实验流程和命令

以下命令假设在服务器仓库根目录执行。

建议先开 tmux：

```bash
tmux new -s subgoal_exp
```

设置环境变量：

```bash
export PYTHONDONTWRITEBYTECODE=1
export LEWM_EXPERIMENTS_DIR=/data/zflin/lewm_re/experiments
export LEWM_SUBGOAL_DATASETS_DIR=/data/zflin/lewm_re/stablewm_data/latent_subgoal_act_datasets
```

### 4.1 生成正式大型数据集

```bash
python -B -m latent_subgoal_act.build_dataset \
  output_dataset=pusht_fixed_g25_k25_t25_ms_128k_train.pt \
  sample_mode=fixed_offset \
  split=train \
  max_sample=128000\
  eval.goal_offset_steps=25 \
  plan_config.action_block=25 \
  plan_config.subgoal_horizon=25 \
  encode_batch_size=128
```

检查数据集：

```bash
python -B -m latent_subgoal_act.inspect_dataset \
  dataset=pusht_fixed_g25_k25_t25_train.pt \
  expected_split=train
```

### 4.2 训练 K=5, T=5 policy

```bash
python -B -m latent_subgoal_act.train \
  dataset=pusht_fixed_g25_k25_t25_train.pt \
  output=subgoal_act_fixed_g25_K5_T5 \
  action_horizon=5 \
  subgoal_horizon=5 \
  train.epochs=200 \
  loader.batch_size=256
```

### 4.3 训练 K=25, T=25 policy

```bash
python -B -m latent_subgoal_act.train \
  dataset=pusht_fixed_g25_k25_t25_train.pt \
  output=subgoal_act_fixed_g25_K25_T25 \
  action_horizon=25 \
  subgoal_horizon=25 \
  train.epochs=200 \
  loader.batch_size=256
```

如果显存不够，把 batch size 降到 128 或 64。

### 4.4 Direct eval

K=5, T=5：

```bash
python -B -m latent_subgoal_act.eval \
  policy_ckpt=subgoal_act_fixed_g25_K5_T5/policy.pt \
  eval.num_eval=50 \
  rerank.enabled=False \
  cem.enabled=False
```

K=25, T=25：

```bash
python -B -m latent_subgoal_act.eval \
  policy_ckpt=subgoal_act_fixed_g25_K25_T25/policy.pt \
  eval.num_eval=50 \
  rerank.enabled=False \
  cem.enabled=False
```

### 4.5 LE rerank eval

K=5, T=5：

```bash
python -B -m latent_subgoal_act.eval \
  policy_ckpt=subgoal_act_fixed_g25_K5_T5/policy.pt \
  eval.num_eval=50 \
  rerank.enabled=True \
  rerank.num_candidates=16 \
  rerank.noise_std=0.2 \
  rerank.target=subgoal \
  cem.enabled=False
```

K=25, T=25：

```bash
python -B -m latent_subgoal_act.eval \
  policy_ckpt=subgoal_act_fixed_g25_K25_T25/policy.pt \
  eval.num_eval=50 \
  rerank.enabled=True \
  rerank.num_candidates=16 \
  rerank.noise_std=0.2 \
  rerank.target=subgoal \
  cem.enabled=False
```

### 4.6 CEM eval

K=5, T=5：

```bash
python -B -m latent_subgoal_act.eval \
  policy_ckpt=subgoal_act_fixed_g25_K5_T5/policy.pt \
  eval.num_eval=50 \
  cem.enabled=True \
  cem.num_iters=3 \
  cem.num_candidates=64 \
  cem.elite_frac=0.1 \
  cem.init_std=0.5 \
  cem.min_std=0.05 \
  rerank.target=subgoal
```

K=25, T=25：

```bash
python -B -m latent_subgoal_act.eval \
  policy_ckpt=subgoal_act_fixed_g25_K25_T25/policy.pt \
  eval.num_eval=50 \
  cem.enabled=True \
  cem.num_iters=3 \
  cem.num_candidates=64 \
  cem.elite_frac=0.1 \
  cem.init_std=0.5 \
  cem.min_std=0.05 \
  rerank.target=subgoal
```

## 5. 建议记录的结果

每个实验目录会保存：

```text
config.yaml
metrics.jsonl
policy.pt
pusht_direct_results.txt
pusht_rerank_to_subgoal_results.txt
pusht_cem_to_subgoal_results.txt
videos
```

建议表格记录：

| 实验 | K | T | eval mode | num_eval | metric | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| `subgoal_act_fixed_g25_K5_T5` | 5 | 5 | direct | 50 | TBD | policy only |
| `subgoal_act_fixed_g25_K5_T5` | 5 | 5 | rerank | 50 | TBD | LE rerank |
| `subgoal_act_fixed_g25_K5_T5` | 5 | 5 | CEM | 50 | TBD | policy warm-start CEM |
| `subgoal_act_fixed_g25_K25_T25` | 25 | 25 | direct | 50 | TBD | full horizon |
| `subgoal_act_fixed_g25_K25_T25` | 25 | 25 | rerank | 50 | TBD | LE rerank |
| `subgoal_act_fixed_g25_K25_T25` | 25 | 25 | CEM | 50 | TBD | policy warm-start CEM |

## 6. 后续方向：latent action prior + CEM refinement

当前 CEM 搜索空间仍然是 raw action chunk：

```text
action_candidate shape = [K, action_dim]
```

如果 `K=25`、`action_dim=2`，搜索空间是 50 维。对更复杂机器人任务，这个空间会更高维、更难搜。

latent action prior 的目标是把 CEM 从 raw action space 转移到更低维、更语义化的 latent action space。

### 6.1 模型形式

引入 latent action：

```text
u in R^d
```

例如 `d=8/16/32`。

模型可以拆成：

```text
prior:
  z_t, z_g -> u_mean, u_std

decoder:
  z_t, z_g, u -> action chunk

future head:
  z_t, z_g, u -> z_hat_{t+1:t+T}
```

训练时可选加入 posterior encoder：

```text
posterior:
  z_t, z_g, action_gt -> u_mean_post, u_std_post
```

如果做 CVAE，训练 loss 是：

```text
L_action = MSE(action_pred, action_gt)
L_future = MSE(z_hat_seq, z_h_seq)
L_KL     = KL(q(u | z_t, z_g, action_gt) || p(u | z_t, z_g))
```

### 6.2 推理时 latent CEM

推理时先由 prior 给出初始 latent action 分布：

```text
z_t, z_g -> u_mean, u_std
```

然后 CEM 在 latent action 空间中搜索：

```text
for iter:
  u_candidates ~ N(u_mean, u_std)
  action_candidates = decoder(z_t, z_g, u_candidates)
  z_rollout = LEWM(z_t, action_candidates)
  cost = || z_rollout_terminal - z_hat_{tT} ||^2
  elite = top low-cost u_candidates
  update u_mean, u_std from elite
```

最后执行：

```text
best_u -> decoder -> best_action_chunk
```

### 6.3 相比当前 CEM 的优势

当前 CEM：

```text
搜索 raw action
candidate = action_mean + noise
```

latent action CEM：

```text
搜索 latent u
candidate_action = decoder(z_t, z_g, u)
```

潜在优势：

1. 搜索维度更低。
2. decoder 约束动作更接近专家分布。
3. prior 给 CEM 更好的初始化分布。
4. CEM 可以探索不同“技能方向”，而不是逐维扰动动作。

### 6.4 建议实现顺序

先不要直接上完整 CVAE。建议分三步：

1. **当前方案 A 跑正式实验**
   - policy warm-start raw-action CEM
   - 对比 direct、LE rerank、CEM

2. **确定性 latent bottleneck**
   - `z_t,z_g -> u -> action/z_hat_seq`
   - 不加 KL，先确认低维 `u` 不会明显损失 action MSE 和 eval 表现

3. **latent action prior + latent CEM**
   - CEM 搜 `u`
   - decoder 输出 action
   - LEWM rollout 计算 cost

如果第 2 步稳定，再进入第 3 步。
