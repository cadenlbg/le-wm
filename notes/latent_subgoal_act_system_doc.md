# Latent Subgoal ACT 系统说明

本文说明 `latent_subgoal_act/` 这个新文件夹的设计逻辑、文件作用、运行流程和主要参数。

## 1. 核心目标

之前的 `latent_act` 效果不理想，一个重要原因是它更像：

```text
z_t, z_g -> action chunk
```

虽然加了 latent auxiliary loss，但 latent 目标不够明确，LeWM 的世界预测能力也没有充分进入 action 选择。

`latent_subgoal_act` 改成更明确的三段式：

```text
1. 短期 subgoal:
   z_t, z_g -> z_hat_{t+5}

2. 动作生成:
   z_t, z_g, z_hat_{t+5} -> a_t, ..., a_{t+4}

3. LeWM 检查/规划:
   z_t, action chunk -> z_rollout_{t+5}
```

直觉是：

```text
先想 5 步后该到哪里，再想这 5 步怎么走。
```

推理时仍然是闭环控制：

```text
observe -> encode -> predict subgoal/action chunk -> execute 1 action -> observe again
```

## 2. 和原始 CEM 的关系

原始 LeWM CEM 大致是：

```text
sample action chunks
LeWM rollout 到未来 latent
选择最接近 z_g 的 action chunk
```

本方法有两种 LeWM 使用方式：

### 2.1 Local rerank

ACT 先给一个 action chunk，然后在它附近加噪声生成候选：

```text
candidate_i = action_ACT + noise_i
```

然后用 LeWM rollout：

```text
z_t, candidate_i -> z_rollout_i
```

选最接近 `z_hat_{t+5}` 的候选。

### 2.2 CEM to predicted subgoal

CEM 不直接追 `z_g`，而是追短期 subgoal：

```text
minimize ||z_rollout_{t+5} - z_hat_{t+5}||^2
```

这相当于：

```text
ACT 负责给短期目标
CEM/LeWM 负责找能到这个短期目标的动作
```

## 3. 数据集设计

新版 dataset 必须包含真实短期未来 latent：

```text
z_t          当前 latent
z_g          目标 latent，默认 t+25
z_h          短期未来 latent，默认 min(t+5, goal_step)
action       归一化后的 5 步 action chunk
action_raw   原始 5 步 action chunk
episode      episode id
step         当前 step
subgoal_step t+5 的 step
goal_step    t+25 的 step
metadata     数据集元信息
```

注意：`z_h` 是训练 target，不是推理输入。如果 `t+5` 已经超过 goal step，则直接令 `z_h = z_g`。

### 防止数据泄露

`build_dataset.py` 支持 episode-level split：

```text
split=train 只用 train episodes
split=test  只用 held-out test episodes
```

默认：

```text
split_seed = 42
test_fraction = 0.1
```

训练 dataset 和评估 episode 使用同一个 split 规则，因此可以避免同一条 episode 同时出现在训练和测试。

## 4. 文件说明

### `latent_subgoal_act/build_dataset.py`

作用：从 HDF5 expert trajectory 构建新版 latent dataset。

它会：

1. 读取 PushT HDF5 dataset。
2. 用 LeWM encoder 编码：
   - 当前帧 `z_t`
   - 短期未来帧 `z_h = z_{min(t+5, goal_step)}`
   - goal 帧 `z_g = z_{t+25}`
3. 提取 5 步 action chunk。
4. 按 episode split 过滤 train/test。
5. 保存 `.pt` 数据集。

支持两种采样模式：

```text
fixed_offset:
  每个 t 使用固定 goal=t+25。

goal_anchored:
  一个 goal 可以生成多个样本，例如 goal=z25:
  (z1, z25 -> z6)
  (z2, z25 -> z7)
  ...
  (z24, z25 -> z25)
```

常用命令：

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

重要参数：

```text
output_dataset      输出文件名
max_samples         最大样本数
sample_mode         fixed_offset 或 goal_anchored
goal_stride         goal_anchored 模式下每隔多少 step 选一个 goal anchor，默认 25
split               train/test/all
split_seed          episode split 随机种子
test_fraction       held-out test episode 比例
lewm_policy         用哪个 LeWM checkpoint 编码
encode_batch_size   编码 batch size
eval.goal_offset_steps     goal offset，默认 25
plan_config.subgoal_horizon subgoal horizon，默认 5
plan_config.cap_subgoal_at_goal 如果 t+H 超过 goal step，则 z_h=zg，默认 true
plan_config.action_block    action chunk 长度，默认 5
```

### `latent_subgoal_act/inspect_dataset.py`

作用：检查 dataset 是否合法，尤其是 split 是否干净。

常用命令：

```bash
python -m latent_subgoal_act.inspect_dataset \
  dataset=pusht_g25_k5_h5_train128k.pt \
  expected_split=train \
  split_seed=42 \
  test_fraction=0.1
```

它会输出：

```text
num_samples
num_episodes
z_t/z_g/z_h/action shape
metadata
split_ok
train_episode_overlap
test_episode_overlap
```

如果字段缺失或 split 不符合预期，会非零退出。

### `latent_subgoal_act/model.py`

作用：定义核心模型 `LatentSubgoalACTPolicy`。

模型分两段：

```text
subgoal transformer:
  input:  z_t, z_g, subgoal_query
  output: z_hat_{t+5}

action transformer:
  input:  z_t, z_g, z_hat_{t+5}, action_queries
  output: action chunk
```

默认结构参数：

```text
hidden_dim = 512
subgoal_depth = 3
action_depth = 4
num_heads = 8
dropout = 0.1
```

### `latent_subgoal_act/train.py`

作用：训练 subgoal ACT policy。

基础 loss：

```text
L_action  = MSE(action_pred, action_expert)
L_subgoal = MSE(z_hat_{t+5}, z_true_{t+5})

L = L_action + lambda_subgoal * L_subgoal
```

可选 LeWM rollout loss：

```text
z_rollout = frozen_LeWM(z_t, action_pred)

L_rollout = MSE(z_rollout, z_true_{t+5})
L_align   = MSE(z_rollout, z_hat_{t+5})
```

完整 loss：

```text
L = L_action
  + lambda_subgoal * L_subgoal
  + lambda_rollout * L_rollout
  + lambda_align * L_align
  + lambda_smooth * L_smooth
```

基础训练：

```bash
python -m latent_subgoal_act.train \
  dataset=pusht_g25_k5_h5_train128k.pt \
  output=subgoal_act_train128k \
  max_samples=128000 \
  seed=42 \
  train.epochs=100 \
  loss.lambda_subgoal=1.0
```

开启 LeWM consistency：

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

训练参数：

```text
loader.batch_size        默认 256
optim.lr                 默认 3e-4
optim.weight_decay       默认 1e-4
train.epochs             默认 100
train.grad_clip          默认 1.0
train.teacher_force_subgoal 默认 false
loss.lambda_subgoal      默认 1.0
loss.lambda_smooth       默认 0.0
wm.enabled               是否启用 frozen LeWM rollout loss
wm.lambda_rollout        rollout 到真实 z_h 的权重
wm.lambda_align          rollout 到预测 subgoal 的权重
```

输出：

```text
/data/zflin/lewm_re/experiments/<output>/policy.pt
/data/zflin/lewm_re/experiments/<output>/config.yaml
/data/zflin/lewm_re/experiments/<output>/metrics.jsonl
```

### `latent_subgoal_act/wm_rollout.py`

作用：用 frozen LeWM predictor 在 latent 空间 rollout。

接口：

```python
rollout_latent_with_actions(wm_model, z_t, action_chunk)
```

输入：

```text
z_t:          [B, latent_dim]
action_chunk: [B, K, action_dim]
```

输出：

```text
z_rollout_{t+K}: [B, latent_dim]
```

这个文件还会自动适配 LeWM action encoder 的输入维度：

```text
如果 LeWM 吃单步动作: [B, K, A]
如果 LeWM 吃 frameskip 展平动作: [B, 1, K*A]
```

### `latent_subgoal_act/policy.py`

作用：把训练好的 policy 包成 `stable_worldmodel.World` 可调用的 policy。

推理流程：

```text
1. 从环境 info 取当前 pixels 和 goal pixels
2. 用 LeWM encoder 得到 z_t, z_g
3. policy 预测 z_hat_{t+5} 和 action chunk
4. 可选 LeWM rerank / CEM
5. 可选 temporal ensemble
6. action 反归一化
7. 执行第一个 action
```

### `latent_subgoal_act/eval.py`

作用：在 PushT 环境中评估 policy。

默认设置：

```text
eval.split = test
rerank.enabled = true
rerank.num_candidates = 16
rerank.noise_std = 0.2
rerank.target = subgoal
cem.enabled = false
temporal_ensemble.enabled = false
```

默认评估：

```bash
python -m latent_subgoal_act.eval \
  policy_ckpt=/data/zflin/lewm_re/experiments/subgoal_act_train128k_wm01/policy.pt \
  lewm_policy=pusht/lewm \
  device=cuda \
  eval.split=test \
  eval.split_seed=42 \
  eval.test_fraction=0.1
```

关闭默认 rerank 做 ablation：

```bash
python -m latent_subgoal_act.eval \
  policy_ckpt=/data/zflin/lewm_re/experiments/subgoal_act_train128k_wm01/policy.pt \
  lewm_policy=pusht/lewm \
  device=cuda \
  rerank.enabled=false
```

开启 CEM：

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

CEM 的目标是：

```text
minimize ||LeWM(z_t, action_chunk)_5 - z_hat_{t+5}||^2
```

开启 temporal ensemble：

```bash
python -m latent_subgoal_act.eval \
  policy_ckpt=/data/zflin/lewm_re/experiments/subgoal_act_train128k_wm01/policy.pt \
  lewm_policy=pusht/lewm \
  device=cuda \
  temporal_ensemble.enabled=true \
  temporal_ensemble.decay=0.01
```

### `latent_subgoal_act/shared.py`

作用：统一路径解析。

数据集默认放在：

```text
/data/zflin/lewm_re/stablewm_data/latent_subgoal_act_datasets
```

实验默认放在：

```text
/data/zflin/lewm_re/experiments
```

可用环境变量覆盖：

```bash
export STABLEWM_HOME=/data/zflin/lewm_re/stablewm_data
export LEWM_EXPERIMENTS_DIR=/data/zflin/lewm_re/experiments
export LEWM_SUBGOAL_DATASETS_DIR=/data/zflin/lewm_re/stablewm_data/latent_subgoal_act_datasets
```

### `scripts/archive_subgoal_act_experiment.sh`

作用：归档一次 subgoal ACT 实验。

用法：

```bash
bash scripts/archive_subgoal_act_experiment.sh \
  subgoal_act_train128k_wm01 \
  "python -m latent_subgoal_act.eval policy_ckpt=/data/zflin/lewm_re/experiments/subgoal_act_train128k_wm01/policy.pt eval.split=test"
```

会生成：

```text
README.md
command.txt
manifest.txt
eval_pusht.yaml
notes/
artifacts/*.mp4
```

## 5. 推荐实验顺序

### Stage 0: 建数据并检查

```bash
python -m latent_subgoal_act.build_dataset \
  output_dataset=pusht_g25_k5_h5_train128k.pt \
  max_samples=128000 \
  split=train \
  split_seed=42 \
  test_fraction=0.1 \
  lewm_policy=pusht/lewm \
  device=cuda

python -m latent_subgoal_act.inspect_dataset \
  dataset=pusht_g25_k5_h5_train128k.pt \
  expected_split=train \
  split_seed=42 \
  test_fraction=0.1
```

### Stage 1: Base subgoal ACT

```bash
python -m latent_subgoal_act.train \
  dataset=pusht_g25_k5_h5_train128k.pt \
  output=subgoal_act_train128k \
  max_samples=128000 \
  seed=42 \
  train.epochs=100 \
  loss.lambda_subgoal=1.0
```

### Stage 2: Add LeWM training consistency

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
  wm.lambda_rollout=0.1 \
  wm.lambda_align=0.1
```

### Stage 3: Eval with default rerank

```bash
python -m latent_subgoal_act.eval \
  policy_ckpt=/data/zflin/lewm_re/experiments/subgoal_act_train128k_wm01/policy.pt \
  lewm_policy=pusht/lewm \
  device=cuda
```

### Stage 4: Eval with CEM

```bash
python -m latent_subgoal_act.eval \
  policy_ckpt=/data/zflin/lewm_re/experiments/subgoal_act_train128k_wm01/policy.pt \
  lewm_policy=pusht/lewm \
  device=cuda \
  cem.enabled=true
```

### Stage 5: Add temporal ensemble

```bash
python -m latent_subgoal_act.eval \
  policy_ckpt=/data/zflin/lewm_re/experiments/subgoal_act_train128k_wm01/policy.pt \
  lewm_policy=pusht/lewm \
  device=cuda \
  temporal_ensemble.enabled=true
```

## 6. 如何看结果

训练时重点看：

```text
train/action_mse
val/action_mse
train/subgoal_mse
val/subgoal_mse
train/wm_rollout_mse
val/wm_rollout_mse
```

如果：

```text
subgoal_mse 降，但 action 成功率不升
```

说明短期目标学到了，但动作头没能稳定到达它。

如果：

```text
wm_rollout_mse 高
```

说明 action chunk 在 LeWM 眼里不能把状态推到 `z_h`，需要加大 `wm.lambda_rollout` 或尝试 CEM/rerank。

如果：

```text
轨迹抖动
```

优先开 temporal ensemble。

如果：

```text
接近成功但最后动作不稳
```

优先开 CEM 或增大 rerank candidates。

## 7. 当前最重要的消融

建议至少跑：

```text
1. base subgoal ACT
2. base + default rerank
3. base + temporal ensemble
4. base + CEM
5. train with wm rollout loss + default rerank
6. train with wm rollout loss + CEM
```

这样才能判断瓶颈在：

```text
subgoal 学不好
action 学不好
LeWM rollout 约束不够
推理阶段 action 选择不够
```
