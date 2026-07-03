# Latent BC 实验逻辑与参数说明

这份文档解释当前 latent goal-conditioned behavior cloning 方案为什么这么做、LeWM 主干如何指导 policy、每个 BC 相关文件负责什么，以及实验时可以调哪些参数。默认任务是 PushT，默认运行环境是远端 SSH：

```bash
cd /data/zflin/lewm_re/le-wm
conda activate lewm
export STABLEWM_HOME=/data/zflin/lewm_re/stablewm_data
export HF_ENDPOINT=https://hf-mirror.com
export LEWM_EXPERIMENTS_DIR=/data/zflin/lewm_re/experiments
```

如果不设置 `LEWM_EXPERIMENTS_DIR`，BC 脚本会默认使用 `/data/zflin/lewm_re/experiments`。目标是让 `le-wm` 仓库尽量只保留代码、脚本和配置，实验数据、Hydra 日志、训练结果都放到仓库外。

## 1. 核心想法

原始 LeWM-CEM 的评估方式是：冻结 LeWM world model，然后在每个控制 step 上用 CEM 搜索一段 action，使 LeWM rollout 后的 latent 更接近 goal latent。

可以简化成：

```text
z_t = encoder(o_t)
z_g = encoder(o_goal)
choose a_{t:t+K} so rollout(z_t, a_{t:t+K}) approaches z_g
```

Latent BC 的想法是把这一步在线搜索换成一个前馈 policy：

```text
pi_psi(z_t, z_g, z_g - z_t) -> a_{t:t+K}
```

也就是说，LeWM 主干不直接输出动作，而是提供一个稳定的 latent 坐标系：

- `z_t` 表示当前图像状态。
- `z_g` 表示目标图像状态。
- `delta_z = z_g - z_t` 表示从当前状态到目标状态的 latent 方向。

BC policy 学的是：在这个 latent 坐标系里，专家通常会采取什么 action chunk。推理时不再运行 CEM 多轮采样和优化，只用一次 MLP forward 得到动作片段，因此预期速度更快。

## 2. LeWM 主干如何指导 Policy

当前方案冻结 LeWM，不微调 encoder 或 predictor。LeWM 的作用分成两段。

离线构建数据集时：

```text
o_t      -> LeWM encoder -> z_t
o_{t+G}  -> LeWM encoder -> z_g
z_g - z_t              -> delta_z
expert action          -> a_{t:t+K}
```

训练 BC policy 时：

```text
input  = concat(z_t, z_g, delta_z)
target = normalized expert action chunk
loss   = MSE(policy(input), target)
```

环境评估时：

```text
current pixels -> LeWM encoder -> z_t
goal pixels    -> LeWM encoder -> z_g
BC policy      -> normalized action chunk
inverse scaler -> environment action chunk
execute first few actions, then replan
```

这里的“指导”不是梯度上共同训练，而是 LeWM 把像素任务转换成 latent goal-conditioned 控制问题。policy 不需要从 raw pixels 重新学习视觉表示，而是直接学习 latent 到动作的映射。

## 3. 与 CEM Baseline 的关系

PushT 的 CEM baseline 已经在远端跑通。它是当前 BC 的主要对照。

CEM：

- 每个控制 step 在线优化 action chunk。
- 通常更贴近 LeWM latent objective。
- 推理慢，因为要采样、rollout、多轮更新。

Latent BC：

- 离线从专家数据中学习 action chunk。
- 推理快，因为只做 encoder + MLP。
- 成功率可能低于 CEM，尤其在专家动作多模态或 goal horizon 较难时。

第一阶段不要求 BC 超过 CEM。更合理的目标是观察：

```text
BC 能否用明显更低的推理成本，达到 CEM 的一部分成功率。
```

如果 BC 接近 CEM，说明 latent policy 替代 CEM 有希望。如果 BC 明显更弱，可以考虑后续做 CEM distillation 或 one-shot LeWM reranking。

## 4. 文件职责

### `scripts/build_latent_bc_dataset.py`

负责从 HDF5 expert dataset 构建 latent BC 数据集。

主要步骤：

1. 复用原始 `eval.py` 的 `get_dataset()`，继续走 HDF5 数据加载。
2. 用 `img_transform()` 保持和 CEM eval 一致的图像预处理。
3. 加载 LeWM checkpoint，默认 `pusht/lewm`。
4. 对合法的起点 index 编码当前图像和未来 goal 图像。
5. 提取专家动作片段 `a_{t:t+K}`。
6. 用 `StandardScaler` 标准化 action。
7. 保存 `.pt` 数据集到 `/data/zflin/lewm_re/experiments/latent_bc_datasets/`。

输出字段：

```text
z_t:       [N, latent_dim]
z_g:       [N, latent_dim]
delta_z:   [N, latent_dim]
action:    [N, K, action_dim]  标准化动作
action_raw:[N, K, action_dim]  原始动作
episode:   [N]
step:      [N]
goal_step: [N]
metadata:  config、action scaler、latent/action 维度等
```

### `latent_bc.py`

定义 policy 模型和环境评估时的 wrapper。

`LatentGoalBCPolicy` 是 MLP：

```text
concat(z_t, z_g, delta_z)
  -> Linear + LayerNorm + GELU + Dropout
  -> ...
  -> action_horizon * action_dim
  -> reshape [K, action_dim]
```

`LatentBCWorldPolicy` 用在环境评估中：

1. 从 `info` 里取当前 pixels 和 goal pixels。
2. 用冻结 LeWM encoder 得到 `z_t` 和 `z_g`。
3. 调用 BC policy 预测 action chunk。
4. 用保存的 action mean/scale 做 inverse transform。
5. 缓存前 `execute_horizon` 个动作并逐步执行。

### `train_latent_bc.py`

负责训练 MLP policy。

主要逻辑：

1. 加载 latent BC `.pt` 数据集。
2. 按 episode 划分 train/val，避免同一条轨迹泄漏到两个 split。
3. 用 MSE 训练标准化 action chunk。
4. 可选加入 smoothness 和 action magnitude 正则。
5. 保存最优 `policy.pt`、`config.yaml` 和 `metrics.jsonl`。

默认输出目录：

```text
/data/zflin/lewm_re/experiments/YYYY-MM-DD_pusht_latent_bc/
```

### `eval_latent_bc.py`

负责在 PushT 环境里评估训练好的 BC policy。

它复用原始评估流程中的：

- HDF5 dataset
- episode/start step 采样
- image transform
- world/evaluate 调用
- callables 设置 state 和 goal state

不同点是把 CEM `WorldModelPolicy` 换成 `LatentBCWorldPolicy`。

评估结果写到 policy checkpoint 所在目录，例如：

```text
/data/zflin/lewm_re/experiments/2026-07-03_pusht_latent_bc/pusht_results.txt
```

Hydra 日志会写到：

```text
/data/zflin/lewm_re/experiments/hydra/eval_latent_bc/
```

## 5. 参数说明

### 数据集构建参数

命令：

```bash
python scripts/build_latent_bc_dataset.py
```

可调参数：

```bash
python scripts/build_latent_bc_dataset.py \
  max_samples=128 \
  encode_batch_size=128 \
  lewm_policy=pusht/lewm \
  output_dataset=latent_bc_datasets/pusht_g25_k5.pt \
  device=cuda
```

参数含义：

- `max_samples`：只构建前 N 个样本，用于 smoke test。全量训练时不设置。
- `encode_batch_size`：LeWM encoder 编码 batch size。3090 可从 `128` 起试，显存够可试 `256`。
- `lewm_policy`：用于编码 latent 的 LeWM checkpoint，PushT 默认 `pusht/lewm`。
- `output_dataset`：输出 `.pt` 路径。相对路径会解析到 `/data/zflin/lewm_re/experiments`。
- `device`：默认 CUDA 可用时用 `cuda`。

还有两个来自 `config/eval/pusht.yaml` 的关键参数：

- `eval.goal_offset_steps`：默认 `25`，决定 `o_g = o_{t+25}`。
- `plan_config.action_block`：默认 `5`，决定 action chunk 长度 `K=5`。

如果要改 goal horizon：

```bash
python scripts/build_latent_bc_dataset.py eval.goal_offset_steps=10 output_dataset=latent_bc_datasets/pusht_g10_k5.pt
```

如果要改 action chunk 长度：

```bash
python scripts/build_latent_bc_dataset.py plan_config.action_block=3 output_dataset=latent_bc_datasets/pusht_g25_k3.pt
```

注意：训练和评估必须使用同一份 dataset metadata 中的 `action_horizon` 和 action scaler。

### 训练参数

命令：

```bash
python train_latent_bc.py
```

可调参数：

```bash
python train_latent_bc.py \
  dataset=latent_bc_datasets/pusht_g25_k5.pt \
  output=2026-07-03_pusht_latent_bc \
  train.epochs=100 \
  loader.batch_size=256 \
  model.hidden_dim=512 \
  model.depth=3 \
  model.dropout=0.1 \
  optim.lr=0.0003 \
  optim.weight_decay=0.0001 \
  loss.lambda_smooth=0.0 \
  loss.lambda_mag=0.0
```

参数含义：

- `dataset`：latent dataset 路径，相对 `/data/zflin/lewm_re/experiments`。
- `output`：训练输出目录，相对 `/data/zflin/lewm_re/experiments`。
- `train.epochs`：训练轮数。smoke test 可用 `1` 或 `2`。
- `train_split`：按 episode 划分训练集比例，默认 `0.9`。
- `loader.batch_size`：训练 batch size。latent 数据较小，通常可以较大。
- `model.hidden_dim`：MLP 隐层宽度。默认 `512`。
- `model.depth`：MLP 层数。默认 `3`。
- `model.dropout`：默认 `0.1`，数据少时可保留，数据足够时可尝试 `0.0`。
- `optim.lr`：AdamW 学习率。默认 `3e-4`。
- `optim.weight_decay`：默认 `1e-4`。
- `loss.lambda_smooth`：动作片段平滑正则。
- `loss.lambda_mag`：动作幅值正则。

第一轮建议保持默认模型，不急着加大网络。先看 train/val MSE 是否正常下降，以及环境成功率是否有信号。

### 评估参数

命令：

```bash
python eval_latent_bc.py policy_ckpt=2026-07-03_pusht_latent_bc/policy.pt
```

小评估：

```bash
python eval_latent_bc.py \
  policy_ckpt=2026-07-03_pusht_latent_bc/policy.pt \
  eval.num_eval=2
```

完整评估：

```bash
python eval_latent_bc.py \
  policy_ckpt=2026-07-03_pusht_latent_bc/policy.pt \
  eval.num_eval=50
```

可调参数：

- `policy_ckpt`：BC policy checkpoint，相对 `/data/zflin/lewm_re/experiments`。
- `lewm_policy`：手动覆盖 LeWM checkpoint。通常不需要，因为 `policy.pt` metadata 里会记录构建 dataset 时用的 `model_policy`。
- `eval.num_eval`：评估 episode 数。smoke test 用 `2`，正式对比用 `50`。
- `eval.goal_offset_steps`：目标 offset。应与构建 dataset 时的 goal offset 保持一致，除非明确做泛化测试。
- `plan_config.receding_horizon`：每次预测后实际执行多少个动作。默认 `5`。
- `device`：默认 CUDA。

## 6. 推荐实验流程

第一步，构建小样本 dataset：

```bash
python scripts/build_latent_bc_dataset.py max_samples=128
```

确认输出在：

```text
/data/zflin/lewm_re/experiments/latent_bc_datasets/pusht_g25_k5.pt
```

第二步，小训练：

```bash
python train_latent_bc.py train.epochs=2 output=2026-07-03_pusht_latent_bc_smoke
```

第三步，小评估：

```bash
python eval_latent_bc.py \
  policy_ckpt=2026-07-03_pusht_latent_bc_smoke/policy.pt \
  eval.num_eval=2
```

第四步，清掉 smoke dataset 或直接覆盖后构建全量：

```bash
python scripts/build_latent_bc_dataset.py
```

第五步，正式训练：

```bash
python train_latent_bc.py output=2026-07-03_pusht_latent_bc
```

第六步，正式评估：

```bash
python eval_latent_bc.py \
  policy_ckpt=2026-07-03_pusht_latent_bc/policy.pt \
  eval.num_eval=50
```

## 7. 结果怎么看

训练阶段先看：

- `train/bc_mse` 是否下降。
- `val/bc_mse` 是否稳定下降或至少不发散。
- train/val gap 是否很大。

评估阶段看：

- `success_rate`
- `episode_successes`
- `evaluation_time`
- 视频中的失败模式

和 CEM baseline 比较时重点看：

```text
BC success_rate / CEM success_rate
CEM evaluation_time / BC evaluation_time
```

如果 BC 成功率接近 CEM 且明显更快，可以继续扩大评估和调模型。如果 BC 明显更弱但行为有一定方向性，可以尝试更大模型、不同 `goal_offset_steps`、CEM distillation 或 one-shot reranking。如果 BC loss 正常但环境行为很乱，优先排查 action normalization、goal offset、episode split 和 action chunk 执行逻辑。

## 8. 常见参数选择建议

第一轮推荐：

```text
G = 25
K = 5
hidden_dim = 512
depth = 3
dropout = 0.1
lr = 3e-4
batch_size = 256
epochs = 100
lambda_smooth = 0.0
lambda_mag = 0.0
```

如果训练 loss 不下降：

- 降低学习率到 `1e-4`。
- 检查 `action` 是否有 NaN。
- 检查 `z_t/z_g` shape 和 scale。

如果 train loss 低、val loss 高：

- 增大 dropout 到 `0.2`。
- 降低 hidden dim 或 depth。
- 检查 episode split 是否有效。

如果 val loss 正常但环境成功率低：

- 检查 action inverse scaling。
- 尝试 `plan_config.receding_horizon=1`，每步重算动作。
- 尝试更短 goal horizon，例如 `eval.goal_offset_steps=10`。
- 后续考虑 CEM distillation 或 one-shot LeWM reranking。

如果推理速度还不够快：

- 确认没有跑 CEM。
- 增大 `plan_config.receding_horizon`，减少重新编码频率。
- 后续可以缓存 goal latent，避免每个 step 重复编码 goal。
