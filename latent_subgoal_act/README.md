# latent_subgoal_act 使用说明

`latent_subgoal_act` 用于训练和评估 **latent subgoal ACT policy**。它的目标是：先用预训练 LEWM/world model 把图像状态编码成 latent，再训练一个策略模型从当前 latent 和目标 latent 出发，预测中间 subgoal latent，并输出一段动作序列。

整体流程：

```text
原始 PushT 数据
  -> build_dataset.py 生成 latent subgoal 数据集
  -> train.py 训练 LatentSubgoalACTPolicy
  -> eval.py 在 stable_worldmodel.World 中评估 policy.pt
```

运行 Python 时建议统一使用 `python -B -m ...`。`-B` 会禁止生成 `.pyc` 和 `__pycache__`。

## 文件夹功能

| 文件 | 功能 |
| --- | --- |
| `build_dataset.py` | 从原始 PushT dataset 构建 latent subgoal ACT 训练数据。 |
| `inspect_dataset.py` | 检查构建好的 `.pt` 数据集是否完整、shape 是否正确、split 是否符合预期。 |
| `dataset.py` | PyTorch Dataset 封装，供 `train.py` 读取 `.pt` payload。 |
| `model.py` | 定义 `LatentSubgoalACTPolicy`，即先预测 subgoal latent、再预测动作块的 Transformer policy。 |
| `train.py` | 训练 policy，输出 `config.yaml`、`metrics.jsonl`、最佳 `policy.pt`。 |
| `policy.py` | 将训练好的 policy 包装成可被 `stable_worldmodel.World` 调用的闭环控制策略。 |
| `eval.py` | 加载 `policy.pt` 并在 PushT world 环境中评估，保存 metrics、结果文本和视频。 |
| `wm_rollout.py` | 使用 frozen LEWM 在 latent 空间中预测动作块执行后的 terminal latent，供训练 loss、rerank、CEM 使用。 |
| `shared.py` | 统一处理 experiment 和 dataset 路径解析。 |
| `__init__.py` | 暴露核心类，标记该目录为 Python package。 |

## 路径规则

实验输出目录由 `shared.py` 的 `experiments_root()` 决定：

| 优先级 | 路径 |
| --- | --- |
| 1 | 环境变量 `LEWM_EXPERIMENTS_DIR` |
| 2 | `STABLEWM_HOME/../experiments` |
| 3 | `/data/zflin/lewm_re/experiments` |

数据集目录由 `datasets_root()` 决定：

| 优先级 | 路径 |
| --- | --- |
| 1 | 环境变量 `LEWM_SUBGOAL_DATASETS_DIR` |
| 2 | `STABLEWM_HOME/latent_subgoal_act_datasets` |
| 3 | `/data/zflin/lewm_re/stablewm_data/latent_subgoal_act_datasets` |

相对 experiment 路径会写入 `experiments_root()`；绝对路径会直接使用。相对 dataset 路径会优先在 `datasets_root()` 下查找。

## SSH 上运行

登录服务器后，推荐用 `tmux` 或 `screen` 跑长任务，避免 SSH 断开后训练中断。

```bash
ssh <user>@<host>
tmux new -s subgoal
cd /data/zflin/lewm_re
```

进入项目目录后，按实际环境加载 conda 或 venv：

```bash
conda activate <env_name>
export STABLEWM_HOME=/data/zflin/lewm_re/stablewm_data
export PYTHONDONTWRITEBYTECODE=1
```

如果想固定实验输出目录：

```bash
export LEWM_EXPERIMENTS_DIR=/data/zflin/lewm_re/experiments
```

如果想固定 latent subgoal 数据集目录：

```bash
export LEWM_SUBGOAL_DATASETS_DIR=/data/zflin/lewm_re/stablewm_data/latent_subgoal_act_datasets
```

所有运行命令都建议在仓库根目录执行：

```bash
python -B -m latent_subgoal_act.<script_name> key=value key.subkey=value
```

这里使用的是 OmegaConf CLI 覆盖参数格式，例如：

```bash
python -B -m latent_subgoal_act.train train.epochs=50 loader.batch_size=128
```

tmux 常用操作：

```bash
# 断开但保留任务
Ctrl-b d

# 回到任务
tmux attach -t subgoal
```

## 1. 构建数据集

脚本：

```bash
python -B -m latent_subgoal_act.build_dataset output_dataset=pusht_g25_k5_h5.pt split=train
```

作用：从原始 dataset 中采样当前状态、subgoal、goal，调用 LEWM 编码图像 latent，保存训练用 `.pt` 文件。

主要可调参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `output_dataset` | `pusht_g25_k5_h5.pt` | 输出 `.pt` 文件名；相对路径写入 dataset root。 |
| `split` | `train` | 生成 `train`、`test`、`eval`、`val` 或 `all` split。 |
| `sample_mode` | `fixed_offset` | 采样方式：`fixed_offset` 或 `goal_anchored`。 |
| `max_samples` | `None` | 最多保存多少样本。 |
| `goal_stride` | `25` | `goal_anchored` 模式下 goal 采样间隔。 |
| `split_seed` | `42` | episode split 随机种子。 |
| `test_fraction` | `0.1` | test episode 比例。 |
| `encode_batch_size` | `128` | LEWM 图像编码 batch size。 |
| `device` | `cuda` | 优先使用 CUDA；无 CUDA 时自动回 CPU。 |
| `lewm_policy` | `pusht/lewm` | 用于编码 latent 的预训练 LEWM。 |
| `eval.dataset_name` | `pusht_expert_train` | 原始 PushT dataset 名称。 |
| `eval.goal_offset_steps` | `25` | 起点到 goal 的步数。 |
| `eval.img_size` | `224` | 图像输入尺寸。 |
| `plan_config.action_block` | `5` | 动作块长度。 |
| `plan_config.subgoal_horizon` | `5` | 起点到 subgoal 的步数。 |
| `plan_config.cap_subgoal_at_goal` | `True` | 是否把 subgoal 限制在 goal 之前。 |

输出 payload 主要包含：

```text
z_t, z_g, z_h, action, action_raw, episode, step, subgoal_step, goal_step, metadata
```

## 2. 检查数据集

脚本：

```bash
python -B -m latent_subgoal_act.inspect_dataset dataset=pusht_g25_k5_h5.pt expected_split=train
```

作用：检查 `.pt` 数据集是否缺 key、shape 是否合理、split 是否符合预期。

主要可调参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `dataset` | 必填 | 数据集路径或文件名。 |
| `expected_split` | metadata 中的 `split` | 期望数据属于哪个 split。 |
| `split_seed` | metadata 中的值，否则 `42` | 重建 episode split 的随机种子。 |
| `test_fraction` | metadata 中的值，否则 `0.1` | test episode 比例。 |

退出码：

| 退出码 | 含义 |
| --- | --- |
| `0` | 检查通过。 |
| `2` | payload 缺少必要字段。 |
| `3` | split 检查失败。 |

## 3. 训练 policy

脚本：

```bash
python -B -m latent_subgoal_act.train dataset=pusht_g25_k5_h5.pt output=pusht_subgoal_act_g25_k5_h5
```

作用：读取 latent subgoal 数据集，训练 `LatentSubgoalACTPolicy`，并保存最佳 checkpoint。

训练逻辑：

1. 读取 `.pt` payload，并用 `LatentSubgoalACTDataset` 包装。
2. 按 episode 划分 train/val，避免同一个 episode 同时出现在训练和验证中。
3. 模型输入 `z_t` 和 `z_g`，先预测 `pred_z_h`，再用 `z_t/z_g/pred_z_h` 预测动作块。
4. 默认不使用 `z_h_teacher`，避免 train/eval mismatch。
5. 每个 epoch 记录 train/val 指标，保存验证分数最好的 `policy.pt`。

主要可调参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `dataset` | `latent_subgoal_act_datasets/pusht_g25_k5_h5.pt` | 训练数据集路径或名称。 |
| `output` | `{today}_pusht_subgoal_act` | 实验输出目录名。 |
| `seed` | `42` | 随机种子。 |
| `train_split` | `0.9` | episode 级训练集比例。 |
| `device` | `cuda` | 训练设备。 |
| `max_samples` | `None` | 限制最多训练样本数。 |
| `loader.batch_size` | `256` | batch size。 |
| `loader.num_workers` | `0` | DataLoader worker 数。 |
| `model.hidden_dim` | `512` | Transformer hidden dim。 |
| `model.subgoal_depth` | `3` | subgoal Transformer 层数。 |
| `model.action_depth` | `4` | action Transformer 层数。 |
| `model.dropout` | `0.1` | dropout。 |
| `model.num_heads` | `8` | attention head 数。 |
| `optim.lr` | `3e-4` | AdamW learning rate。 |
| `optim.weight_decay` | `1e-4` | AdamW weight decay。 |
| `train.epochs` | `100` | 训练 epoch 数。 |
| `train.grad_clip` | `1.0` | 梯度裁剪阈值。 |
| `train.teacher_force_subgoal` | `False` | 是否用真实 `z_h` 训练 action head。 |
| `loss.lambda_subgoal` | `1.0` | subgoal MSE 权重。 |
| `loss.lambda_smooth` | `0.0` | 动作平滑 loss 权重。 |
| `wm.enabled` | `True` | 是否加载 frozen LEWM/world model。 |
| `wm.policy` | `pusht/lewm` | world model 名称。 |
| `wm.history_size` | `1` | latent rollout 使用的历史长度。 |
| `wm.lambda_rollout` | `0.0` | rollout latent 对真实 `z_h` 的 loss 权重。 |
| `wm.lambda_align` | `0.0` | rollout latent 对 `pred_z_h` 的 loss 权重。 |

loss 组成：

```text
total =
  action_mse
  + lambda_subgoal * subgoal_mse
  + lambda_smooth * smooth_loss
  + wm.lambda_rollout * wm_rollout_mse
  + wm.lambda_align * wm_align_mse
```

`action_mse` 是对 `[B, H, A]` 中所有元素取平均后的标量，不是长度为 `H` 的向量。

训练输出：

| 文件 | 说明 |
| --- | --- |
| `config.yaml` | 本次训练配置。 |
| `metrics.jsonl` | 每个 epoch 的 train/val 指标。 |
| `policy.pt` | 验证集分数最佳的 checkpoint。 |

后台运行示例：

```bash
nohup python -B -m latent_subgoal_act.train \
  dataset=pusht_g25_k5_h5.pt \
  output=pusht_subgoal_act_g25_k5_h5 \
  train.epochs=100 \
  > train_subgoal.log 2>&1 &
```

## 4. 评估 policy

默认 rerank 评估：

```bash
python -B -m latent_subgoal_act.eval policy_ckpt=pusht_subgoal_act_g25_k5_h5/policy.pt eval.num_eval=50
```

CEM 评估：

```bash
python -B -m latent_subgoal_act.eval policy_ckpt=pusht_subgoal_act_g25_k5_h5/policy.pt cem.enabled=True
```

附加 direct 评估：

```bash
python -B -m latent_subgoal_act.eval policy_ckpt=pusht_subgoal_act_g25_k5_h5/policy.pt rerank.enabled=False cem.enabled=False
```

作用：加载训练好的 `policy.pt`，构造 `LatentSubgoalACTWorldPolicy`，在 `stable_worldmodel.World` 中闭环评估，并保存结果。

评估模式：

| 模式 | 配置 | 说明 |
| --- | --- | --- |
| rerank eval | `rerank.enabled=True`, `cem.enabled=False` | 默认模式。ACT 先输出动作块，再加噪声生成候选，用 LEWM rollout 选择最接近 target 的候选。 |
| CEM eval | `cem.enabled=True` | 以 ACT 输出为初始化，用 CEM 迭代优化动作块。 |
| direct eval | `rerank.enabled=False`, `cem.enabled=False` | 附加/ablation。只执行 ACT 输出，不用 LEWM 做动作选择。 |

主要可调参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `policy_ckpt` | `None` | 必填，训练得到的 `<experiment>/policy.pt`。 |
| `seed` | `42` | 评估起点采样随机种子。 |
| `lewm_policy` | `pusht/lewm` | 图像编码和 rerank/CEM 使用的 LEWM。 |
| `device` | `cuda` | 评估设备。 |
| `world.env_name` | `swm/PushT-v1` | 评估环境。 |
| `world.num_envs` | `50` | 并行环境数。 |
| `eval.num_eval` | `50` | 评估起点数量。 |
| `eval.goal_offset_steps` | `25` | goal 相对起点的步数。 |
| `eval.eval_budget` | `50` | 每次评估预算步数。 |
| `eval.dataset_name` | `pusht_expert_train` | 评估起点来源 dataset。 |
| `eval.split` | `test` | 起点 split：`train`、`test`、`eval`、`val`、`all`。 |
| `eval.split_seed` | `42` | episode split 随机种子。 |
| `eval.test_fraction` | `0.1` | test episode 比例。 |
| `plan_config.receding_horizon` | `1` | 每次从动作块中执行多少步。 |
| `rerank.enabled` | `True` | 是否启用 rerank。 |
| `rerank.num_candidates` | `16` | rerank 候选数。 |
| `rerank.noise_std` | `0.2` | 候选动作噪声标准差。 |
| `rerank.target` | `subgoal` | rerank 目标：`subgoal` 或 `goal`。 |
| `cem.enabled` | `False` | 是否启用 CEM。 |
| `cem.num_iters` | `3` | CEM 迭代次数。 |
| `cem.num_candidates` | `64` | 每轮候选数。 |
| `cem.elite_frac` | `0.1` | elite 比例。 |
| `cem.init_std` | `0.5` | 初始动作噪声标准差。 |
| `cem.min_std` | `0.05` | 标准差下限。 |
| `temporal_ensemble.enabled` | `False` | 是否启用 temporal ensemble。 |
| `temporal_ensemble.decay` | `0.01` | temporal ensemble 衰减系数。 |
| `output.filename` | `None` | 自定义结果文件名。 |

默认结果文件名：

| 条件 | 文件 |
| --- | --- |
| `cem.enabled=True` | `pusht_cem_to_{target}_results.txt` |
| `rerank.enabled=True` | `pusht_rerank_to_{target}_results.txt` |
| 两者都关闭 | `pusht_direct_results.txt` |

评估结果和视频会保存在 `policy.pt` 所在目录。

## 推荐完整流程

```bash
# 1. 构建训练数据
python -B -m latent_subgoal_act.build_dataset \
  output_dataset=pusht_g25_k5_h5.pt \
  split=train

# 2. 检查数据
python -B -m latent_subgoal_act.inspect_dataset \
  dataset=pusht_g25_k5_h5.pt \
  expected_split=train

# 3. 训练
python -B -m latent_subgoal_act.train \
  dataset=pusht_g25_k5_h5.pt \
  output=pusht_subgoal_act_g25_k5_h5

# 4. 默认 rerank 评估
python -B -m latent_subgoal_act.eval \
  policy_ckpt=pusht_subgoal_act_g25_k5_h5/policy.pt \
  eval.num_eval=50

# 5. CEM 评估
python -B -m latent_subgoal_act.eval \
  policy_ckpt=pusht_subgoal_act_g25_k5_h5/policy.pt \
  cem.enabled=True
```

## 常见注意事项

1. `policy_ckpt` 可以写相对 experiment 路径，例如 `pusht_subgoal_act_g25_k5_h5/policy.pt`。
2. `train.teacher_force_subgoal=False` 是当前推荐默认值，因为评估时没有真实 `z_h_teacher`，这样可以减少 train/eval mismatch。
3. `wm.enabled=True` 只代表会加载 frozen world model；只有 `wm.lambda_rollout` 或 `wm.lambda_align` 大于 0 时，world model loss 才会影响训练。
4. 评估默认是 rerank；direct eval 是附加 ablation，需要显式关闭 rerank 和 CEM。
5. 如果不想生成 `__pycache__`，使用 `python -B` 或设置 `PYTHONDONTWRITEBYTECODE=1`。

## 建议补充内容，待审核

下面这些内容我建议后续可以补进 README，但先留给你决定：

1. **服务器环境表**：记录推荐 conda 环境名、CUDA 版本、Python 版本、关键依赖版本。
2. **实验命名规范**：例如 `pusht_subgoal_g{goal_offset}_k{action_block}_h{subgoal_horizon}`，方便后续找 checkpoint。
3. **指标解释**：说明 eval metrics 中每个字段代表什么、哪个指标最重要。
4. **常见报错排查**：例如找不到 dataset、找不到 LEWM checkpoint、MuJoCo/EGL 初始化失败、CUDA OOM。
5. **结果归档规范**：训练结束后如何整理 `config.yaml`、`metrics.jsonl`、`policy.pt`、评估视频和结果文本。
