# Latent Goal-Conditioned Behavior Cloning 计划

## 1. 目标

本计划探索在冻结 LeWM 主干之后，用一个非 CEM 的 policy 层来完成目标条件动作预测。

冻结 LeWM 提供：

```text
z_t = encoder(o_t)
z_g = encoder(o_goal)
z_{t+1} = predictor(z_t, a_t)
```

新增学习的部分是：

```text
pi_psi(z_t, z_g, delta_z) -> a_{t:t+K}
delta_z = z_g - z_t
```

核心问题是：一个在 LeWM latent 上训练的 goal-conditioned policy，能否替代或部分替代原始 LeWM 的 CEM planner。

## 2. 对比基线

原始 LeWM 评估通过 `stable_worldmodel` 使用 CEM。

相关文件：

- `eval.py`：加载 LeWM checkpoint，并包装成 `WorldModelPolicy`。
- `jepa.py`：提供 `get_cost()`、`rollout()`、`criterion()` 给 CEM 调用。
- `config/eval/solver/cem.yaml`：CEM solver 配置。
- `config/eval/*.yaml`：规划 horizon、receding horizon、action block 和数据集配置。

原始规划目标：

```text
cost(a_{t:t+H}) = || rollout(z_t, a_{t:t+H})_H - z_g ||_2^2
```

CEM baseline 需要记录两种设置：

```text
1. 完整 baseline：
   python eval.py --config-name=pusht.yaml policy=pusht/lewm

2. 快速调试 baseline：
   python eval.py --config-name=pusht.yaml policy=pusht/lewm eval.num_eval=2 solver.num_samples=50 solver.n_steps=5
```

需要记录的指标：

- success rate
- 每个 episode 是否成功
- evaluation wall time
- 如果方便，记录平均每个控制 step 的延迟
- CEM 配置：`num_samples`、`n_steps`、`horizon`、`receding_horizon`、`action_block`

## 3. 方法：Latent Goal-Conditioned BC

### 3.1 输入构造

对每个训练样本，构造：

```text
当前观测：      o_t
目标观测：      o_g = o_{t+G}
专家动作片段：  a_{t:t+K}
```

用冻结 LeWM 编码：

```text
z_t = encoder(o_t)
z_g = encoder(o_g)
delta_z = z_g - z_t
```

policy 输入：

```text
x_t = concat(z_t, z_g, delta_z)
```

policy 输出：

```text
pi_psi(x_t) = a_{t:t+K}
```

PushT 第一版建议：

```text
G = config eval goal_offset_steps，默认 25
K = plan_config.action_block 或 plan_config.receding_horizon，默认 5
```

### 3.2 训练损失

第一版使用确定性 behavior cloning：

```text
L_BC = mean || pi_psi(z_t, z_g, delta_z) - a_expert_{t:t+K} ||_2^2
```

可选稳定项：

```text
L_smooth = mean || a_hat_{i+1} - a_hat_i ||_2^2
L_mag = mean || a_hat_i ||_2^2
L_total = L_BC + lambda_smooth L_smooth + lambda_mag L_mag
```

这一阶段不训练、不微调 LeWM。

## 4. 实现计划

### 4.1 生成 latent 数据集

新增脚本：

```text
scripts/build_latent_bc_dataset.py
```

职责：

1. 用 `swm.wm.utils.load_pretrained()` 加载 LeWM checkpoint。
2. 用 `stable_worldmodel.data.load_dataset()` 加载 expert dataset。
3. 使用和 `eval.py` 一致的图像 transform。
4. 对合法 index 编码 `o_t` 和 `o_{t+G}`，得到 `z_t` 和 `z_g`。
5. 提取专家动作片段 `a_{t:t+K}`。
6. 把紧凑 tensor 数据集保存到 `experiments/latent_bc_datasets/`。

建议输出字段：

```text
z_t:       float32 [N, D]
z_g:       float32 [N, D]
delta_z:   float32 [N, D]
action:    float32 [N, K, action_dim]
episode:   int64   [N]
step:      int64   [N]
goal_step: int64   [N]
```

第一版只使用 CLS embedding，因为当前 `jepa.py` 暴露的是 `info["emb"]` 中的 CLS latent。dense ViT patch tokens 可以作为后续扩展。

### 4.2 Policy 模块

新增模块：

```text
latent_bc.py
```

第一版 policy：

```text
class LatentGoalBCPolicy(nn.Module):
    input:  concat(z_t, z_g, z_g - z_t)
    trunk:  MLP
    output: action chunk [K, action_dim]
```

初始网络建议：

```text
input_dim = 3 * latent_dim
hidden_dim = 512
depth = 3 or 4
dropout = 0.1
output_dim = K * action_dim
```

policy 先保持小而清楚，避免把问题变成“更大的控制器是否更强”。

### 4.3 训练脚本

新增脚本：

```text
train_latent_bc.py
```

职责：

1. 加载 latent BC dataset。
2. 按 episode 划分 train/val，避免同一轨迹泄漏到两个 split。
3. 训练 `LatentGoalBCPolicy`。
4. 记录 train/val action MSE。
5. 保存 checkpoint 和 config 到：

```text
experiments/YYYY-MM-DD_pusht_latent_bc/
```

最小命令形式：

```text
python train_latent_bc.py \
  dataset=experiments/latent_bc_datasets/pusht_g25_k5.pt \
  output=experiments/YYYY-MM-DD_pusht_latent_bc
```

### 4.4 评估 policy

新增评估入口：

```text
eval_latent_bc.py
```

它应尽量复用 `eval.py` 的逻辑，但把：

```text
WorldModelPolicy(solver=CEM, model=LeWM)
```

替换成：

```text
LatentBCWorldPolicy(lewm_encoder=frozen_lewm, bc_policy=policy)
```

每个控制 step：

```text
1. 编码当前 pixels -> z_t。
2. 编码目标 pixels -> z_g。
3. policy 预测 action chunk -> a_{t:t+K}。
4. 执行第一个动作或前几个动作。
5. 重新编码新观测并重复。
```

关键要求：图像 transform、数据集起点/目标选择逻辑、action 标准化方式必须和 `eval.py` 一致，否则和 CEM 的比较不公平。

## 5. 与原始 CEM 性能的关系

BC policy 和 CEM 可以有三种关系。

### 5.1 BC 直接替代 CEM

推理：

```text
a_{t:t+K} = pi_psi(z_t, z_g, delta_z)
```

预期权衡：

- 比 CEM 快很多。
- 如果专家数据覆盖不足，或 latent 条件不够表达动作需求，成功率可能低于 CEM。
- 推理时没有迭代优化。

这是第一阶段的主比较。

### 5.2 BC 蒸馏 CEM

后续变体：

```text
pi_psi(z_t, z_g) -> a_CEM
```

不是模仿 expert action，而是离线运行 LeWM-CEM，把 CEM 选出的动作作为监督标签。

预期权衡：

- policy 会更直接模仿 CEM 的 latent objective。
- 推理时仍然不需要 CEM。
- 需要先离线跑 CEM 生成标签。

如果 expert BC 明显弱于 CEM，这是很自然的下一步。

### 5.3 BC 作为 expert prior，加 one-shot LeWM reranking

后续变体，仍然不是 CEM：

```text
围绕 pi_psi(z_t, z_g) 采样或扰动 N 个 action chunks
用 frozen LeWM rollout 每个 chunk
选择 terminal latent 距离目标最近的 chunk
```

它没有 elite selection、没有重拟合分布、没有多轮迭代，所以不是 CEM。它测试的是：learned expert prior 加 frozen LeWM verifier，能否用更低成本恢复一部分 CEM 精度。

## 6. 实验矩阵

先从 PushT 开始，因为仓库里已有一个小规模 LeWM-CEM smoke test：

```text
experiments/2026-07-01_pusht_lewm/
```

| ID | 方法 | Planner | 训练目标 | 作用 |
| --- | --- | --- | --- | --- |
| B0 | Random | 无 | 无 | sanity lower bound |
| B1 | LeWM-CEM full | CEM | 无 | 原始 baseline |
| B2 | LeWM-CEM debug | 少量 sample/step 的 CEM | 无 | 快速迭代 baseline |
| M1 | Latent GCBC | 无 | expert action chunk | 第一版非 CEM 方法 |
| M2 | Latent GCBC 去掉 delta | 无 | expert action chunk | 测试 `delta_z` 是否有帮助 |
| M3 | Latent GCBC + one-shot rerank | 一次性 verifier | expert action chunk | 测试廉价 LeWM verification |

主比较：

```text
M1 success_rate / B1 success_rate
M1 eval_time / B1 eval_time
```

解释方式：

- 如果 M1 成功率接近 CEM，但延迟显著更低，说明 BC 是强替代方案。
- 如果 M1 更弱但快很多，尝试 M3 reranking 或 CEM distillation。
- 如果 M1 明显失败，先检查 latent 可分性、action 标准化和专家动作多模态，再考虑 diffusion policy。

## 7. 评估指标

任务指标：

- success rate
- episode successes
- 如果环境提供，记录 task-specific distance to goal

policy 指标：

- validation action MSE
- rollout action smoothness
- action magnitude distribution vs expert
- 超出 expert action min/max 范围的比例

运行指标：

- total evaluation wall time
- 平均每个环境 step 的 policy inference time
- 如果方便，记录近似 GPU memory usage

公平比较要求：

- 和 CEM 使用相同 eval episodes 和 start indices。
- 相同 `goal_offset_steps`。
- 相同 image transform。
- 相同 action normalization / inverse transform。
- 相同 environment budget。

## 8. 里程碑

### Milestone 0：CEM Reference

确认当前 LeWM-CEM 仍能运行：

```text
python eval.py --config-name=pusht.yaml policy=pusht/lewm eval.num_eval=2 solver.num_samples=50 solver.n_steps=5
```

保存结果到：

```text
experiments/YYYY-MM-DD_pusht_cem_reference/
```

### Milestone 1：Latent Dataset

生成：

```text
experiments/latent_bc_datasets/pusht_g25_k5.pt
```

验证：

- tensor shape 符合预期
- `z_t`、`z_g`、`action` 没有 NaN
- episode split 合法

### Milestone 2：离线 BC 训练

训练 `LatentGoalBCPolicy`。

验证：

- train loss 下降
- val loss 有限且稳定
- 预测动作分布大致覆盖 expert action 分布

### Milestone 3：环境评估

运行：

```text
python eval_latent_bc.py --config-name=pusht.yaml policy_ckpt=experiments/YYYY-MM-DD_pusht_latent_bc/policy.pt
```

记录：

- success rate
- episode successes
- wall time
- 如果可用，保存 qualitative videos

### Milestone 4：对比报告

生成：

```text
experiments/YYYY-MM-DD_pusht_latent_bc/comparison.md
```

包含：

- CEM 命令和指标
- Latent BC 命令和指标
- success-rate ratio
- speedup ratio
- 失败模式观察
- 下一步决策：继续 BC、加 reranking，或转向 diffusion prior

## 9. 风险与诊断

### 风险：BC 平均化多模态专家动作

现象：

- action MSE 不高，但环境成功率差
- 动作平滑但犹豫、不果断

下一步：

- 从确定性 BC 转向 Gaussian、mixture density 或 diffusion policy。

### 风险：CLS latent 过度压缩

现象：

- policy 分不清需要不同动作的状态
- latent nearest-neighbor 检索中出现视觉或动作上差异很大的状态

下一步：

- 加入短历史 latent
- 如果可用，加入 proprio/state 特征
- 后续在不改 LeWM 权重的前提下暴露 dense ViT patch tokens

### 风险：goal relabeling 太难

现象：

- 大 `G` 下训练 loss 不稳定
- policy 只对短 horizon goal 有效

下一步：

- 对 `G` 做 curriculum
- 每条轨迹采样多个未来 goal
- 按 goal horizon bucket 分开评估

### 风险：action 标准化不一致

现象：

- 预测动作 scale 错误
- normalized loss 低，但环境行为很乱

下一步：

- 复用 `eval.py` 中的 `StandardScaler` 逻辑
- 在 policy checkpoint 里显式保存 action normalization statistics

## 10. 决策标准

第一轮 PushT 结果后：

```text
如果 BC success >= 80% CEM success，且 latency 明显更低：
  继续扩大 BC 架构和评估规模。

如果 BC success 是 CEM 的 40%-80%：
  加 one-shot LeWM reranking 或 CEM distillation。

如果 BC success 低于 CEM 的 40%：
  先检查数据和 normalization，再转向 diffusion expert prior。
```

第一阶段不必要求 BC 一定超过 CEM。更现实的结论是：

```text
Frozen LeWM latents 可以支撑一个低成本 goal-conditioned policy；
它用一部分 planning optimality 换取显著更低的 inference latency，
后续可以通过 expert prior、one-shot LeWM verification 或 diffusion policy 继续增强。
```
