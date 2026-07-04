# LeWM + ACT 下游实验构思

本文档基于 ACT 论文 `papers/act_action_chunking_transformers_2304.13705.pdf`，讨论如何把 Action Chunking with Transformers 融入当前 LeWM latent BC 下游任务。

## 1. 当前问题

我们目前的 latent BC 形式是：

```text
z_t = LeWM.encoder(o_t)
z_g = LeWM.encoder(o_{t+G})
delta_z = z_g - z_t
policy(z_t, z_g, delta_z) -> a_{t:t+K}
```

这个方法已经能跑通，但 MLP 版本在 PushT `eval.num_eval=50` 上只有 `1/50` 成功。主要问题是：

- policy 直接预测 action，没有真正使用 LeWM 的 latent dynamics / world prediction 能力。
- `z_g` 是较远目标，`G=25` 时直接从 `z_t` 到 `z_g` 可能太难。
- action chunk 是 open-loop 片段，如果前几步偏了，后续动作会继续偏。
- deterministic BC 容易平均化多模态专家动作。

ACT 的贡献不是“只把 MLP 换成 Transformer”，而是三个核心设计：

```text
1. action chunking: 一次预测未来 K 个动作
2. temporal ensembling: 每一步都重新预测 chunk，并融合重叠预测
3. CVAE style variable: 建模 demonstration action chunk 的多模态性
```

这些点都可以和 LeWM latent 表示结合。

## 2. ACT 原始方法对我们的启发

ACT 原论文中，policy 学的是：

```text
pi(a_{t:t+K} | observation_t)
```

而我们可以改成：

```text
pi(a_{t:t+K} | z_t, z_g, delta_z)
```

甚至更进一步，把 LeWM 的 rollout 验证也接进来：

```text
ACT policy 生成 action chunk candidates
LeWM rollout 预测 terminal latent
选择更接近 z_g 的 action chunk
```

这样 ACT 提供 action prior，LeWM 提供 verifier。它比纯 BC 更像 planner，比完整 CEM 更便宜。

## 3. 实验方向 A：Latent ACT Policy

这是最直接版本，替代当前 `LatentGoalBCPolicy`。

### 输入

```text
condition tokens:
  z_t
  z_g
  delta_z = z_g - z_t

action query tokens:
  q_1, q_2, ..., q_K
```

### 输出

```text
a_{t:t+K}
```

### 训练目标

第一版仍用标准化 action chunk 的监督：

```text
L = || pi(z_t, z_g, delta_z) - a_expert_{t:t+K} ||^2
```

这已经在当前 Transformer policy 里基本实现。它对应 ACT 的 action chunking + Transformer decoder 思想，但还没有 temporal ensembling 和 CVAE。

### 对比

```text
MLP latent BC
Transformer latent BC
Transformer latent BC + receding_horizon=1
```

如果 Transformer 明显超过 MLP，说明 action chunk sequence model 有价值。

## 4. 实验方向 B：ACT Temporal Ensembling

当前评估逻辑是：

```text
预测 action chunk
执行前 receding_horizon 个动作
再预测
```

ACT 的 temporal ensembling 更细：每个环境 step 都重新预测一个 chunk，因此同一个实际执行时间点可能收到多个历史 chunk 的预测。然后对这些预测做加权平均。

假设第 `t` 步预测：

```text
[a_t, a_{t+1}, ..., a_{t+K-1}]
```

第 `t+1` 步又预测：

```text
[a_{t+1}, a_{t+2}, ..., a_{t+K}]
```

那么实际执行 `a_{t+1}` 时，可以融合来自多个 chunk 的预测：

```text
a_exec(t+1) = weighted_average(
  prediction_from_step_t_for_t+1,
  prediction_from_step_t+1_for_t+1
)
```

### 为什么适合我们

这可以缓解 BC action chunk 的 open-loop 问题：

- 每一步都看新 observation。
- 动作不会突然从一个 chunk 切到另一个 chunk。
- 对 PushT 这种接触任务，平滑性可能很重要。

### 实现建议

在 `LatentBCWorldPolicy` 里新增模式：

```text
execution_mode=first
execution_mode=chunk
execution_mode=temporal_ensemble
```

其中：

- `first`：每步只执行当前 chunk 的第一个动作。
- `chunk`：执行前 `receding_horizon` 个动作。
- `temporal_ensemble`：每步预测 chunk，把未来动作放进 buffer，执行当前时刻的加权平均。

第一版权重可以用 ACT 的指数衰减思想：

```text
w_i = exp(-m * age_i)
```

其中 `age_i` 表示这个预测来自几步之前的 chunk。

## 5. 实验方向 C：Latent ACT-CVAE

ACT 用 CVAE 建模 action chunk 的多模态性。原因是同一个 observation 下，示范者可能有不同风格：

```text
快推 / 慢推
绕左边 / 绕右边
短暂停顿 / 连续推动
```

Deterministic BC 会平均这些动作，导致看起来 loss 不差，但环境行为犹豫。

### 训练时

增加一个 action encoder：

```text
q_phi(style | z_t, z_g, action_chunk)
```

然后 decoder：

```text
pi_theta(action_chunk | z_t, z_g, style)
```

loss：

```text
L = reconstruction_loss + beta * KL(q_phi(style) || N(0, I))
```

### 推理时

第一版可以像 ACT 一样取 prior mean：

```text
style = 0
```

也可以采样多个 style：

```text
style_i ~ N(0, I)
```

生成多个 action chunks，再用 LeWM rollout rerank。

### 为什么适合 LeWM

CVAE 负责生成多种可能动作，LeWM 负责判断哪种动作更靠近 `z_g`：

```text
style_i -> action_i
rollout(z_t, action_i) -> z_pred_i
choose min || z_pred_i - z_g ||
```

这比 deterministic Transformer 更接近 CEM 的多候选搜索。

## 6. 实验方向 D：ACT Prior + LeWM Reranking

这是我最推荐的 LeWM 结合方式。

### 流程

```text
1. 当前观测编码为 z_t
2. goal 编码为 z_g
3. ACT policy 生成 N 个候选 action chunks
4. LeWM rollout 每个 action chunk
5. 用 terminal latent 距离打分
6. 执行最优 chunk 的第一个动作或 temporal ensemble 动作
```

评分：

```text
score_i = || rollout(z_t, action_i)_terminal - z_g ||^2
```

### 候选动作来源

可以从简单到复杂：

```text
1. action_pred + Gaussian noise
2. dropout 多次 forward
3. CVAE style sampling
4. CEM distillation policy 输出 top-N candidates
```

### 优点

- Transformer 不需要一次猜中。
- LeWM 的 world prediction 能力真正参与决策。
- 比完整 CEM 便宜，因为候选来自 learned prior，不需要多轮分布拟合。

## 7. 实验方向 E：Subgoal ACT

不要直接预测动作，而是先预测中间 latent subgoal：

```text
ACT_subgoal(z_t, z_g) -> z_mid
```

然后低层 policy：

```text
ACT_low(z_t, z_mid) -> a_{t:t+K}
```

监督信号来自同一条 expert trajectory：

```text
z_mid = encoder(o_{t+5}) 或 encoder(o_{t+10})
```

这样把 `G=25` 拆成更短目标，可能更适合 BC。

## 8. 推荐实验顺序

按实现难度和信息量排序：

### Step 1：Transformer ACT-style baseline

已经基本可跑：

```bash
python train_latent_bc.py \
  dataset=latent_bc_datasets/pusht_g25_k5_128k.pt \
  output=2026-07-04_pusht_latent_act_transformer_128k \
  model.architecture=transformer \
  model.hidden_dim=512 \
  model.depth=4 \
  model.num_heads=8 \
  train.epochs=100
```

### Step 2：每步重算

```bash
python eval_latent_bc.py \
  policy_ckpt=2026-07-04_pusht_latent_act_transformer_128k/policy.pt \
  eval.num_eval=50 \
  plan_config.receding_horizon=1
```

### Step 3：Temporal Ensemble

新增 `execution_mode=temporal_ensemble`，测试：

```text
Transformer + temporal ensemble
```

对比：

```text
receding_horizon=1
receding_horizon=5
temporal_ensemble
```

### Step 4：ACT Prior + LeWM Reranking

新增 rerank 逻辑，先用 deterministic action 加噪声：

```text
N = 8 or 16 candidates
noise_std = 0.1 or 0.2
```

如果成功率有提升，说明 LeWM verifier 是关键。

### Step 5：Latent ACT-CVAE

加入 style latent，让 policy 生成多模态 action chunks，再配合 LeWM reranking。

## 9. 成功标准

当前 MLP BC 是：

```text
1 / 50 = 2%
```

建议第一阶段目标：

```text
Transformer ACT-style >= 5/50
Temporal ensemble >= Transformer baseline
LeWM reranking >= temporal ensemble
```

如果 LeWM reranking 明显提升，即使 Transformer 单独不强，也说明方向对了：policy 负责提候选，world model 负责验候选。

## 10. 小结

ACT 对我们的最大启发不是“用 Transformer 替换 MLP”，而是：

```text
action chunking + overlapping predictions + multimodal action generation
```

而 LeWM 可以补上 ACT 原始方法里没有显式使用的 world-model verification：

```text
ACT generates action chunks
LeWM predicts their latent consequences
choose the chunk that best approaches z_g
```

这条路线比纯 latent BC 更像 planner，也更有机会利用 LeWM 主干的真正价值。
