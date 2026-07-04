# LeWM Latent-Aware ACT 计划书

本文是一个从零构建的方案计划，不继承现有 `latent_bc.py` 的实现思路。目标是把 ACT 的 action chunking 思想和 LeWM 的 latent dynamics 结合起来，让 policy 不只预测动作，还对短期未来 latent 负责。

## 1. 问题定义

当前单纯的 goal-conditioned BC 有三个弱点：

1. 只学 `action`，不学状态演化。
2. 直接从 `z_t` 追 `z_g` 太远，长 horizon 容易失真。
3. open-loop chunk 一旦起步偏了，后续动作会继续偏。

因此我们要做的不是“更大一点的 policy”，而是一个 **latent-aware action chunk policy**：

```text
输入:  z_t, z_g, delta_z
输出:  action chunk + short-term latent prediction
```

其中：

```text
delta_z = z_g - z_t
```

我们希望 policy 学到两件事：

- `a_{t:t+K}` 怎么走。
- 这段动作会把世界带到什么短期 latent 状态。

这样 LeWM 的世界模型能力才真正进入控制回路。

## 2. 原理

### 2.1 基本形式

最基础的 ACT-like policy 形式是：

```text
pi(z_t, z_g, delta_z) -> a_{t:t+K}
```

其中 `K` 是动作 chunk 长度，默认建议 `K=5`。

### 2.2 加入短期 latent 监督

在 action chunk 之外，再预测一个或多个短期 future latent：

```text
pi(z_t, z_g, delta_z) -> a_{t:t+K}, z_{t+H}
```

或者：

```text
pi(z_t, z_g, delta_z) -> a_{t:t+K}, z_{t+1}, z_{t+2}, ..., z_{t+H}
```

其中 `H` 建议先取 5，和 action chunk 同量级。

### 2.3 训练目标

总 loss 建议拆成三部分：

```text
L_total = L_action + lambda_latent * L_latent + lambda_reg * L_reg
```

默认第一版：

```text
L_action = MSE(a_pred, a_expert)
L_latent = MSE(z_pred, z_true_future)
L_reg    = 0
```

如果后续引入多模态版本，再加：

```text
L = L_action + lambda_latent * L_latent + beta * KL(q(style) || N(0, I))
```

但第一版不急着上 CVAE。

### 2.4 推理时闭环

推理不做长 open-loop，而是：

```text
observe -> encode -> predict short chunk -> execute 1 step -> re-observe
```

建议第一版只执行 1 步再重算，这样最稳。

## 3. 模块架构思路

建议从零拆成 5 个模块，不直接在旧 BC 代码上叠。

### 3.1 数据模块

职责：

- 从 HDF5 expert dataset 取轨迹。
- 用 LeWM encoder 编码当前帧、目标帧、短期未来帧。
- 生成 action chunk 监督。
- 保存紧凑 latent 数据集。

建议字段：

```text
z_t
z_g
z_h1
z_h2
...
delta_z
action_chunk
episode
step
goal_step
metadata
```

### 3.2 模型模块

职责：

- 接收 latent 条件。
- 用 Transformer 或 ACT-style decoder 生成 action chunk。
- 用 latent head 预测短期 future latent。

建议输出：

```text
action_chunk: [K, action_dim]
latent_pred:  [latent_dim] or [H, latent_dim]
```

### 3.3 损失模块

职责：

- action regression loss。
- latent auxiliary loss。
- 可选 smoothness / magnitude / KL。

### 3.4 Policy / Inference 模块

职责：

- 维护 action buffer。
- 每步重新编码当前帧。
- 支持 receding horizon。
- 支持 temporal ensemble（后续）。

### 3.5 Experiment Runner 模块

职责：

- train。
- eval。
- ablation。
- archive。

## 4. Python 模块设计思路

建议新建一个独立包，例如：

```text
act_lewm/
  __init__.py
  data/
    build_dataset.py
    dataset.py
  models/
    latent_act.py
    latent_heads.py
  losses.py
  policy.py
  train.py
  eval.py
  config.py
  utils.py
scripts/
  build_latent_act_dataset.py
  train_latent_act.py
  eval_latent_act.py
```

### 4.1 `act_lewm/data/build_dataset.py`

建议函数：

```python
build_latent_act_dataset(...)
encode_latent(...)
collect_future_latents(...)
collect_action_chunk(...)
```

职责：

- 读取 `stable_worldmodel` 的 HDF5 数据。
- 用冻结 LeWM encoder 编码。
- 生成训练样本。

### 4.2 `act_lewm/data/dataset.py`

建议类：

```python
class LatentACTDataset(torch.utils.data.Dataset):
    ...
```

职责：

- 只负责索引和返回张量。
- 不做训练逻辑。

### 4.3 `act_lewm/models/latent_act.py`

建议类：

```python
class LatentACTPolicy(nn.Module):
    ...

class LatentQueryBlock(nn.Module):
    ...
```

如果是 Transformer 版：

- 输入 token = `z_t, z_g, delta_z, q_1..q_K`
- 输出 token = action positions + latent head

如果是 ACT-style CVAE 版：

- encoder 估计 style latent。
- decoder 生成 action chunk。

### 4.4 `act_lewm/losses.py`

建议函数：

```python
compute_action_loss(...)
compute_latent_loss(...)
compute_total_loss(...)
```

职责：

- 所有损失统一入口。
- 便于做 ablation。

### 4.5 `act_lewm/policy.py`

建议类：

```python
class LatentACTWorldPolicy:
    ...
```

职责：

- 接收当前 `info`。
- 编码 `z_t` 和 `z_g`。
- 调用 policy。
- 执行 action chunk。
- 支持 `receding_horizon=1`。
- 后续支持 temporal ensemble 和 rerank。

### 4.6 `act_lewm/train.py`

职责：

- 训练 policy。
- 记录 action loss、latent loss、eval loss。
- 保存 checkpoint 和 config。

### 4.7 `act_lewm/eval.py`

职责：

- 跑 PushT / 其他 downstream task。
- 评估 success rate、wall time、latent rollout error。
- 输出视频和结果。

## 5. 参数设计思路

### 5.1 数据参数

```text
G_goal = 25
K_action = 5
H_latent = 5
```

建议解释：

- `G_goal`：goal offset。
- `K_action`：一次输出的动作长度。
- `H_latent`：短期 latent 预测的 horizon。

### 5.2 模型参数

```text
architecture = transformer
hidden_dim = 512
depth = 4
num_heads = 8
dropout = 0.1
```

解释：

- `architecture`：先用 Transformer，再和 MLP 对照。
- `hidden_dim`：token width。
- `depth`：decoder/encoder 层数。
- `num_heads`：attention heads。
- `dropout`：防止过拟合。

### 5.3 loss 参数

```text
lambda_latent = 0.1
lambda_reg = 0.0
```

第一轮建议：

- `lambda_latent` 从 `0.1` 起。
- 如果 latent loss 太弱，再升到 `0.3` 或 `1.0`。
- 如果 latent loss 压过 action loss，就降一点。

### 5.4 推理参数

```text
receding_horizon = 1
temporal_ensemble = false
candidate_num = 1
```

第一轮先稳：

- 每步重算。
- 不开 temporal ensemble。
- 不开多候选 rerank。

### 5.5 训练参数

```text
batch_size = 256
epochs = 100
lr = 3e-4
weight_decay = 1e-4
train_split = 0.9
```

数据量建议：

- smoke: `128`
- quick compare: `10000`
- first real run: `100000`
- larger run: `128000` or full

## 6. 实验要跑什么

### 6.1 必做 baseline

1. `MLP latent BC`
2. `Transformer latent BC`

### 6.2 必做 latent-aware ablation

3. `Transformer + latent auxiliary loss`
4. `Transformer + receding_horizon=1`
5. `Transformer + latent auxiliary + receding_horizon=1`

### 6.3 进阶实验

6. `Transformer + temporal ensemble`
7. `Transformer + LeWM reranking`
8. `Transformer + subgoal latent`
9. `Transformer + CVAE style latent`

### 6.4 推荐实验顺序

#### Stage A: 通管线

- 构建 128-sample smoke dataset。
- 训练 1-2 epoch。
- 跑 `eval.num_eval=2`。

#### Stage B: 看 action chunk 是否有用

- MLP baseline。
- Transformer baseline。

#### Stage C: 加 latent auxiliary loss

- 只改 loss。
- 看 latent loss 是否改善环境成功率。

#### Stage D: 闭环执行

- `receding_horizon=1`。
- 看是否比 open-loop 更稳。

#### Stage E: 再往上加

- temporal ensemble。
- rerank。
- subgoal。
- CVAE。

## 7. 评价指标

必须同时看三类指标：

### 7.1 环境指标

- success rate
- episode successes
- wall time

### 7.2 行为指标

- action MSE
- action smoothness
- action magnitude

### 7.3 latent 指标

- latent prediction MSE
- rollout terminal latent error
- short-term latent consistency

如果 latent loss 降了，但环境成功率没起来，说明只学会 latent 还不够，可能需要 rerank 或 subgoal。

## 8. 结论

这套方案的核心不是“更大模型”，而是：

```text
action chunking + short-term latent prediction + closed-loop replanning
```

它比纯 action BC 更像 ACT，也更能把 LeWM 的 world model 能力用起来。
