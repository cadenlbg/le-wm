# Latent BC Transformer Policy 说明

这份文档说明当前 latent BC 中 Transformer policy 的设计、为什么它可能比 MLP 更合适、怎么训练和评估，以及第一轮建议怎么调参数。

## 1. 为什么考虑 Transformer

第一版 MLP policy 的输入是：

```text
concat(z_t, z_g, z_g - z_t)
```

然后一次性输出：

```text
a_{t:t+K}
```

这个设计简单、稳定、参数少，但它把所有条件压成一个向量，action chunk 内部的第 1 步、第 2 步、...、第 K 步之间没有显式结构。PushT 这类任务里，动作序列通常不是独立的单步动作，而是一个短程控制片段；如果只用 MLP，模型可能更容易学成平均动作。

Transformer policy 的目标是让模型显式看到两类 token：

- 条件 token：当前 latent、目标 latent、latent 差值。
- 动作 query token：每一个未来动作 step 一个 query。

## 2. Token 设计

当前实现位于 `latent_bc.py` 的 `LatentGoalBCPolicy`。

Transformer 输入序列是：

```text
[z_t, z_g, delta_z, q_1, q_2, ..., q_K]
```

其中：

```text
delta_z = z_g - z_t
```

前 3 个 token 是条件，后 K 个 token 是可学习 action queries。Transformer Encoder 处理整段 token 后，只取 action query 对应的输出 token，再映射成动作：

```text
action_tokens -> Linear -> [K, action_dim]
```

直觉上：

- `z_t` 告诉模型当前在哪里。
- `z_g` 告诉模型目标在哪里。
- `delta_z` 给出从当前到目标的方向。
- `q_i` 让模型为第 i 个动作位置生成不同的动作。

这样比把 `z_t/z_g/delta_z` 简单拼接后丢给 MLP 更有结构。

## 3. 当前实现参数

训练脚本里通过 `model.*` 控制结构：

```bash
model.architecture=transformer
model.hidden_dim=512
model.depth=4
model.num_heads=8
model.dropout=0.1
```

参数含义：

- `model.architecture`：`mlp` 或 `transformer`。
- `model.hidden_dim`：Transformer token hidden size。
- `model.depth`：Transformer Encoder 层数。
- `model.num_heads`：attention heads 数。
- `model.dropout`：attention/MLP dropout。

输出动作维度由 dataset metadata 决定：

```text
action_horizon = K
action_dim = PushT action dimension
```

因此只要训练和评估用同一个 `policy.pt`，不需要手动指定 `K`。

## 4. 推荐训练命令

默认使用完整 latent dataset：

```bash
cd /data/zflin/lewm_re/le-wm
conda activate lewm
export STABLEWM_HOME=/data/zflin/lewm_re/stablewm_data
export LEWM_EXPERIMENTS_DIR=/data/zflin/lewm_re/experiments

python train_latent_bc.py \
  dataset=latent_bc_datasets/pusht_g25_k5.pt \
  output=2026-07-03_pusht_latent_bc_transformer \
  model.architecture=transformer \
  model.hidden_dim=512 \
  model.depth=4 \
  model.num_heads=8 \
  model.dropout=0.1 \
  train.epochs=100
```

如果训练不稳定或 val loss 抖动明显，可以先保守一点：

```bash
python train_latent_bc.py \
  dataset=latent_bc_datasets/pusht_g25_k5.pt \
  output=2026-07-03_pusht_latent_bc_transformer_small \
  model.architecture=transformer \
  model.hidden_dim=256 \
  model.depth=3 \
  model.num_heads=4 \
  model.dropout=0.1 \
  train.epochs=100
```

## 5. 推荐评估命令

先做小评估：

```bash
python eval_latent_bc.py \
  policy_ckpt=2026-07-03_pusht_latent_bc_transformer/policy.pt \
  eval.num_eval=2
```

正式评估：

```bash
python eval_latent_bc.py \
  policy_ckpt=2026-07-03_pusht_latent_bc_transformer/policy.pt \
  eval.num_eval=50
```

如果怀疑一次执行 5 个动作导致闭环太弱，可以试每步重算：

```bash
python eval_latent_bc.py \
  policy_ckpt=2026-07-03_pusht_latent_bc_transformer/policy.pt \
  eval.num_eval=50 \
  plan_config.receding_horizon=1
```

## 6. 第一轮实验矩阵

建议和 MLP 做直接对比：

| ID | 架构 | G | K | hidden | depth | heads | receding | 目的 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-B0 | mlp | 25 | 5 | 512 | 3 | - | 5 | 当前 baseline |
| TR-B0 | transformer | 25 | 5 | 512 | 4 | 8 | 5 | 测试架构提升 |
| TR-B1 | transformer | 25 | 5 | 512 | 4 | 8 | 1 | 测试更强闭环 |
| TR-S | transformer | 25 | 5 | 256 | 3 | 4 | 5 | 小模型防过拟合 |

如果 `G=25` 仍然太难，可以单独构建短 goal dataset：

```bash
python scripts/build_latent_bc_dataset.py \
  eval.goal_offset_steps=10 \
  output_dataset=latent_bc_datasets/pusht_g10_k5.pt
```

然后训练：

```bash
python train_latent_bc.py \
  dataset=latent_bc_datasets/pusht_g10_k5.pt \
  output=2026-07-03_pusht_latent_bc_transformer_g10 \
  model.architecture=transformer \
  model.hidden_dim=512 \
  model.depth=4 \
  model.num_heads=8
```

## 7. 结果判断

先看训练指标：

- `train/bc_mse` 是否明显下降。
- `val/bc_mse` 是否比 MLP 更低。
- train/val gap 是否过大。

再看环境指标：

- success rate 是否超过 MLP 的 `1/50`。
- 视频里动作是否更果断、更连贯。
- 每步重算 `receding_horizon=1` 是否提升成功率。

可能出现的情况：

- val MSE 低但成功率差：说明 action MSE 不足以刻画闭环控制，优先看视频和 action scale。
- train MSE 低、val MSE 高：Transformer 过拟合，减小 `hidden_dim/depth` 或增大 dropout。
- MSE 和成功率都差：可能是 `G=25` 太难，尝试 `G=10`。

## 8. 归档命令

```bash
bash scripts/archive_latent_bc_experiment.sh \
  2026-07-03_pusht_latent_bc_transformer \
  "python eval_latent_bc.py policy_ckpt=2026-07-03_pusht_latent_bc_transformer/policy.pt eval.num_eval=50"
```
