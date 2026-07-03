# Latent BC 实验说明书

这份说明书对应当前仓库里的 latent goal-conditioned BC 方案，默认任务是 `PushT`。

## 1. 依赖与准备

先确认环境里能正常导入这些包：

```bash
python -c "import hydra, torch, stable_worldmodel"
```

需要的数据和 checkpoint：

- `pusht_expert_train.h5`
- `pusht/lewm_object.ckpt`

它们都应放在 `$STABLEWM_HOME/` 对应位置。

## 2. 生成 latent 数据集

默认配置：

- `G = 25`
- `K = 5`
- 输出：`experiments/latent_bc_datasets/pusht_g25_k5.pt`

运行命令：

```bash
python scripts/build_latent_bc_dataset.py
```

建议先做小样本 smoke：

```bash
python scripts/build_latent_bc_dataset.py max_samples=128
```

检查项：

- `z_t`, `z_g`, `delta_z`, `action` 形状正确
- 没有 `NaN`
- `goal_step = step + 25`
- `action` 是标准化后的 chunk

## 3. 训练 latent BC

默认输出目录：

```text
experiments/YYYY-MM-DD_pusht_latent_bc/
```

运行命令：

```bash
python train_latent_bc.py
```

如果要指定数据集和输出目录：

```bash
python train_latent_bc.py dataset=experiments/latent_bc_datasets/pusht_g25_k5.pt output=experiments/2026-07-03_pusht_latent_bc
```

训练产物：

- `policy.pt`
- `config.yaml`
- `metrics.jsonl`

检查项：

- train loss 下降
- val loss 稳定
- `policy.pt` 可正常加载

## 4. 评估 latent BC

运行命令：

```bash
python eval_latent_bc.py policy_ckpt=experiments/2026-07-03_pusht_latent_bc/policy.pt
```

建议先跑小评估：

```bash
python eval_latent_bc.py policy_ckpt=experiments/2026-07-03_pusht_latent_bc/policy.pt eval.num_eval=2
```

评估结果会写到 checkpoint 目录下，并打印：

- success rate
- episode successes
- evaluation wall time

## 5. 对比 CEM baseline

建议先跑一份参考 CEM：

```bash
python eval.py --config-name=pusht.yaml policy=pusht/lewm eval.num_eval=50
```

然后跑 latent BC：

```bash
python eval_latent_bc.py policy_ckpt=experiments/2026-07-03_pusht_latent_bc/policy.pt eval.num_eval=50
```

对比重点：

- success rate
- wall time
- 单步推理开销
- 失败模式

## 6. 推荐实验顺序

1. 先跑 CEM reference
2. 再构建 latent dataset
3. 做 1-2 epoch 小训练确认管线通
4. 跑 `eval.num_eval=2` 的小评估
5. 最后跑完整评估并记录结果

## 7. 常见问题

- `hydra` / `stable_worldmodel` 导入失败：先安装仓库依赖
- dataset 构建报 checkpoint 找不到：检查 `$STABLEWM_HOME/pusht/lewm_object.ckpt`
- 评估报动作 shape 不匹配：优先检查 `K` 和 action normalization 是否与 dataset 一致
- 成功率低但 loss 正常：先看 action 标准化、goal offset 和 episode split
