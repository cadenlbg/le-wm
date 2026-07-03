# Latent BC 实验说明书

这份说明书对应当前仓库里的 latent goal-conditioned BC 方案，默认任务是 `PushT`。

## 1. 依赖与准备

所有命令默认在远端 SSH 的 Linux 环境执行：

```bash
cd /data/zflin/lewm_re/le-wm
conda activate lewm
export STABLEWM_HOME=/data/zflin/lewm_re/stablewm_data
export HF_ENDPOINT=https://hf-mirror.com
```

BC 相关运行产物默认不写入 `le-wm` 仓库。脚本会把相对实验路径解析到：

```text
/data/zflin/lewm_re/experiments
```

Hydra 日志默认写到：

```text
/data/zflin/lewm_re/experiments/hydra/<job_name>/YYYY-MM-DD/HH-MM-SS/
```

如果需要临时改实验根目录，可以设置：

```bash
export LEWM_EXPERIMENTS_DIR=/data/zflin/lewm_re/experiments
```

先确认环境里能正常导入这些包：

```bash
python -c "import hydra, torch, stable_worldmodel"
```

需要的数据和 checkpoint：

- `pusht_expert_train.h5`
- `policy=pusht/lewm` 对应的 loader 缓存

当前远端已经跑通 PushT 的 CEM baseline，因此基础 HDF5 数据和 LeWM checkpoint 已经可用。checkpoint 使用当前缓存布局：

```text
$STABLEWM_HOME/checkpoints/models--pusht--lewm/weights.pt
$STABLEWM_HOME/checkpoints/models--pusht--lewm/config.json
```

BC 脚本继续复用原始 `eval.py` 的 HDF5 数据加载逻辑，不要求切换到新版 `load_dataset()`，也不要求生成旧式 `$STABLEWM_HOME/pusht/lewm_object.ckpt`。

## 2. 生成 latent 数据集

默认配置：

- `G = 25`
- `K = 5`
- 输出：`/data/zflin/lewm_re/experiments/latent_bc_datasets/pusht_g25_k5.pt`

运行命令：

```bash
python scripts/build_latent_bc_dataset.py
```

建议先做小样本 smoke：

```bash
python scripts/build_latent_bc_dataset.py max_samples=128
```

如果需要显式指定 LeWM checkpoint：

```bash
python scripts/build_latent_bc_dataset.py lewm_policy=pusht/lewm max_samples=128
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
python train_latent_bc.py dataset=latent_bc_datasets/pusht_g25_k5.pt output=2026-07-03_pusht_latent_bc
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
python eval_latent_bc.py policy_ckpt=2026-07-03_pusht_latent_bc/policy.pt
```

建议先跑小评估：

```bash
python eval_latent_bc.py policy_ckpt=2026-07-03_pusht_latent_bc/policy.pt eval.num_eval=2
```

`eval_latent_bc.py` 会优先使用 policy checkpoint metadata 里记录的 `model_policy`。如果需要手动覆盖：

```bash
python eval_latent_bc.py policy_ckpt=2026-07-03_pusht_latent_bc/policy.pt lewm_policy=pusht/lewm eval.num_eval=2
```

评估结果会写到 checkpoint 目录下，并打印：

- success rate
- episode successes
- evaluation wall time

## 5. 对比 CEM baseline

PushT CEM baseline 已经在远端跑通。需要重新确认时再跑：

```bash
python eval.py --config-name=pusht.yaml policy=pusht/lewm eval.num_eval=50
```

然后跑 latent BC：

```bash
python eval_latent_bc.py policy_ckpt=2026-07-03_pusht_latent_bc/policy.pt eval.num_eval=50
```

对比重点：

- success rate
- wall time
- 单步推理开销
- 失败模式

## 6. 推荐实验顺序

1. 确认当前 shell 已设置 `STABLEWM_HOME` 并在 `/data/zflin/lewm_re/le-wm`。
2. 构建小样本 latent dataset：`python scripts/build_latent_bc_dataset.py max_samples=128`。
3. 做 1-2 epoch 小训练确认管线通。
4. 跑 `eval.num_eval=2` 的小评估。
5. 构建完整 latent dataset、完整训练、完整评估，并和已跑通的 CEM baseline 对比。

## 7. 常见问题

- `hydra` / `stable_worldmodel` 导入失败：先安装仓库依赖
- dataset 构建报 checkpoint 找不到：检查 `$STABLEWM_HOME/checkpoints/models--pusht--lewm/weights.pt` 和 `config.json`
- dataset 构建报 HDF5 找不到：检查 `$STABLEWM_HOME/datasets/pusht_expert_train.h5`
- 评估报动作 shape 不匹配：优先检查 `K` 和 action normalization 是否与 dataset 一致
- 成功率低但 loss 正常：先看 action 标准化、goal offset 和 episode split
