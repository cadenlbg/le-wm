# Subgoal Only Predictor

这个 baseline 只训练 Transformer 预测未来 latent trajectory，不预测 action。

目标：

```text
z_t, z_g -> z_hat_{t+1:t+T}
```

输入：

```text
z_t: 当前 latent
z_g: goal latent
```

输出：

```text
pred_z_h_seq: [B, T, latent_dim]
```

训练 loss：

```text
L = MSE(pred_z_h_seq, z_h_seq)
```

可选项：

```text
lambda_terminal * MSE(pred_z_h_seq[:, -1], z_h_seq[:, -1])
lambda_smooth * smoothness(pred_z_h_seq)
```

训练命令：

```bash
CUDA_VISIBLE_DEVICES=3 python -B -m latent_subgoal_act.subgoal_only.train \
  dataset=pusht_fixed_g25_k25_t25_ms_128k_train.pt \
  output=subgoal_only_g25_T5_ms128k \
  subgoal_horizon=5 \
  train.epochs=200 \
  loader.batch_size=256 \
  device=cuda
```

如果要训练完整 T=25：

```bash
CUDA_VISIBLE_DEVICES=3 python -B -m latent_subgoal_act.subgoal_only.train \
  dataset=pusht_fixed_g25_k25_t25_ms_128k_train.pt \
  output=subgoal_only_g25_T25_ms128k \
  subgoal_horizon=25 \
  train.epochs=200 \
  loader.batch_size=256 \
  device=cuda
```

用途：

```text
1. 判断 z_t,z_g 是否足够预测未来 latent trajectory。
2. 和 action head 解耦，单独观察 subgoal prediction 的 train/val gap。
3. 后续可以把这个 predictor 的输出接到 action prior、diffusion prior 或 CEM target 中。
```

