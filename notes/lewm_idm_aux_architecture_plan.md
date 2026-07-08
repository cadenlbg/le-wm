# LeWM 架构解析与 IDM Auxiliary 实验计划

本文档用于梳理当前 LeWM 的原始架构、训练目标、planning 流程，以及一个不改动原始代码的新实验方向：

\[
D_\psi(z_t, z_{t+1}) \rightarrow a_t
\]

也就是在 LeWM 表征训练中加入一个 inverse dynamics decoder，用相邻 latent 恢复 embedding 前的原始 action block。

## 1. 当前 LeWM 的核心思想

LeWM 是一个 latent world model。它不直接预测未来图像，而是把图像编码到 latent 空间，在 latent 空间中学习动作条件下的未来状态预测：

\[
o_t \xrightarrow{\text{encoder + projector}} z_t
\]

\[
(z_t, a_t) \rightarrow \hat{z}_{t+1}
\]

其中 \(o_t\) 是图像观测，\(a_t\) 是环境动作，\(z_t\) 是 LeWM latent。训练和 planning 都主要发生在 latent 空间中。

当前模型主体是：

```text
JEPA(
  encoder,
  projector,
  action_encoder,
  predictor,
  pred_proj
)
```

对应源码：

```text
jepa.py
module.py
train.py
config/train/lewm.yaml
config/train/model/lewm.yaml
config/train/data/pusht.yaml
```

## 2. 图像如何被 encode

输入图像序列：

```text
pixels: (B, T, C, H, W)
```

当前 PushT 配置：

```yaml
img_size: 224
embed_dim: 192
```

图像 encoder 配置：

```yaml
encoder:
  _target_: stable_pretraining.backbone.utils.vit_hf
  size: tiny
  patch_size: 14
  image_size: 224
  pretrained: false
  use_mask_token: false
```

代码中先把 batch 和时间维展平：

```python
pixels = rearrange(pixels, "b t ... -> (b t) ...")
output = self.encoder(pixels, interpolate_pos_encoding=True)
pixels_emb = output.last_hidden_state[:, 0]
```

也就是取 ViT 的 CLS token。随后经过 projector：

```yaml
projector:
  _target_: module.MLP
  input_dim: 192
  output_dim: 192
  hidden_dim: 2048
  norm_fn: BatchNorm1d
```

得到 LeWM latent：

```text
emb / z: (B, T, 192)
```

因此，每一帧图像最终被编码成一个 192 维 latent。

## 3. Action 如何被 encode

LeWM planning 搜索和环境执行的对象都是 embedding 前的原始 action block，不是 action embedding。

PushT 原始 action 维度为：

```text
action_dim = 2
```

当前 PushT 训练配置：

```yaml
dataset:
  frameskip: 5
```

训练时会动态设置：

```python
cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
```

所以 PushT 中：

```text
action_encoder.input_dim = 5 * 2 = 10
```

也就是说，每个 LeWM 时间步的 action 是连续 5 个低层环境动作拼成的 action block：

```text
raw action block: (B, T, 10)
```

action encoder 配置：

```yaml
action_encoder:
  _target_: module.Embedder
  input_dim: 10
  emb_dim: 192
```

`Embedder` 结构：

```text
raw action block
  -> Conv1d(kernel_size=1)
  -> Linear
  -> SiLU
  -> Linear
  -> action embedding
```

输入输出：

```text
raw action block: (B, T, 10)
act_emb:          (B, T, 192)
```

注意：`act_emb` 只在 LeWM 内部作为 predictor 的条件输入。它不是环境可执行动作，也不是 planner 直接搜索的变量。

## 4. Predictor 的输入和输出

当前 predictor 配置：

```yaml
predictor:
  _target_: module.ARPredictor
  num_frames: 3
  input_dim: 192
  hidden_dim: 192
  output_dim: 192
  depth: 6
  heads: 16
  mlp_dim: 2048
  dim_head: 64
  dropout: 0.1
  emb_dropout: 0.0
```

全局训练配置：

```yaml
history_size: 3
num_preds: 1
```

PushT 数据配置：

```yaml
dataset:
  num_steps: ${eval:'${num_preds} + ${history_size}'}
```

因此：

```text
num_steps = 1 + 3 = 4
```

每个训练样本包含 4 个 LeWM 时间步。

训练 forward 中：

```python
output = self.model.encode(batch)

emb = output["emb"]          # (B, 4, 192)
act_emb = output["act_emb"]  # (B, 4, 192)

ctx_emb = emb[:, :ctx_len]
ctx_act = act_emb[:, :ctx_len]

tgt_emb = emb[:, n_preds:]
pred_emb = self.model.predict(ctx_emb, ctx_act)
```

当前：

```text
ctx_len = history_size = 3
num_preds = 1
```

所以：

```text
ctx_emb:  (B, 3, 192)
ctx_act:  (B, 3, 192)
tgt_emb:  (B, 3, 192)
pred_emb: (B, 3, 192)
```

直观理解：

```text
输入:
  z_0, z_1, z_2
  u_0, u_1, u_2

输出:
  预测 z_1, z_2, z_3
```

其中：

```text
z_i = 图像 latent
u_i = action block embedding
```

## 5. 原始 LeWM 的训练 loss

当前 LeWM 的 loss 主要由两部分组成：

```python
pred_loss = (pred_emb - tgt_emb).pow(2).mean()
sigreg_loss = self.sigreg(emb.transpose(0, 1))
loss = pred_loss + lambd * sigreg_loss
```

配置：

```yaml
loss:
  sigreg:
    weight: 0.09
    kwargs:
      knots: 17
      num_proj: 1024
```

数学形式：

$$
\mathcal{L}_{LeWM}
=
\mathcal{L}_{pred}
+
\lambda_{sig}\mathcal{L}_{SIGReg}
$$

其中：

$$
\mathcal{L}_{pred}
=
\|\hat{z}_{future} - z_{future}\|_2^2
$$

`pred_loss` 让 world model 学会在 latent 空间中预测未来。`SIGReg` 约束 latent 分布，避免表示坍塌，并鼓励 latent 分布更接近健康的各向同性高斯结构。

如果只有 prediction loss，理论上存在退化解：所有 \(z\) 都 collapse 成常数时，预测常数也可能得到很小的 prediction loss。`SIGReg` 用来打破这种退化。

## 6. Planning 时发生了什么

Planning 时，CEM/MPPI 搜索的是 embedding 前的原始 action block，不是 action embedding。

对 PushT：

```text
每个 LeWM planning step 的 action block 维度 = frameskip * action_dim = 5 * 2 = 10
```

Planning 流程：

```text
1. CEM/MPPI 在 raw action block 空间采样候选动作序列
2. LeWM 将候选 raw action block 送入 action_encoder
3. action_encoder 得到 act_emb
4. predictor 根据当前 latent 和 act_emb rollout 未来 latent
5. 比较预测末端 latent 和目标 latent
6. planner 选择 cost 最小的动作序列
7. 环境执行最优动作序列的前一段
```

源码中 cost 计算逻辑：

```python
cost = F.mse_loss(
    pred_emb[..., -1:, :],
    goal_emb[..., -1:, :].detach(),
    reduction="none",
).sum(...)
```

数学形式：

$$
J(a_{t:t+H})
=
\|\hat{z}_{t+H} - z_g\|_2^2
$$

所以：

```text
planner 搜索空间: raw action block
LeWM 评估空间: latent space
action_encoder: raw action block -> predictor condition
```

## 7. 新实验：IDM Auxiliary Decoder

我们希望新增一个 inverse dynamics decoder：

\[
\hat{a}_t = D_\psi(z_t, z_{t+1})
\]

它的目标是从相邻 latent transition 中恢复 embedding 前的原始 action block。

按当前决定，IDM decoder 的输入只包含：

```text
z_t
z_next
```

不加入：

```text
z_next - z_t
```

输出是原始 action block：

```text
frameskip * action_dim
```

对 PushT：

```text
output_dim = 5 * 2 = 10
```

## 8. IDM Decoder 建议架构

借鉴 LGP / GC-IDM 的 MLP 风格，但去掉 goal horizon 条件，因为这里是相邻 transition inverse dynamics。

建议第一版：

```text
input = concat(z_t, z_next)
input_dim = 192 + 192 = 384

MLP:
  Linear(384 -> 512)
  LayerNorm(512)
  GELU
  Dropout(0.1)

  Linear(512 -> 512)
  LayerNorm(512)
  GELU
  Dropout(0.1)

  Linear(512 -> 512)
  LayerNorm(512)
  GELU
  Dropout(0.1)

  Linear(512 -> 10)
```

数学形式：

$$
D_\psi: \mathbb{R}^{192} \times \mathbb{R}^{192}
\rightarrow
\mathbb{R}^{10}
$$

## 9. 新增 IDM loss

IDM loss：

$$
\mathcal{L}_{IDM}
=
\|D_\psi(z_t, z_{t+1}) - a_t\|_2^2
$$

总 loss：

$$
\mathcal{L}
=
\mathcal{L}_{pred}
+
\lambda_{sig}\mathcal{L}_{SIGReg}
+
\lambda_{idm}\mathcal{L}_{IDM}
$$

第一版建议：

```yaml
loss:
  idm:
    enabled: true
    weight: 0.1
```

后续可以扫：

```text
lambda_idm = 0.01, 0.05, 0.1, 0.2
```

## 10. 为什么 IDM loss 可能防 collapse

如果 latent 完全 collapse：

\[
z_t = z_{t+1} = c
\]

那么 decoder 输入恒定：

\[
\hat{a}_t = D_\psi(c,c)
\]

对所有样本都只能输出同一个动作或平均动作，无法拟合多样的真实动作 \(a_t\)。因此：

\[
\mathcal{L}_{IDM}
=
\mathbb{E}\|D_\psi(c,c)-a_t\|^2
\]

不会很小。

所以 IDM loss 可以惩罚最严重的常数 collapse，并鼓励 latent transition 保留 action-identifiable 信息。

但 IDM loss 和 SIGReg 的作用不同：

```text
IDM loss:
  约束 z_t -> z_{t+1} 的 transition 必须能恢复 action。

SIGReg:
  约束 latent 的全局分布，避免低方差、低秩、常数 collapse 等分布退化。
```

因此，IDM loss 更像 control-aware representation regularizer，而不是 SIGReg 的完全替代品。

## 11. 实验分组

建议第一阶段做四组：

```text
A. baseline
   pred_loss + SIGReg

B. pred only
   pred_loss

C. IDM only
   pred_loss + IDM loss

D. combined
   pred_loss + SIGReg + IDM loss
```

关键观察：

```text
1. pred + IDM 是否避免常数 collapse？
2. pred + IDM 的 latent 是否更适合训练 GC-IDM？
3. pred + SIGReg + IDM 是否提升 long-horizon control？
4. IDM loss 是否破坏原本 LeWM 的 latent prediction ability？
```

## 12. wandb 记录计划

训练阶段记录：

```text
train/loss
train/pred_loss
train/sigreg_loss
train/idm_loss
val/loss
val/pred_loss
val/sigreg_loss
val/idm_loss
```

latent 健康度记录：

```text
latent/variance_mean
latent/variance_min
latent/variance_max
latent/norm_mean
latent/pairwise_dist_mean
```

下游评估记录：

```text
eval/gcidm_success_rate
eval/gcidm_ms_per_episode
eval/goal_offset
eval/eval_budget
```

## 13. 代码组织原则

不改动原始 LeWM 代码。新增独立目录：

```text
lewm_idm_aux/
  README.md
  module_idm.py
  jepa_idm.py
  train_idm_aux.py
  configs/
    lewm_idm_aux.yaml
    ablation_pred_only.yaml
    ablation_pred_idm.yaml
    ablation_pred_sigreg_idm.yaml
```

原始代码只作为参考：

```text
module.py
jepa.py
train.py
config/train/*
```

这样可以保证 baseline 和新实验隔离，避免污染原始复现路径。
```

如果你确认，我下一步会在本地仓库新建这份 md 文件，并且暂时只写文档，不写训练代码。 
