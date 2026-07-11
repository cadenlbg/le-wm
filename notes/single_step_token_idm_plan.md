# 单步预测计划书：参考 GC-IDM + SimpleVLA-RL 的动作分布建模方案

本文档目标是：**先不做 ACT / 长序列 chunk policy，先做单步预测**。  
核心思路是沿用 GC-IDM 的 goal-conditioned inverse dynamics 形式，但把输出从“单点动作回归”改成“动作分布”，并参考 SimpleVLA-RL 的逐维离散化方式，把连续动作转成 token 分布来学。

---

## 1. 目标定义

我们当前要解决的问题是：

$$
\pi_\theta(a_t \mid o_t, g, \Delta t)
$$

其中：

- $o_t$：当前观测
- $g$：目标观测 / 目标 latent
- $\Delta t$：距离目标的剩余步数，或者一个 horizon embedding
- $a_t$：当前一步动作

这一步**只预测单步动作**，不做 ACT 式的 action chunk rollout。

---

## 2. 原理解析

### 2.1 参考 GC-IDM 的核心思想

GC-IDM 的本质是：

$$
f_\theta(z_t, z_g, \Delta t) \rightarrow a_t
$$

它把：

- 当前 latent \(z_t\)
- 目标 latent \(z_g\)
- 剩余步数 \(\Delta t\)

映射到下一步动作。

这类方法的优势是：

1. 推理时只需一次前向；
2. 不需要显式搜索；
3. 闭环执行，能不断修正偏差；
4. 很适合 goal-conditioned 控制。

### 2.2 为什么要把动作做成“分布”

如果直接回归一个动作点，模型容易学成平均值。  
在多解任务里，这会出问题：

- 左绕可行
- 右绕也可行
- 回归平均后可能两边都不对

所以更合理的是学：

$$
\pi_\theta(a_t \mid z_t, z_g, \Delta t)
$$

即输出动作分布，再从分布中采样或取 argmax。

### 2.3 参考 SimpleVLA-RL 的离散化方式

SimpleVLA-RL 的做法不是 learned tokenizer，而是：

1. 把连续动作归一化到 \([-1,1]\)
2. 固定切成 \(B\) 个 bins
3. 每个动作维度单独离散化
4. token id 只是在词表尾部预留的一段动作 id

因此我们可以把 PushT 的二维动作写成：

$$
a_t = (a_t^x, a_t^y)
$$

并离散化为：

$$
b_t^x, b_t^y \in \{0, \dots, B-1\}
$$

最终模型学习的是：

$$
p_\theta(b_t^x \mid z_t, z_g, \Delta t), \quad p_\theta(b_t^y \mid z_t, z_g, \Delta t)
$$

---

## 3. 推荐模型架构

### 3.1 总体结构

建议第一版直接做一个 **goal-conditioned stochastic IDM**：

```text
obs_t  -> frozen encoder -> z_t
goal   -> frozen encoder -> z_g
Δt     -> horizon embedding

[z_t, z_g, z_g - z_t, Δt_emb]
            |
            MLP / small Transformer
            |
      action distribution head
            |
   logits over per-dim bins
```

### 3.2 输入设计

推荐输入：

$$
[z_t, z_g, z_g - z_t, \phi(\Delta t)]
$$

其中：

- \(z_t\)：当前状态表示
- \(z_g\)：目标状态表示
- \(z_g - z_t\)：显式差分特征
- \(\phi(\Delta t)\)：时间步嵌入

### 3.3 输出设计

对 PushT 这种二维动作，推荐：

$$
\text{logits} \in \mathbb{R}^{2 \times B}
$$

若只预测单步：

- \(B\)：每一维的离散 bins 数
- 输出两个 categorical 分布，分别对应 \(x,y\)

若未来扩展到 action chunk：

$$
\text{logits} \in \mathbb{R}^{K \times 2 \times B}
$$

但当前计划先不做 chunk。

---

## 4. 动作离散化 / 反离散化

### 4.1 离散化

连续动作先归一化到 \([-1,1]\)：

$$
\tilde a = 2 \cdot \frac{a - a_{\min}}{a_{\max} - a_{\min}} - 1
$$

然后映射到 bins：

$$
b = \operatorname{clip}\left(
\operatorname{round}\left(\frac{\tilde a + 1}{2}(B-1)\right),
0,
B-1
\right)
$$

### 4.2 反离散化

从 logits 得到类别后，映射回 bin center：

$$
\hat a_{\text{norm}} = c_b
$$

其中 \(c_b\) 是第 \(b\) 个 bin center。  
之后再反归一化回环境动作空间。

---

## 5. Loss 设计

### 5.1 主 loss：Cross Entropy

如果标签是 bin index，那么最直接的监督是：

$$
\mathcal{L}_{\text{CE}}
=
- \sum_j \log p_\theta(b_j^* \mid z_t, z_g, \Delta t)
$$

对 PushT 二维动作就是：

$$
\mathcal{L}_{\text{CE}}
=
- \log p_\theta(b_x^*) - \log p_\theta(b_y^*)
$$

### 5.2 辅助 loss：expected action L1

为了让离散分布更贴近连续动作，可加入期望动作的 L1：

$$
\hat a_j = \sum_{b=0}^{B-1} p_\theta(b_j=b)\, c_b
$$

$$
\mathcal{L}_{\text{L1}} = \|\hat a - a^*\|_1
$$

### 5.3 可选正则：entropy

为了避免过早塌缩成 deterministic policy，可加熵正则：

$$
\mathcal{L}_{\text{ent}} = - H(p_\theta)
$$

### 5.4 总 loss

推荐第一版：

$$
\mathcal{L}
=
\mathcal{L}_{\text{CE}}
+
\lambda_{\text{L1}}\mathcal{L}_{\text{L1}}
- \beta H
$$

默认超参：

- \(\lambda_{\text{L1}} = 0.1\)
- \(\beta = 0\)

---

## 6. 训练与推理方式

### 6.1 训练

训练时使用：

- 当前观测 latent
- 目标 latent
- 剩余步数
- 监督动作 bin label

训练目标是让模型学到一个**分布式 inverse dynamics policy**。

### 6.2 推理

推理时有三种方式：

1. `argmax`：最稳定
2. `temperature sampling`：更适合探索
3. `top-k / nucleus`：避免低质量动作

PushT 第一版建议：

$$
b_j \sim \operatorname{Categorical}(\operatorname{softmax}(\ell_j / \tau))
$$

然后把类别映射回连续动作。

---

## 7. 需要构建哪些代码

### 7.1 新增 `tokenization` 工具

建议新增：

- `idm/tokenization.py`

职责：

- 连续动作 \(\leftrightarrow\) bin index
- bin index \(\leftrightarrow\) bin center
- 支持 PushT 的二维动作

### 7.2 新增单步 IDM 模型

建议新增：

- `idm/model_token.py`

职责：

- 输入 \(z_t, z_g, \Delta t\)
- 输出每个动作维度的 categorical logits
- 支持 `sample_action()` / `predict_action()`

### 7.3 新增数据集包装

建议新增：

- `idm/dataset_token.py`

职责：

- 从原始轨迹中采样 \((z_t, z_g, \Delta t, a_t)\)
- 把连续动作转成 token label
- 训练 / 验证 split

### 7.4 新增训练脚本

建议新增：

- `train_idm_token.py`

职责：

- 加载 LeWM embeddings 或直接加载 encoder 输出
- 构造 token labels
- 训练 CE + L1 + entropy
- 保存 checkpoint 和 history

### 7.5 新增推理 / 评估脚本

建议新增：

- `eval_idm_token.py`

职责：

- 单步闭环评估
- goal cache
- 动作采样策略切换
- success rate / reward 统计

### 7.6 如需接现有 LeWM 管线

可以考虑在这些现有文件上加入口，而不是重写全部：

- `idm/model.py`
- `idm/dataset.py`
- `train_idm.py`

如果想保持最小改动，可以在这些文件外包一层 token 版本。

---

## 8. 参考代码路径

### 8.1 GC-IDM 参考

重点看：

- `other exp/Latent-Geometry-Beyond-Search-Amortizing-Planning-in-World-Models/idm/model.py`
- `other exp/Latent-Geometry-Beyond-Search-Amortizing-Planning-in-World-Models/idm/dataset.py`
- `other exp/Latent-Geometry-Beyond-Search-Amortizing-Planning-in-World-Models/train_idm.py`
- `other exp/Latent-Geometry-Beyond-Search-Amortizing-Planning-in-World-Models/eval_idm.py`

可复用点：

- frozen encoder + latent state
- goal sampling
- steps remaining
- 单步闭环执行

### 8.2 SimpleVLA-RL 参考

重点看：

- `SimpleVLA-RL/verl/utils/vla_utils/openvla_oft/constants.py`
- `SimpleVLA-RL/verl/utils/vla_utils/openvla_oft/modeling_prismatic.py`
- `SimpleVLA-RL/verl/utils/vla_utils/openvla_oft/train_utils.py`
- `SimpleVLA-RL/verl/workers/rollout/rob_rollout.py`

可复用点：

- 逐维离散化
- `bin_centers`
- vocab 尾部 action token 预约
- token logits -> action 的反解码

---

## 9. 第一版落地顺序

### Phase 1: 只做单步分布预测

1. 定义 tokenization
2. 定义模型 head
3. 跑通训练 loss
4. 跑通单步推理

### Phase 2: 加闭环评估

1. 目标编码缓存
2. step-by-step rollout
3. 成功率 / reward 评估

### Phase 3: 再考虑扩展

1. action chunk
2. 更复杂的 policy head
3. RL 微调

---

## 10. 结论

这条线的最小可行方案是：

$$
\text{LeWM latent} + \text{goal-conditioned IDM} + \text{per-dim discrete action distribution}
$$

它保留了 GC-IDM 的单步闭环优势，同时借用了 SimpleVLA-RL 的离散动作表达方式。  
对我们当前项目来说，这是一个比 ACT 更轻、更稳、也更适合先验证的起点。
