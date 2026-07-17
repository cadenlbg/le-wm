# Single-Step Token IDM 阶段实验报告

## 摘要

本阶段实现了一个基于冻结 LeWM 表征的单步目标条件动作模型：

$$
\pi_\theta(a_t\mid z_t,z_g,\Delta t)
$$

模型输入当前 latent $z_t$、未来目标 latent $z_g$ 和剩余 horizon $\Delta t$，输出每个动作维度的离散 token 分布，再解码为连续环境动作。研究目标是验证冻结的 LeWM latent 是否已经包含足够的控制信息，使轻量 goal-conditioned policy 能替代昂贵的在线搜索规划。

当前阶段结果（eval budget = H + 25，seed = 42）：

| 目标 horizon | Token IDM | GC-IDM(paper上) | CEM | Token 相对 GC-IDM |
|---:|---:|---:|---:|---:|
| H25 | 89% | 84.5% | 89% | +4.5 pp |
| H35 | 81% | 76.0% | 未测 | +5.0 pp |
| H50 | 81% | 70.0% | 40% | +11.0 pp |
| H75 | 51% | 未测 | 15% | - |
| H100 | 38% | 未测 | 13% | - |

Token IDM 在 H25/H35/H50 上均优于 GC-IDM，分别领先 4.5、5.0、11.0 个百分点；与 CEM 相比，H25 持平，H50/H75/H100 明显更高。

## 1. 要解决的问题

视觉世界模型可以把图像映射到紧凑 latent space，但测试时从当前状态到目标状态通常仍依赖 CEM、MPPI 等搜索型规划器。这类方法存在以下问题：

1. 每个环境 step 都要评估大量候选动作序列。
2. 候选动作通常需要通过 world model rollout 比较。
3. 规划成本随候选数量、规划 horizon 和优化轮数增长。
4. 专家 transition 中已有的局部动作几何关系没有被直接摊销。

GC-IDM 直接学习：

$$
(z_t,z_g,\Delta t)\rightarrow a_t
$$

本项目进一步将连续动作回归改为逐维 token 分类，希望缓解 MSE 回归在多模态动作分布下产生平均动作的问题。

## 2. 总体数据流

~~~text
HDF5 pixels/actions/episode_ids
        |
        v
冻结 LeWM encoder + projector
        |
        v
embeddings.npz
  embeddings / actions / episode_ids / action_stats
        |
        v
TransitionEmbeddingDataset
  z_t / z_g / steps_remaining / a_t
        |
        v
GoalConditionedTokenIDM
        |
        +-- action_dim x n_bins logits
        +-- CE + L1 auxiliary loss
        +-- argmax/sampling -> detokenize
        |
        v
PushT 闭环执行
~~~

## 3. LeWM 表征提取

LeWM encoder 和 projector 在 Token IDM 训练中冻结。每一帧图像 $o_t$ 被编码为：

$$
z_t=P\left(\operatorname{CLS}(E(o_t))\right)
$$

对应逻辑为：

~~~python
out = encoder(image, interpolate_pos_encoding=True)
z = projector(out.last_hidden_state[:, 0])
~~~

预提取文件保存：

- embeddings：每一帧的 LeWM latent。
- actions：对应的专家动作。
- episode_ids：episode 边界信息。
- action_stats：动作 q01/q99 或 min/max 统计量。

当前配置：

~~~text
frameskip = 1
max_goal_horizon = 50
~~~

因此模型每次只预测一步动作。H75/H100 属于训练 horizon 之外的测试。

## 4. Transition dataset

对于当前帧 $t$，从同一 episode 的未来帧中采样目标帧 $g$：

$$
g\sim\operatorname{Uniform}
\left(t+1,\min(t+H_{\max},T_{\mathrm{episode}}-1)\right)
$$

训练样本为：

$$
(z_t,z_g,\Delta t,a_t),
\qquad \Delta t=g-t
$$

训练时随机采样 goal，使同一当前状态能够对应不同目标和目标距离。

当前流程先通过 train_split=0.9 按 episode 留出 10% held-out episodes，再对其余 transition 做约 90%/10% train/validation 切分。最终闭环 eval 使用 held-out episodes。训练 validation 主要用于选 best checkpoint，不等价于严格的 episode-level 泛化结果。

## 5. 动作 tokenization

每个动作维度先归一化到 $[-1,1]$：

$$
\tilde a_d=
2\frac{a_d-a_{\min,d}}
{a_{\max,d}-a_{\min,d}+\epsilon}-1
$$

再映射到 $N$ 个 bins：

$$
b_d=
\operatorname{round}
\left(\frac{\tilde a_d+1}{2}(N-1)\right)
$$

推理时使用 bin center $c_b$ 解码：

$$
\hat a_d=
\frac{c_{b_d}+1}{2}
(a_{\max,d}-a_{\min,d})+a_{\min,d}
$$

主要配置：

~~~text
n_bins = 256
token_offset = 0
normalization = bounds_q99
~~~

同时测试了 n_bins=128。bins 越多，动作分辨率越高，但精确 token 命中更困难。

## 6. Token IDM 模型架构

### 6.1 输入组织

默认输入为：

$$
x=[z_t;z_g;z_g-z_t]
$$

若 LeWM latent 维度 $D=192$，输入维度为：

$$
3D=576
$$

显式加入 $z_g-z_t$，用于表示当前状态到目标状态的 latent 位移。

### 6.2 Horizon embedding

剩余步数归一化为：

$$
\tau=\frac{\Delta t}{H_{\max}}
$$

然后经过 64 维 sinusoidal embedding 和两层 MLP，得到 horizon condition。

### 6.3 MLP backbone

默认配置：

~~~text
hidden_dim = 512
n_layers = 3
dropout = 0.1
activation = GELU
action_dim = 2
~~~

每个 backbone block 为：

~~~text
Linear -> LayerNorm -> GELU -> Dropout
~~~

### 6.4 Horizon modulation

主干输出 $h$ 使用 AdaLN/FiLM 风格的调制：

$$
h'=h\odot(1+s(e_\tau))+b(e_\tau)
$$

scale 和 shift 层使用 zero initialization，使初始化时接近 identity modulation。

### 6.5 Token head

输出形状为：

$$
\operatorname{logits}\in
\mathbb R^{B\times d_a\times N_{\mathrm{bins}}}
$$

PushT 默认是 $B\times2\times256$。当前每个动作维度独立分类：

$$
p(a_x,a_y\mid z_t,z_g,\Delta t)
\approx
p(a_x\mid\cdot)p(a_y\mid\cdot)
$$

这种设计降低了输出空间，但不能完整表达两个动作维度的联合相关性。

## 7. 训练目标

真实动作被转换为目标 token $b^\star$。主损失为：

$$
\mathcal L_{\mathrm{CE}}
=
-\frac{1}{Bd_a}\sum_{i,d}
\log p_\theta(b^\star_{i,d})
$$

使用 categorical distribution 的 bin center 计算期望动作：

$$
\bar a_{i,d}=\sum_b p_{i,d}(b)c_b
$$

辅助 L1 为：

$$
\mathcal L_{\mathrm{L1}}
=
\frac{1}{Bd_a}\sum_{i,d}
|\bar a_{i,d}-a_{i,d}|
$$

baseline 总损失：

$$
\mathcal L=
\mathcal L_{\mathrm{CE}}
+0.1\mathcal L_{\mathrm{L1}}
$$

纯 CE 消融令 L1 系数为 0。

主要训练参数：

~~~text
optimizer = AdamW
batch_size = 2048
epochs = 200
initial lr = 3e-4
weight_decay = 1e-4
~~~

原始 baseline 使用 cosine schedule：

$$
\eta_t=
\eta_{\min}
+\frac12(\eta_{\max}-\eta_{\min})
\left(1+\cos\frac{\pi t}{T}\right)
$$

其中：

$$
\eta_{\max}=3\times10^{-4},
\qquad
\eta_{\min}=3\times10^{-6}
$$

后续已加入 warm-start、constant/cosine scheduler、warmup、每 10 epoch checkpoint、best checkpoint 和 last checkpoint。当前观察中，$3\times10^{-4}$ cosine 下降到 $3\times10^{-5}$ 的 continuation 效果最好。

## 8. 训练结果

当前记录的训练指标：

~~~text
train loss = 3.512373
val CE     = 3.403548
val L1     = 0.047025
token acc  = 0.1615
~~~

指标含义：

- train loss：CE 与 0.1 L1 的训练总损失。
- val CE：validation token cross entropy。
- val L1：期望连续动作与专家动作的 L1。
- token acc=0.1615：逐动作维度 token 精确命中率为 16.15%。

对于 256 bins，随机 token accuracy 为：

$$
\frac1{256}=0.003906=0.39\%
$$

因此 16.15% 显著高于随机基线。token accuracy 要求精确命中同一 bin，相邻 bin 也算错误，所以还必须结合 val L1 和闭环 success rate 判断。

需要注意，train loss 包含 L1，而 val CE 只报告 CE，两者不是完全相同的量。当前 validation goal 还会动态采样，因此 epoch 间存在一定采样波动。

## 9. 闭环评估

闭环 eval 在真实 PushT 环境中执行：

~~~text
当前图像 -> LeWM 编码 z_t
固定目标图像 -> LeWM 编码 z_g，并缓存
(z_t, z_g, horizon) -> Token IDM
token argmax -> 连续动作 -> env.step
下一步重新编码当前图像
~~~

当前 Token IDM、GC-IDM 与 CEM 的闭环结果为：

| Goal horizon | Token IDM | GC-IDM | CEM | Token vs GC-IDM | Token vs CEM |
|---:|---:|---:|---:|---:|---:|
| H25 | 89% | 84.5% | 89% | +4.5 pp | 0 pp |
| H35 | 81% | 76.0% | 未测 | +5.0 pp | - |
| H50 | 81% | 70.0% | 40% | +11.0 pp | +41 pp |
| H75 | 51% | 未测 | 15% | - | +36 pp |
| H100 | 38% | 未测 | 13% | - | +25 pp |

结果说明：

1. H25 上 Token IDM 为 89%，GC-IDM 为 84.5%，Token IDM 提高 4.5 个百分点，并与 CEM 的 89% 持平。
2. H35 上 Token IDM 为 81%，GC-IDM 为 76.0%，Token IDM 提高 5.0 个百分点。
3. H50 上 Token IDM 为 81%，GC-IDM 为 70.0%，CEM 为 40%；Token IDM 分别提高 11 和 41 个百分点。
4. H75 上 Token IDM 为 51%，CEM 为 15%，绝对提高 36 个百分点。
5. H100 上 Token IDM 为 38%，CEM 为 13%，绝对提高 25 个百分点。
6. 从 H25 到 H50，Token IDM 下降 8 个百分点，GC-IDM 下降 14.5 个百分点，CEM 下降 49 个百分点。
7. H75/H100 上 Token IDM 自身也明显下降，latent goal 距离、训练 horizon 上限和闭环误差累积仍是瓶颈。

按成功率相对倍率计算：

$$
\frac{SR_{\mathrm{Token}}}{SR_{\mathrm{CEM}}}
=
\begin{cases}
1.00, & H25\\
2.03, & H50\\
3.40, & H75\\
2.92, & H100
\end{cases}
$$

该结果表明 Token IDM 的优势不仅来自推理速度，在当前协议下也体现为更高的 long-horizon goal-reaching success rate。形成最终结论前，仍需确认两种方法使用完全相同的 episode、start step、goal、eval budget、success 判定和随机种子。

### 9.1 Eval budget 与 horizon condition

当前 horizon condition 为：

$$
\Delta t_t=\min(B-t,H_{\max})
$$

其中 $B$ 是 eval budget。这导致 budget 不只是允许执行的最大步数，还会改变模型条件。例如同为 goal offset 50，budget=75 与 budget=100 在前半段输入的 horizon schedule 不同。

后续建议改为：

$$
\Delta t_t=
\max(1,\min(G-t,H_{\max}))
$$

其中 $G$ 是 goal offset。这样不同 budget 的前半段 policy 输入一致，才能纯粹比较额外执行预算的收益。

## 10. 当前结论

当前实验验证了：

1. 冻结 LeWM latent 已经支持有效的 goal-conditioned 单步控制。
2. 逐维动作 token 分类可以显著超过随机分类基线。
3. H25/H35/H50 上 Token IDM 分别比 GC-IDM 高 4.5、5.0、11.0 个百分点。
4. H25 上 Token IDM 与 CEM 持平；H50/H75/H100 上分别领先 CEM 41、36、25 个百分点。
5. Token IDM 对 horizon 增长的鲁棒性强于当前 GC-IDM 和 CEM baseline。
6. H75/H100 上 Token IDM 自身性能仍明显下降，说明单步误差、latent goal 距离和训练分布外外推仍是主要问题。

当前主要瓶颈：

- 三层 MLP 对复杂 current/goal 几何关系的表达能力有限。
- 逐动作维度独立分类忽略联合动作结构。
- 单步控制存在动作抖动和误差累积。
- 训练最大 horizon 50，与 H75/H100 eval 不匹配。
- validation 仍是 transition-level split，goal pair 未固定。
- eval budget 和 horizon condition 当前耦合。

## 11. 后续改进方向

### 11.1 修正评估和数据协议

优先完成：

1. 固定 validation 的 current/goal pairs。
2. 使用严格 episode-level train/val/test split。
3. 动作统计量只从训练 episode 计算。
4. 用 goal_offset-step_count 作为闭环 horizon condition。
5. 不同 budget 使用相同起点和目标任务。
6. 记录 argmax action L1、within-1-bin accuracy 和 success rate。
7. 保存失败轨迹和视频，分析过冲、振荡和方向错误。

### 11.2 MLP-Transformer

将单个拼接向量改为轻量 token sequence：

~~~text
[current_latent, goal_latent, delta_latent, horizon]
~~~

使用 2--4 层 Transformer encoder 建模 token 间交互，再接动作 token head。它可以更灵活地建模 current/goal/horizon 关系，并为 latent waypoint、subgoal sequence 和 autoregressive action decoding 留出结构。

需要控制参数量，避免单步监督不足导致过拟合。

### 11.3 ACT / action chunking

当前模型只预测 $a_t$。ACT 可以预测：

$$
a_{t:t+K-1}
$$

潜在收益：

- 减少单步动作抖动。
- 学习短期动作协同。
- 减少每一步重新调用策略的开销。

建议从 $K=5$ 开始，比较 execute_horizon=1 和 execute_horizon=5，并严格处理训练 target 与闭环执行的时间对齐。

### 11.4 联合动作建模

当前近似：

$$
p(a_x,a_y)\approx p(a_x)p(a_y)
$$

可以比较：

1. action-dimension autoregressive token head。
2. 共享 action token decoder。
3. 低维 joint codebook。
4. diffusion/action prior。

不建议直接建立 $256^2$ 类 joint classification，因为类别空间过大。

### 11.5 GRPO / RL fine-tuning

将监督模型作为初始策略：

$$
\pi_{\mathrm{SFT}}\rightarrow\pi_{\mathrm{RL}}
$$

再利用环境 success reward 或 shaped reward 做 GRPO 等 RL fine-tuning。目标包括：

- 提高 H50/H75/H100 成功率。
- 减少动作抖动和终点过冲。
- 适应真实闭环状态分布，而不只拟合专家 transition。

RL 阶段应冻结或用极小学习率更新 LeWM encoder，使用 SFT policy 作为 KL reference，并组合成功奖励、动作平滑和越界惩罚，避免 reward hacking。

### 11.6 Horizon curriculum

当前训练 horizon 最高为 50，可以使用：

$$
H_{\mathrm{train}}:
10\rightarrow25\rightarrow50\rightarrow75
$$

或混合短、中、长目标采样，使模型显式见到更远目标。还应按 horizon bucket 分别记录 loss 和 success rate。

### 11.7 层次化 waypoint

若直接从 $z_t$ 追远期 $z_g$ 不稳定，可以拆为：

$$
z_t\rightarrow z_{w_1}\rightarrow z_{w_2}\rightarrow z_g
$$

上层选择 latent waypoint，下层 Token IDM 或 ACT 完成训练分布内的局部控制，从而把 long-horizon 任务转化为多个 short-horizon 子问题。

## 12. 下一阶段实验建议

1. 修正 budget 与 horizon condition 的耦合。
2. 在相同任务上重测 G25-B50、G50-B50、G50-B75、G50-B100。
3. 固定 validation pairs，重新确认学习率 continuation。
4. 系统比较 128 bins、256 bins、CE-only 和 CE+L1。
5. 增加轻量 MLP-Transformer baseline。
6. 增加 $K=5$ 的 ACT/action chunk baseline。
7. 再进行 horizon curriculum 和 waypoint 实验。
8. 最后使用 GRPO/RL fine-tuning 优化真实闭环成功率。

## 13. 总结

Single-Step Token IDM 将连续动作回归：

$$
(z_t,z_g,\Delta t)\rightarrow a_t
$$

改写为：

$$
(z_t,z_g,\Delta t)\rightarrow p(b_t)\rightarrow a_t
$$

当前 H25/H50 的结果说明冻结 LeWM latent 已支持局部和中距离控制；H75/H100 的下降说明单步 MLP 受到 long-horizon 累积误差和训练分布外目标的限制。下一阶段应先保证评估协议严格、可复现，再通过 Transformer、ACT、horizon curriculum、waypoint 和 GRPO 提升长距离闭环能力。
