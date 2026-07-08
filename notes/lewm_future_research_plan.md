# LeWM 后续研究计划

本文档整理学长提出的三个后续方向，并结合当前仓库里已经跑通的 LeWM 复现、`latent_subgoal_act`、LeWM-guided rerank/CEM 和 diffusion policy 工作线，给出可落地的实验路线。

三个方向可以先统一理解为：

```text
方向 1：让 LeWM 本体更强
  -> 改 encoder / predictor / 训练输入，让 world model 更会利用历史上下文。

方向 2：让 LeWM 有 self critic 能力
  -> 不只预测未来，还能判断自己的预测是否可信、动作是否合理、rollout 是否可能失败。

方向 3：让 LeWM 有 short-to-long generalization 能力
  -> 用短 horizon 学到的 dynamics 支撑更长 horizon 的规划、评估和控制。
```

当前最稳妥的策略不是同时大改 LeWM 和下游 policy，而是：先冻结或轻改 LeWM，设计能快速验证的 auxiliary head / adapter / critic；如果有效，再回到 LeWM 预训练阶段做更重的结构改动。

## 1. 当前基础与问题定位

目前已有基础：

- 原始 LeWM PushT 复现链路已经跑通，包括 checkpoint 转换、数据加载和 CEM baseline evaluation。
- `latent_subgoal_act` 已经从“直接预测动作”推进到：

```text
当前 latent z_t
  + 目标 latent z_g
  -> 预测未来 latent 序列 \hat{z}_{t+1:t+H}
  -> 预测 action chunk a_{t:t+K}
  -> 可选 LeWM rollout / rerank / CEM 验证
```

- diffusion policy 分支也已经开始接入 LeWM 表示和 rollout video / wandb 评估。

但当前主线仍然偏“利用 frozen LeWM 做 downstream control”。学长提出的三个方向更靠近：

```text
LeWM 作为世界模型本身还能不能变强？
LeWM 能不能不只是 predictor，而是 predictor + judge？
LeWM 能不能从短期预测泛化到长期推理？
```

所以后续计划应该把“改下游 policy”和“改 LeWM 本体能力”分开记录，避免实验结论混在一起。

## 2. 方向一：让 LeWM 本身更有力

### 2.1 核心想法

原始 LeWM 的预测形式可以抽象为：

$$
\hat{z}_{t+1:t+H} = f_\theta(z_t, a_{t:t+H-1}, z_g)
$$

或者在 CEM / rollout 中近似使用当前状态和候选 action chunk 预测未来 latent。它的问题是：当前输入可能太短，只看当前状态很难判断物体速度、接触历史、动作是否已经产生效果。

可以把 LeWM 改成 history-conditioned world model：

$$
\hat{z}_{t+1:t+H} = f_\theta(z_{t-L:t}, a_{t-L:t-1}, a_{t:t+H-1}, z_g)
$$

其中：

- $z_{t-L:t}$ 是更早的 state / image latent history。
- $a_{t-L:t-1}$ 是已经执行过的 action history。
- $a_{t:t+H-1}$ 是 candidate future action。
- $z_g$ 可选，用于 goal-conditioned prediction。

直觉是：如果 LeWM 能看到“刚才怎么推、推到哪里、速度趋势是什么”，它的预测会比只看单帧更可靠。

### 2.2 可做方案

#### 方案 A：History token adapter

尽量不重训整个 LeWM，只在 encoder / predictor 前面加一个 history adapter。

输入：

```text
[z_{t-L}, a_{t-L}], [z_{t-L+1}, a_{t-L+1}], ..., [z_t]
```

输出一个 context token：

$$
c_t = \mathrm{Adapter}(z_{t-L:t}, a_{t-L:t-1})
$$

再把 $c_t$ 注入 predictor：

$$
\hat{z}_{t+h} = f_\theta(z_t, c_t, a_{t:t+h-1})
$$

优点：改动小，可以冻结大部分 LeWM 参数，只训练 adapter。

风险：如果原 predictor 接口不方便注入额外 token，需要先读 `stable_worldmodel` 具体模型结构。

#### 方案 B：In-context trajectory prefix

把 LeWM predictor 当成 sequence model 使用，在预测前给它一段 demonstration prefix：

```text
context:
  (z_i, a_i, z_{i+1}), ..., (z_{i+m}, a_{i+m}, z_{i+m+1})
query:
  (z_t, a_t, ?)
target:
  z_{t+1}
```

目标是让模型学到：同一 episode 或同一任务中的前几步转移，可以帮助判断后续 dynamics。

训练目标：

$$
\mathcal{L}_{\mathrm{dyn}} = \sum_{h=1}^{H} \left\| \hat{z}_{t+h} - z_{t+h} \right\|_2^2
$$

如果要更像 in-context learning，可以让 context 和 query 来自同一 episode 的不同时间段；也可以混入不同 episode，让模型学会区分上下文是否相关。

#### 方案 C：Action-conditioned memory bank

为每个 episode 构建一小段 latent/action memory，预测时通过 attention 检索相似历史：

$$
m_t = \mathrm{Attn}(q=z_t, K=z_{1:t}, V=[z_{1:t}, a_{1:t-1}])
$$

然后：

$$
\hat{z}_{t+h} = f_\theta(z_t, m_t, a_{t:t+h-1})
$$

这个方向有研究味，但工程复杂度比 adapter 更高，建议作为第二阶段。

### 2.3 最小实验

第一版不要直接重训完整 LeWM，建议先做 frozen encoder + trainable history predictor：

```text
1. 用现有 LeWM encoder 离线抽取 z_t。
2. 构建 history dataset：输入 z_{t-L:t}, a_{t-L:t-1}, a_{t:t+H-1}，目标 z_{t+1:t+H}。
3. 训练一个轻量 Transformer / MLP-Mixer predictor。
4. 和原 LeWM rollout 比较 latent prediction error。
5. 再接入 rerank / CEM，看 control success 是否提升。
```

建议从 PushT 开始：

| 项目 | 建议设置 |
| --- | --- |
| history length $L$ | 2, 4, 8 |
| prediction horizon $H$ | 5, 10, 20 |
| 输入 | $z_{t-L:t}$、$a_{t-L:t-1}$、candidate $a_{t:t+H-1}$ |
| 指标 | latent MSE、terminal latent error、rerank 后 success rate |
| baseline | 原始 LeWM rollout、当前 `latent_subgoal_act` |

## 3. 方向二：LeWM self critic

### 3.1 核心想法

现在 LeWM 更像一个 predictor：给状态和动作，预测未来。但在 planning / control 中，我们还需要它回答几个“自我判断”问题：

```text
这个 rollout 可信吗？
这个 action chunk 会不会导致失败？
这个预测是不是 out-of-distribution？
多个预测之间是否自洽？
```

这就是 LeWM self critic 的入口。类比 LLM self critic，不是让模型重新生成答案，而是让模型对自己的预测给出质量判断。

### 3.2 三种 critic 形式

#### 形式 A：Uncertainty critic

训练 LeWM 或额外 head 输出预测不确定性：

$$
(\hat{z}_{t+h}, \sigma_{t+h}) = f_\theta(z_t, a_{t:t+h-1})
$$

规划时不只看目标距离：

$$
J(a) = \left\| \hat{z}_{t+H} - z_g \right\|_2^2 + \lambda \cdot \sigma_{t+H}
$$

如果模型对某个 action rollout 很不确定，就降低它的排名。

实现方式：

- ensemble 多个轻量 predictor。
- dropout sampling。
- 直接预测 variance，并用 Gaussian negative log likelihood 训练。

#### 形式 B：Consistency critic

让 LeWM 从不同路径预测同一个未来 latent，然后检查是否一致。

例如一步一步 rollout：

$$
\hat{z}_{t+H}^{\mathrm{step}} = f(f(...f(z_t, a_t), a_{t+1}), ...)
$$

直接多步预测：

$$
\hat{z}_{t+H}^{\mathrm{direct}} = f(z_t, a_{t:t+H-1})
$$

critic 分数：

$$
s_{\mathrm{cons}} = \left\| \hat{z}_{t+H}^{\mathrm{step}} - \hat{z}_{t+H}^{\mathrm{direct}} \right\|_2^2
$$

如果两条预测路径差异很大，说明模型可能不确定或 action 进入了训练分布之外。

#### 形式 C：Outcome critic

额外训练一个 head 预测当前 rollout 的成功概率或离目标的可达性：

$$
p_{\mathrm{success}} = g_\phi(z_t, \hat{z}_{t+1:t+H}, z_g, a_{t:t+H-1})
$$

规划目标变成：

$$
J(a) = \alpha \left\| \hat{z}_{t+H} - z_g \right\|_2^2 - \beta \log p_{\mathrm{success}}
$$

这比单纯 terminal latent distance 更强，因为有些 latent 上接近目标的轨迹可能动作不可执行、不稳定或容易越界。

### 3.3 最小实验

建议先做 consistency critic，因为它不需要额外人工标签，也比较符合“self critic”：

```text
1. 用同一 action chunk 做两种 rollout：step rollout 和 direct / chunk rollout。
2. 计算 terminal latent consistency error。
3. 在 rerank 中加入 consistency penalty。
4. 比较 success rate、mean score、bad rollout 数量。
```

可以先在 `latent_subgoal_act/wm_rollout.py` 附近做实验，因为那里已经有 frozen LeWM rollout 接口。

建议指标：

| 指标 | 含义 |
| --- | --- |
| terminal distance | $\|\hat{z}_{t+H} - z_g\|_2$ |
| consistency error | $\|\hat{z}_{t+H}^{\mathrm{step}} - \hat{z}_{t+H}^{\mathrm{direct}}\|_2$ |
| critic-rerank success | 加 critic 后的 PushT success rate |
| rejection quality | 被 critic 拒绝的 action 是否确实表现差 |

## 4. 方向三：short-to-long generalization

### 4.1 核心想法

LeWM 训练时常见问题是短期预测相对容易，长期预测会误差累积。但控制任务往往需要长期目标：

```text
短 horizon：预测未来 5 步 latent，比较稳定。
长 horizon：规划到 25 / 50 步目标，误差可能爆炸。
```

short-to-long generalization 的目标是：训练时主要学短期可靠 dynamics，推理时能组合成长期规划。

可以抽象为：

$$
f_\theta^{(1)}: z_t, a_t \rightarrow z_{t+1}
$$

通过递推得到：

$$
\hat{z}_{t+H} = f_\theta^{(1)} \circ f_\theta^{(1)} \circ ... \circ f_\theta^{(1)}(z_t, a_{t:t+H-1})
$$

关键不是“能递推”，而是“递推很多步后仍然在合理 latent manifold 上”。

### 4.2 可做方案

#### 方案 A：Horizon curriculum

训练 horizon 从短到长逐步增加：

```text
阶段 1：H = 1 或 5
阶段 2：H = 10
阶段 3：H = 20 / 25
阶段 4：混合不同 H，并随机 mask 中间 latent
```

loss：

$$
\mathcal{L} = \sum_{h \in \mathcal{H}} w_h \left\| \hat{z}_{t+h} - z_{t+h} \right\|_2^2
$$

其中 $w_h$ 可以随着 horizon 增大而降低，避免长 horizon noise 过强。

#### 方案 B：Latent manifold regularization

如果长期 rollout 离开真实 latent 分布，模型即使 terminal distance 小也不可信。可以加一个 latent realism loss：

$$
\mathcal{L}_{\mathrm{real}} = - \log D_\psi(\hat{z}_{t+h})
$$

或者更简单：用真实 latent 的均值/方差、nearest-neighbor distance、PCA 范围做离线约束。

第一版不建议直接上 adversarial discriminator，可以先做 nearest-neighbor / density score 作为 critic。

#### 方案 C：Subgoal chain planning

不要求一次预测到最终目标，而是把长目标拆成多个短 subgoal：

$$
z_t \rightarrow \hat{z}_{t+5} \rightarrow \hat{z}_{t+10} \rightarrow ... \rightarrow z_g
$$

这和当前 `latent_subgoal_act` 已经预测 `z_h_seq` 很接近。后续可以把它升级成：

```text
policy 先产生 latent subgoal chain
LeWM critic 检查每段 subgoal 是否可达
action planner 只负责到下一个可达 subgoal
```

这样 short-to-long 不是靠一次超长 rollout，而是靠“短段可达性 + subgoal chain consistency”。

### 4.3 最小实验

建议基于现有 `z_h_seq` 做，不要另开全新系统：

```text
1. 训练不同 subgoal_steps：例如 [1, 3, 5]、[1, 5, 10]、[5, 10, 20]。
2. 比较短 horizon 训练是否能在长 horizon eval 上保持 terminal latent accuracy。
3. 在 eval 中记录每个 horizon 的 latent error 曲线。
4. 加入 subgoal chain consistency penalty，看 long-horizon control 是否更稳。
```

建议指标：

| 指标 | 含义 |
| --- | --- |
| error@h | $\|\hat{z}_{t+h} - z_{t+h}\|_2$ |
| terminal error | $\|\hat{z}_{t+H} - z_g\|_2$ |
| rollout drift | 长期 rollout 偏离真实 latent 分布的程度 |
| control success | PushT success rate / max reward |
| horizon transfer | 训练短 $H_{train}$，测试长 $H_{eval}$ 的性能下降 |

## 5. 推荐优先级

我建议按下面顺序推进：

### 第一优先级：self critic 的 consistency rerank

原因：

- 最贴近学长的“self critic”提法。
- 不需要重训完整 LeWM。
- 可以直接接到当前 `latent_subgoal_act` 的 rerank / CEM。
- 如果成功，论文叙事很清楚：LeWM 不仅预测未来，还评估自己的预测可靠性。

最小结果：

```text
baseline rerank:
  score = terminal_distance

self-critic rerank:
  score = terminal_distance + lambda * consistency_error
```

如果 success rate 或 bad-action rejection 明显改善，就值得继续深入。

### 第二优先级：short-to-long 的 subgoal chain consistency

原因：

- 和当前 `z_h_seq` 设计天然一致。
- 能把已有 latent subgoal 线升级成更明确的长期泛化问题。
- 评价方式比较清楚，可以画 error-vs-horizon 曲线。

最小结果：

```text
比较不同训练 horizon / eval horizon 的 latent error 和 PushT success。
证明短期 latent subgoal 学习是否能组合成长期控制。
```

### 第三优先级：history-conditioned / in-context LeWM

原因：

- 最像“让 LeWM 本体变强”。
- 但工程改动最大，可能涉及 `stable_worldmodel` 内部结构。
- 建议先用 frozen encoder + 轻量 predictor 证明 history 有用，再决定是否改 LeWM 预训练。

最小结果：

```text
history predictor 比 no-history predictor 有更低 multi-step latent error；
接入 rerank 后 control success 有提升。
```

## 6. 四周实验路线

### Week 1：建立 critic / horizon evaluation 工具

目标：先让评估可见。

任务：

- 在 latent dataset 或 eval 过程中记录不同 horizon 的 latent prediction error。
- 实现 consistency error 计算。
- 保存 candidate action 的 terminal distance、consistency error、最终 reward。

产出：

```text
metrics.jsonl 中能看到：
  terminal_distance
  consistency_error
  selected_candidate_rank
  episode_reward / success
```

### Week 2：self critic rerank

目标：验证 critic 是否能提升 action selection。

任务：

- 在 rerank score 中加入：

$$
J(a) = d_{goal}(a) + \lambda d_{cons}(a)
$$

- 扫描 $\lambda \in \{0.0, 0.1, 0.3, 1.0\}$。
- 和原始 terminal-distance rerank 对比。

产出：

```text
self_critic_rerank_results.txt
critic_ablation_metrics.jsonl
```

### Week 3：short-to-long subgoal chain

目标：看短 horizon 训练能否迁移到长 horizon 控制。

任务：

- 训练不同 `subgoal_steps` 组合。
- 评估 `error@h` 曲线。
- 加入 chain consistency penalty。

可尝试 loss：

$$
\mathcal{L} = \mathcal{L}_{action} + \alpha \mathcal{L}_{subgoal} + \beta \mathcal{L}_{chain}
$$

其中：

$$
\mathcal{L}_{chain} = \sum_i \left\| \hat{z}_{t+h_i}^{rollout} - \hat{z}_{t+h_i}^{policy} \right\|_2^2
$$

产出：

```text
horizon_ablation_results.md
error_vs_horizon.png
```

### Week 4：history-conditioned predictor prototype

目标：低成本验证 history / in-context 是否值得改 LeWM 本体。

任务：

- 构建 history latent dataset。
- 训练 no-history predictor 和 history predictor。
- 比较 latent rollout error。
- 可选接入 rerank。

产出：

```text
history_predictor_ablation.md
history_len_vs_error.png
```

## 7. 推荐先写成的实验命名

为了后续结果清楚，建议实验名统一：

| 实验名 | 含义 |
| --- | --- |
| `critic_consistency_rerank` | terminal distance + consistency critic rerank |
| `critic_uncertainty_ensemble` | ensemble/dropout uncertainty critic |
| `horizon_subgoal_chain` | short-to-long subgoal chain consistency |
| `history_predictor_frozen_lewm` | frozen encoder + history-conditioned predictor |
| `icl_prefix_predictor` | in-context trajectory prefix predictor |

结果文件也保持分开，避免覆盖：

```text
pusht_rerank_baseline_results.txt
pusht_self_critic_rerank_results.txt
pusht_horizon_chain_results.txt
pusht_history_predictor_results.txt
```

## 8. 和学长讨论时的简短版本

可以这样汇报：

```text
我觉得这三个方向可以按风险拆开：

1. self critic 最适合先做，因为可以不重训 LeWM，直接在现有 rollout/rerank 上加 consistency 或 uncertainty 分数，验证 LeWM 能不能判断自己的预测质量。

2. short-to-long 可以基于我们现在的 z_h_seq 做，把长期目标拆成 subgoal chain，评估短 horizon latent prediction 是否能组合成长 horizon control。

3. in-context / history-conditioned LeWM 是最像增强模型本体的方向，但改动最大。建议先用 frozen encoder + history predictor 做 ablation，证明历史 state/action 真的降低 multi-step latent error 后，再考虑改 LeWM 预训练结构。
```

## 9. 当前最建议的下一步

下一步建议直接做：

```text
self critic consistency rerank
```

因为它最容易和当前代码接上，实验结论也最容易解释：

```text
LeWM 原来只负责预测未来；
现在让 LeWM 同时评估这个预测是否自洽；
规划时选择既接近目标、又被 LeWM 自己认为可信的 action。
```

如果这个方向跑出提升，再把 critic 从 consistency 扩展到 uncertainty / outcome critic；如果没有提升，再转向 short-to-long 的 subgoal chain，因为那条线和当前 `z_h_seq` 更紧密。
