# LeWM 本周工作总结

## 1. 本周主线

这一周的工作主线可以概括为：

```text
先把 LeWM 原始复现链路跑通，
再在冻结 LeWM 主干的基础上探索更快、更可控的 downstream action policy。
```

我们先完成了远程环境、数据、checkpoint、PushT baseline evaluation 和实验归档规范；随后从简单的 latent behavior cloning 出发，逐步转向更有希望的 `latent_subgoal_act` 路线，即：

```text
短期 latent subgoal prediction
  + action chunk prediction
  + frozen LeWM rollout / rerank / CEM verification
```

这使研究从“能不能复现 LeWM”推进到了“能不能利用冻结 LeWM latent 改进 action selection”。

## 2. 主要尝试

### 2.1 LeWM 原始复现链路

我们首先整理并跑通了 LeWM 的原始实验链路。

主要尝试包括：

- 在远程服务器上建立统一工作目录：

```text
代码目录：/data/zflin/lewm_re/le-wm
数据与 checkpoint：/data/zflin/lewm_re/stablewm_data
实验归档：/data/zflin/lewm_re/experiments
```

- 配置并验证 `lewm` conda 环境。
- 确认 PyTorch、CUDA、`stable_worldmodel`、`stable_pretraining` 可正常导入。
- 下载并解压 PushT 和 TwoRoom 数据。
- 将实验产物从代码仓库中分离出去，避免数据、checkpoint、视频和 Hydra outputs 污染 Git 仓库。

这一部分的意义是：后续所有方法都可以在同一个可复现环境中和原始 LeWM-CEM baseline 做比较。

### 2.2 PushT 作者 checkpoint 转换与 baseline evaluation

第二个尝试是让作者提供的 PushT checkpoint 能被当前 LeWM loader 正确加载。

我们解决了几个关键问题：

- 当前 `stable_worldmodel` 的数据读取接口和旧版不同，需要改用新版 dataset loading 方式。
- 压缩 HDF5 数据需要安装并使用 `hdf5plugin`。
- HF checkpoint 中的模型配置是 Hydra 风格，不能直接用普通 `dict` 初始化模型。
- PyTorch 新版本默认 `weights_only` 行为会影响完整模型对象反序列化，因此最终改为保存纯 `state_dict`。

转换后的 checkpoint 布局为：

```text
$STABLEWM_HOME/checkpoints/models--pusht--lewm/weights.pt
$STABLEWM_HOME/checkpoints/models--pusht--lewm/config.json
```

之后我们跑通了 PushT 小规模评估：

```text
success_rate: 50.0
episode_successes: [False, True]
```

这一结果本身不是最终性能结论，但它证明了：

```text
数据 -> checkpoint -> LeWM model -> CEM planner -> PushT environment
```

这条完整链路已经可用。

### 2.3 Frozen LeWM downstream policy 计划

在 baseline 跑通后，我们开始思考：是否可以冻结 LeWM 主干，只训练一个更快的 action policy。

LeWM 主干提供：

$$
z_t = f_\theta(o_t), \quad z_g = f_\theta(o_g)
$$

冻结 encoder 和 predictor 后，新增 policy：

$$
\pi_\psi(z_t, z_g, z_g - z_t) \rightarrow a_{t:t+K}
$$

这个尝试的核心问题是：

```text
冻结 LeWM latent 是否足够支持一个 goal-conditioned policy，
从而部分替代原始 CEM planner？
```

我们因此形成了中文计划文档，明确了：

- CEM baseline 应记录哪些指标。
- latent dataset 应如何构造。
- action normalization、image transform、episode split 必须和 `eval.py` 保持一致。
- 新 policy 应该和原始 CEM 在相同 `goal_offset_steps`、相同 eval episodes 和相同 eval budget 下比较。

### 2.4 第一版 Latent Goal-Conditioned BC(VLA-JEPA)

随后我们实现了第一版 latent behavior cloning baseline。

它的输入是：

$$
x_t = \mathrm{concat}(z_t, z_g, z_g - z_t)
$$

输出是专家动作片段：

$$
\pi_\psi(x_t) = a_{t:t+K}
$$

对应实现包括：

```text
latent_bc.py
train_latent_bc.py
eval_latent_bc.py
scripts/build_latent_bc_dataset.py
notes/latent_goal_conditioned_bc_plan.md
```

这一版的价值是建立最小 baseline。它把问题从原始 CEM planning 改成监督学习，便于测试 LeWM latent 是否含有足够的控制信息。

但我们也观察到它的局限：

- 模型结构偏弱，只是直接从 `(z_t, z_g, z_g - z_t)` 回归动作。
- LeWM 的世界预测能力没有真正参与 action selection。
- 如果专家动作存在多模态，确定性 BC 容易学出平均动作。

因此，这条线更适合作为 baseline，而不是主要冲刺方向。

### 2.5 Latent ACT 与 latent-aware action prediction

在普通 latent BC 之后，我们尝试让 action policy 不只是预测动作，还显式预测未来 latent。

这条思路大致是：

```text
z_t, z_g -> action chunk
z_t, action chunk -> predicted future latent
```

训练时加入 latent auxiliary loss，希望 policy 的动作不仅接近专家动作，也能在 latent 空间中朝正确未来状态推进。

这个尝试帮助我们意识到一个关键问题：

```text
只加 latent auxiliary loss 还不够。
模型需要更明确地知道短期应该到达哪个 latent subgoal。
```

因此后续方法从“附带预测 latent”进一步转向“先预测短期 subgoal，再生成动作”。

### 2.6 Latent Subgoal ACT

本周最重要的尝试是 `latent_subgoal_act`。

它把控制问题拆成三段：

```text
1. 短期 subgoal:
   z_t, z_g -> z_hat_{t+H}

2. 动作生成:
   z_t, z_g, z_hat_{t+H} -> a_{t:t+K}

3. LeWM 检查:
   z_t, a_{t:t+K} -> z_rollout_{t+K}
```

直觉是：

```text
先想清楚短期应该到哪里，再决定这几步怎么走。
```

训练损失包括：

$$
L_{\mathrm{action}} =
\left\| \hat{a}_{t:t+K} - a_{t:t+K}^{\mathrm{expert}} \right\|_2^2
$$

$$
L_{\mathrm{subgoal}} =
\left\| \hat{z}_{t+H} - z_{t+H} \right\|_2^2
$$

可选的 LeWM consistency loss 为：

$$
L_{\mathrm{rollout}} =
\left\| \mathrm{LeWM}(z_t, \hat{a}_{t:t+K}) - z_{t+H} \right\|_2^2
$$

推理时可以使用 LeWM local rerank 或 CEM。此时 CEM 不再直接追远期目标 `z_g`，而是追 policy 预测的短期 subgoal：

$$
\min_a
\left\|
\mathrm{LeWM}(z_t, a_{t:t+K}) - \hat{z}_{t+H}
\right\|_2^2
$$

这条线的优势是：它真正把 LeWM 的 latent rollout 能力放回 action selection 中，而不只是把 LeWM 当成一个 frozen encoder。

### 2.7 Episode-level split 与数据泄露控制

为了保证训练和评估公平，我们专门加入了 episode-level split。

关键设置包括：

```text
split=train/test/all
split_seed=42
test_fraction=0.1
```

数据集构建和评估使用同一套 split 规则，避免同一条 trajectory 同时出现在训练集和测试集中。

这件事很重要，因为如果只按 sample 随机划分，模型可能在训练时看到同一个 episode 的相邻状态，从而高估泛化能力。

### 2.8 Goal-anchored sampling 与 subgoal 设计

我们还尝试了不同的数据采样方式。

基础方式是 fixed offset：

```text
goal = t + 25
subgoal = t + 5
action chunk = a_{t:t+4}
```

后来引入 goal-anchored sampling：

```text
固定一个 goal，例如 z_25
从多个 start 生成样本：
z_1  -> z_25
z_2  -> z_25
...
z_24 -> z_25
```

这样可以让同一个 goal 对应多个中间状态，增强 goal-conditioned policy 的训练覆盖度。

### 2.9 K=5/T=5 小规模实验

我们生成了 128k PushT latent dataset，并启动了 K=5/T=5 的小规模实验。

数据集设置大致为：

```text
goal_offset_steps = 25
action_horizon = 25
subgoal_horizon = 25
max_samples = 128000
```

训练时先切出：

```text
action_horizon = 5
subgoal_horizon = 5
```

也就是先验证短期 subgoal control 是否可行。

这一阶段的观察是：

- 小规模训练链路已经建立。
- CEM eval 如果直接用 `num_eval=100`、`num_candidates=300`、`num_iters=30` 会非常重。
- 后续应先使用轻量 CEM eval 确认链路稳定，再扩大规模。

### 2.10 Diffusion Action Prior

由于 deterministic BC/ACT 容易平均化多模态动作，我们也规划了 diffusion action prior。

目标是学习：

$$
p_\psi(a_{t:t+K} \mid z_t, z_g)
$$

而不是只输出一个确定性动作序列。

当前规划的重点是 K=25：

```text
goal_offset_steps = 25
action_horizon = 25
diffusion.num_steps = 50
train.epochs = 400
```

后续可以把 diffusion samples 作为 CEM candidates 或 action proposal，再用 frozen LeWM rollout 进行筛选。

## 3. 已获得成果

### 3.1 工程链路成果

本周已经获得的工程成果包括：

- LeWM 远程环境可用。
- PushT 和 TwoRoom 数据准备完成。
- PushT checkpoint 转换成功。
- PushT LeWM-CEM 小规模评估跑通。
- 实验归档目录和脚本规范建立。
- 数据、checkpoint、实验产物与代码仓库分离。
- `latent_bc` baseline 实现完成。
- `latent_subgoal_act` 方法框架基本完成。
- dataset build、inspect、train、eval、archive 闭环建立。

### 3.2 研究路线成果

研究上，我们完成了从复现到改进的过渡。

现在的主问题已经比较清楚：

```text
原始 LeWM 用 CEM 在 latent space 中搜索动作，成功但推理较慢。
我们希望冻结 LeWM 主干，训练一个 learned action prior 或 subgoal-conditioned policy。
policy 给出更好的动作候选，LeWM rollout 负责验证或重排。
```

这条路线的目标不是马上完全替代 CEM，而是逐步证明：

- LeWM latent 可以作为 downstream control 的状态表示。
- learned policy 可以提供比随机 Gaussian 更好的 action prior。
- LeWM verifier/rerank 可以弥补纯 BC 的不足。
- subgoal-conditioned objective 可能比直接追远期 goal 更适合短 horizon control。

### 3.3 方法判断成果

我们也形成了一些判断：

- Plain latent BC 是必要 baseline，但不应作为主要路线。
- 只预测动作不够，应该让 policy 显式理解短期 latent subgoal。
- LeWM 不应只作为 encoder 使用，还应通过 rollout 参与 action selection。
- CEM 参数过重时评估成本很高，必须先做轻量调试。
- 对 PushT 这类任务，diffusion action prior 可能比 deterministic policy 更合适。

## 4. 当前问题与观察

### 4.1 当前实验记录还需要补齐

`latent_subgoal_act/EXPERIMENT_LOG.md` 中仍有若干结果待补充：

```text
inspect_dataset 是否通过
K=5/T=5 训练是否完成
best epoch
val_score
policy.pt 路径
direct eval 结果
轻量 CEM eval 结果
```

这些是下一步最应该先补完的内容。

### 4.2 K=5/T=5 与原始 CEM 不是严格同一目标

当前 K=5/T=5 的 CEM 目标是：

$$
\hat{z}_{t+5}
$$

而原始 LeWM-CEM baseline 的目标是：

$$
z_g = z_{t+25}
$$

因此 K=5/T=5 更像局部 subgoal CEM，不能直接说它严格对齐原始 25-step CEM。

如果后续要和原始 baseline 做严格比较，需要训练 K=25/T=25 policy，或者在报告中明确说明两者目标不同。

### 4.3 评估参数需要分层

目前重 CEM 设置计算量过大：

```text
eval.num_eval = 100
world.num_envs = 100
cem.num_candidates = 300
cem.num_iters = 30
```

这会导致每次 replanning 的 candidate rollout 数量非常大。

更合理的顺序是：

```text
先 direct eval
再 local rerank
再轻量 CEM
最后才扩大到正式 CEM 设置
```

## 5. 未来努力方向

### 5.1 补齐当前实验闭环

优先完成：

```text
dataset inspect
K=5/T=5 train result
direct eval
default rerank eval
lightweight CEM eval
```

先确认链路稳定，再扩大参数。

### 5.2 做公平 baseline 对比

后续所有方法都应尽量固定：

```text
eval episodes
goal_offset_steps
image transform
action normalization
eval_budget
receding_horizon
split_seed
test_fraction
```

否则 success rate 的差异可能来自评估条件，而不是方法本身。

### 5.3 系统做消融实验

建议下一轮至少比较：

```text
1. plain latent BC
2. latent ACT
3. latent subgoal ACT direct
4. latent subgoal ACT + local rerank
5. latent subgoal ACT + temporal ensemble
6. latent subgoal ACT + CEM to predicted subgoal
7. diffusion action prior
8. diffusion action prior + LeWM/CEM verification
```

这样可以判断瓶颈到底来自：

- subgoal 学不好；
- action head 学不好；
- LeWM rollout consistency 不够；
- 推理阶段 candidate selection 不够；
- action distribution 多模态导致 deterministic policy 平均化。

### 5.4 推进 K=25/T=25 版本

为了更严格对齐原始 PushT CEM baseline，需要推进：

```text
goal_offset_steps = 25
action_horizon = 25
subgoal_horizon = 25
```

这样 CEM-to-subgoal 和原始 CEM-to-goal 的比较会更清楚。

### 5.5 重点推进 diffusion action prior

如果 deterministic subgoal ACT 的动作成功率不够，下一步应重点尝试 diffusion prior。

原因是 PushT 的专家动作可能存在多种可行路径，确定性 MSE policy 容易输出平均动作，而 diffusion 可以建模多峰动作分布。

后续可以采用：

```text
diffusion sample 多个 action chunks
用 LeWM rollout 计算 terminal latent cost
选择最接近目标或 subgoal 的 candidate
```

### 5.6 最终研究目标

最终可以形成的研究 claim 是：

```text
冻结 LeWM latent world model 后，
学习式 action prior / subgoal policy 可以降低 CEM 搜索成本；
LeWM rollout verifier 可以提高纯 behavior cloning 的可靠性；
短期 latent subgoal 为 long-horizon goal-conditioned control 提供了更稳定的中间目标。
```

换句话说，我们这一周完成了从“复现 LeWM”到“基于 LeWM 做 downstream control 改进”的第一阶段转换。

