# 公平 A-B Trade-off 对照论文集

更新时间：2026-07-17

这个论文集关注的不是某一种具体方法，而是论文如何在尽量公平的 setting 下研究两种互补能力之间的 trade-off，例如：

- planning/search 与 learned policy；
- model-based 与 model-free；
- test-time compute 与 model capacity；
- 模型规模、数据量和计算预算之间的分配。

## 1. 与当前 model-policy 问题最相关的论文

| 优先级 | 论文 | A | B | 公平对照方式 | 可借鉴点 |
|---|---|---|---|---|---|
| S | [Planning vs Offline RL](./01_planning_search_vs_policy/2502.14819_Planning-vs-Offline-RL.pdf) | JEPA latent dynamics + planning | Goal-conditioned/model-free offline RL | 相同 reward-free offline datasets，扫描数据质量、数量、环境变化和泛化维度 | 最接近 LeWM planner vs learned policy 的受控对照 |
| S | [PIDM vs BC](./01_planning_search_vs_policy/2601.21718_PIDM-vs-BC.pdf) | Future predictor + IDM | Direct behavior cloning | 相同 demonstrations，理论分解和数据规模曲线 | 把收益拆成 action uncertainty reduction 与 predictor bias |
| S | [Expert Iteration](./01_planning_search_vs_policy/1705.08439_Expert-Iteration.pdf) | Tree search expert | Amortized policy apprentice | Search 产生更强 target，再蒸馏给 policy | 研究 strong planner 如何训练 small policy |
| S | [AlphaZero](./01_planning_search_vs_policy/1712.01815_AlphaZero.pdf) | MCTS | Policy/value network | 控制 search simulations，比较 network-only 与 search-enhanced action | 扫描 policy strength 与 search budget |
| S | [MuZero](./01_planning_search_vs_policy/1911.08265_MuZero.pdf) | Learned model + MCTS | Policy/value prediction | 同一系统消融 model、search simulations 和 policy/value | model、policy、test-time planning 三方分配 |
| S | [Think Too Fast Nor Too Slow](./01_planning_search_vs_policy/2005.07404_Planning-Learning-Compute-Tradeoff.pdf) | 更多 planning compute | 更多 learning/acting compute | 固定总计算预算，扫描从 exhaustive search 到 model-free RL 的连续谱 | 直接借鉴为 model-policy-search iso-compute 曲线 |
| S | [FlowMPC](./01_planning_search_vs_policy/2606.16286_FlowMPC.pdf) | World model + MPPI | Flow-matching policy-only | 相同 imitation policy、任务和数据，仅增加 WM test-time planning | 最接近 LeWM + action head 与 direct policy 的端点/混合对照 |
| A | [Imagination-Augmented Agents](./01_planning_search_vs_policy/1707.06203_Imagination-Augmented-Agents.pdf) | Model rollout | Model-free controller | 同任务比较无 imagination、固定使用 imagination 和 learned use | 让 policy 学习何时使用 simulator |
| A | [Policy Distillation](./01_planning_search_vs_policy/1511.06295_Policy-Distillation.pdf) | Teacher/search policy | Student direct policy | 相同环境与教师行为，比较蒸馏前后速度和表现 | 将慢 planner 摊销成快 action policy |
| A | [POPLIN](./01_planning_search_vs_policy/1906.08649_POPLIN.pdf) | Policy-guided MPC | Distilled policy direct execution | 同一 dynamics model 与 policy，比较 action-space planning、parameter-space planning 和 direct policy | policy proposal + lightweight CEM 的直接先例 |
| A | [IMPLANT](./01_planning_search_vs_policy/2204.03597_IMPLANT.pdf) | Decision-time planning | Base imitation policy | 保留同一 policy/reward model，只改变测试时是否规划并施加 dynamics perturbation | 检验 search 是否主要补偿 policy covariate shift |

## 2. Planning/Search vs Learned Policy

### 已验证本地 PDF

1. **Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models**
   - arXiv:2502.14819。
   - 对比 latent JEPA planning、goal-conditioned RL 和 zero-shot RL。
   - 使用 23 个不同质量的 offline datasets，并评估 unseen layouts、unseen tasks、trajectory stitching 等六种能力。
   - 关键结论不是“planning 总是更强”，而是 model-free RL 更依赖大量高质量数据，而 latent planning 在低质量数据和新布局上更有优势。

2. **When Does Predictive Inverse Dynamics Outperform Behavior Cloning?**
   - arXiv:2601.21718。
   - A 是 future-state predictor + IDM，B 是 direct BC。
   - 公平点是相同 expert demonstrations 和相同任务，扫描数据规模。
   - 核心 trade-off：future conditioning 降低 action conditional variance，但 future predictor 引入 bias、variance 和 covariate shift。

### 已验证本地 PDF

3. **[Thinking Fast and Slow with Deep Learning and Tree Search](./01_planning_search_vs_policy/1705.08439_Expert-Iteration.pdf)**
   - arXiv:1705.08439，Expert Iteration。
   - A 是强但慢的 tree-search expert，B 是快但近似的 neural policy。
   - 采用 search -> distillation -> improved policy 的循环，而不是只比较两个端点。

4. **[Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](./01_planning_search_vs_policy/1712.01815_AlphaZero.pdf)**
   - arXiv:1712.01815，AlphaZero。
   - A 是 MCTS deliberation，B 是 policy/value network 的 amortized prediction。
   - 最值得借鉴的是扫描 search simulations，并报告 network-only 与 MCTS 的差距。

5. **[Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](./01_planning_search_vs_policy/1911.08265_MuZero.pdf)**
   - arXiv:1911.08265，MuZero。
   - 同时包含 learned dynamics、policy/value 和 MCTS。
   - 是研究 model-policy-search 三方 trade-off 的标准结构。

6. **[Imagination-Augmented Agents for Deep Reinforcement Learning](./01_planning_search_vs_policy/1707.06203_Imagination-Augmented-Agents.pdf)**
   - arXiv:1707.06203。
   - 比较 model-free controller、固定使用 imagined rollout，以及由 policy 学习如何解释/使用 rollout 的 hybrid。

7. **[Policy Distillation](./01_planning_search_vs_policy/1511.06295_Policy-Distillation.pdf)**
   - arXiv:1511.06295。
   - 研究强而慢的 teacher policy 与小而快的 student policy 之间的性能-效率差距。
   - 对 GC-IDM 蒸馏 CEM、policy proposal 初始化 planner 很有参考价值。

8. **[Think Too Fast Nor Too Slow: The Computational Trade-off Between Planning And Reinforcement Learning](./01_planning_search_vs_policy/2005.07404_Planning-Learning-Compute-Tradeoff.pdf)**
   - arXiv:2005.07404。
   - 在 planning、learning 与 acting 之间分配有限计算预算，构造从 exhaustive search 到 model-free RL 的连续谱。
   - 关键结论是最优点通常位于中间，而非任一端点；适合作为 LeWM model-policy-search iso-compute 实验的直接模板。

9. **[Exploring Model-based Planning with Policy Networks](./01_planning_search_vs_policy/1906.08649_POPLIN.pdf)**
   - arXiv:1906.08649，POPLIN。
   - 用 policy 初始化 action-space planning，或直接在 policy parameter space 中规划，并比较蒸馏 policy 的直接执行。
   - 与“action head proposal + small-budget CEM/LeWM rerank”几乎同构，可直接借鉴 policy-only、random-init MPC、policy-init MPC 三组消融。

10. **[Imitating, Fast and Slow: Robust Learning from Demonstrations via Decision-time Planning](./01_planning_search_vs_policy/2204.03597_IMPLANT.pdf)**
    - arXiv:2204.03597，IMPLANT。
    - 在相同 imitation policy 基础上增加 decision-time planning，重点测试 dynamics perturbation 下的鲁棒性。
    - 可借鉴为 direct action head 与 planner-corrected action head 的 OOD/闭环误差对照。

11. **[Dual Policy Iteration](./01_planning_search_vs_policy/1805.10755_Dual-Policy-Iteration.pdf)**
    - arXiv:1805.10755。
    - 交替优化 fast reactive policy 与 slow non-reactive planner：planner 提供监督，policy 又反过来引导 planner。
    - 为 planner-policy co-training 提供理论框架，也能帮助区分一次性蒸馏与持续交替改进。

12. **[FlowMPC: Improving Flow Matching Policies with World Models](./01_planning_search_vs_policy/2606.16286_FlowMPC.pdf)**
    - arXiv:2606.16286。
    - 在相同 flow-matching imitation policy 上加入 learned world model 与 MPPI，对比 policy-only 和 WM-planning hybrid。
    - 与当前 LeWM/action-head 方向最接近，应重点看 candidate proposal、planning budget、终局 success 和 policy prior 的控制方式。

## 3. Model-Based vs Model-Free

1. **[When to Trust Your Model: Model-Based Policy Optimization](./02_model_based_vs_model_free/1906.08253_MBPO.pdf)**
   - arXiv:1906.08253，MBPO。
   - A 是更多、更长的 model rollout，B 是更依赖真实数据的 model-free update。
   - 扫描 synthetic rollout horizon，展示模型利用收益与 model bias 的 crossover。

2. **[Model-Based Value Expansion for Efficient Model-Free Reinforcement Learning](./02_model_based_vs_model_free/1803.00101_Model-Based-Value-Expansion.pdf)**
   - arXiv:1803.00101，MVE。
   - 比较纯 model-free target 与加入不同长度 model rollout 的 value target。
   - 适合学习如何用连续超参数构造 A-B 中间点。

3. **[Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models](./02_model_based_vs_model_free/1805.12114_PETS.pdf)**
   - arXiv:1805.12114，PETS。
   - 将 probabilistic model + planning 与主流 model-free RL 按环境交互量进行比较。
   - 重点是 sample efficiency，而不是只控制网络参数量。

4. **[Benchmarking Model-Based Reinforcement Learning](./02_model_based_vs_model_free/1907.02057_Benchmarking-MBRL.pdf)**
   - arXiv:1907.02057。
   - 在统一的 18 个以上环境与噪声设置中比较多类 MBRL 方法，减少各论文自定义任务造成的不可比性。
   - 提炼出 dynamics bottleneck、planning-horizon dilemma 与 early-termination dilemma，适合指导 LeWM 对 action horizon 和 model error 的扫描。

## 4. LLM Test-Time Compute vs Model Capacity

1. **[Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](./03_test_time_compute_vs_model_capacity/2408.03314_Scaling-Test-Time-Compute.pdf)**
   - arXiv:2408.03314。
   - A 是更强 base model，B 是给较小模型更多 test-time search/verification compute。
   - 固定总 inference compute，研究不同难度问题上的 crossover point。

2. **[Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](./03_test_time_compute_vs_model_capacity/2407.21787_Large-Language-Monkeys.pdf)**
   - arXiv:2407.21787。
   - 扫描 sampling budget，比较更大模型与更多重复采样。
   - 适合借鉴 success-compute Pareto curve。

3. **[s1: Simple test-time scaling](./03_test_time_compute_vs_model_capacity/2501.19393_s1-Test-Time-Scaling.pdf)**
   - arXiv:2501.19393。
   - 研究 inference budget、reasoning length 和模型能力的关系。

4. **[Distilling System 2 into System 1](./03_test_time_compute_vs_model_capacity/2407.06023_Distilling-System-2-into-System-1.pdf)**
   - arXiv:2407.06023。
   - A 是昂贵的显式推理过程，B 是蒸馏后的直接 System-1 prediction。
   - 与 planner -> policy amortization 高度同构。

5. **[Self-Consistency Improves Chain of Thought Reasoning in Language Models](./03_test_time_compute_vs_model_capacity/2203.11171_Self-Consistency.pdf)**
   - arXiv:2203.11171。
   - 比较单路径 greedy reasoning 与多路径 sampling/voting。
   - 展示如何单独扫描 test-time candidate 数量。

## 5. 固定总预算的 Scaling 研究

1. **[Training Compute-Optimal Large Language Models](./04_fixed_budget_scaling/2203.15556_Chinchilla.pdf)**
   - arXiv:2203.15556，Chinchilla。
   - 在固定 training compute 下分配 model parameters 与 training tokens。
   - 最值得借鉴的是 iso-compute curves，而不是单次等参数比较。

2. **[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](./04_fixed_budget_scaling/1905.11946_EfficientNet.pdf)**
   - arXiv:1905.11946。
   - 固定 FLOPs 下研究 depth、width、resolution 的联合分配。

3. **[Scaling Laws for Neural Language Models](./04_fixed_budget_scaling/2001.08361_Scaling-Laws.pdf)**
   - arXiv:2001.08361。
   - 建立参数、数据、计算与 loss 的经验关系。
   - 对 model-policy 容量分配的启发是拟合趋势和 crossover，而不是只报告哪个端点更好。

4. **[PonderNet: Learning to Ponder](./04_fixed_budget_scaling/2107.05407_PonderNet.pdf)**
   - arXiv:2107.05407。
   - 让模型学习每个样本需要多少计算步骤，研究 accuracy-compute trade-off。

5. **[Adaptive Computation Time for Recurrent Neural Networks](./04_fixed_budget_scaling/1603.08983_Adaptive-Computation-Time.pdf)**
   - arXiv:1603.08983。
   - 通过 learned halting 分配可变 test-time compute。

## 6. 这些论文共同使用的研究模板

### 模板 A：固定预算扫描分配比例

$$
C_A+C_B=C_{\mathrm{total}},
\qquad
\alpha=\frac{C_A}{C_{mathrm{total}}}.
$$

扫描 $\alpha$，而不是只比较两个端点。

### 模板 B：端点 + Hybrid

$$
A,\qquad B,\qquad A+B.
$$

例如 pure planner、direct policy、policy proposal + lightweight planning。

### 模板 C：扫描任务条件并寻找 crossover

$$
J=J(\text{data},\text{horizon},\text{OOD},\text{compute budget}).
$$

结论应是“什么条件下谁更好”，而不是一个平均分。

### 模板 D：报告 Pareto frontier

至少同时报告：

- task success/accuracy；
- train-time compute；
- test-time compute/latency；
- data/environment interactions；
- 参数量和内存。

## 7. 对 LeWM model-policy 研究的直接映射

建议构造：

1. LeWM + CEM：model-heavy/search-heavy；
2. Direct GC-IDM/ACT/diffusion：policy-heavy；
3. Policy proposal + LeWM rerank：hybrid；
4. Policy proposal + small-budget CEM：hybrid；
5. Frozen LeWM + action-head GRPO：将 planning signal 摊销进 policy。

然后在固定数据下扫描：

$$
\text{goal horizon},\quad
\text{training data},\quad
\text{planner budget},\quad
\text{policy capacity}.
$$

最终报告：

$$
\text{success rate}
\quad\text{vs.}\quad
\text{latency / model calls / training data}.
$$

## 8. 下载状态

- 已验证并保存：26 篇，总大小约 84.7 MiB。
- 结构检查：26/26 通过，无损坏 PDF；本次新增论文页数与首页文本均已检查。
- 首页标题检查：26/26 与文件名及 arXiv 编号匹配。
- `download_papers.ps1` 保留用于以后补下载或在其他机器复现该论文集；脚本现在会自动覆盖小于 100 KB 或缺少 `%PDF-` 文件头的中断文件。
