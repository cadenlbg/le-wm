# World Model / VLA + RL Post-training 论文集

更新时间：2026-07-14  
收集范围：World Model、VLA、imagined rollout、robot policy 与 RL post-training。当前共收录 **33 篇**论文。

## 1. 如何理解这批论文

这批工作主要分成三类：

1. **World model 作为 RL 环境**：冻结或迭代更新 world model，在 imagined rollout 中用 GRPO、PPO 或其他 RL 方法更新 VLA policy。
2. **World model 本体的 RL post-training**：根据多步 rollout 的质量或任务奖励，更新 world model，使其在闭环、自回归或策略分布下更可靠。
3. **VLA RL post-training 对照组**：不一定使用 world model，但直接研究 GRPO、PPO、actor-critic、RLVR、preference optimization 等 VLA 后训练方法。

最典型的 imagined-rollout GRPO 形式是：

$$
\hat{\tau}_i \sim p_{\phi}(\hat{o}_{1:H}\mid o_0,a_{0:H-1}^{(i)}),
\qquad
a_{0:H-1}^{(i)} \sim \pi_{\theta_{\mathrm{old}}},
$$

$$
\hat{A}_i = \frac{R(\hat{\tau}_i)-\operatorname{mean}_{j}R(\hat{\tau}_j)}
{\operatorname{std}_{j}R(\hat{\tau}_j)+\epsilon},
$$

$$
J_{\mathrm{GRPO}}(\theta)
=
\mathbb{E}\left[
\frac{1}{G}\sum_{i=1}^{G}
\min\left(
\rho_i(\theta)\hat{A}_i,
\operatorname{clip}(\rho_i,1-\epsilon,1+\epsilon)\hat{A}_i
\right)
\right].
$$

其中，world model 参数为 $\phi$，VLA policy 参数为 $\theta$。大部分工作更新 $\theta$；PRWM、Reward as An Agent 等工作则更直接地研究如何用 RL 改善 $\phi$。

## 2. 最推荐的阅读顺序

如果重点是 **WMPO 类 imagined rollout + RL post-training**，建议按以下顺序阅读：

1. [WMPO](./01_world_model_vla_rl/2511.09515_WMPO.pdf)：最直接的 pixel world model + on-policy GRPO 基线。
2. [World-Gymnast](./01_world_model_vla_rl/2602.02454_World-Gymnast.pdf)：明确给出 model-based GRPO with world-model rollouts。
3. [WoVR](./01_world_model_vla_rl/2602.13977_WoVR.pdf)：重点解决 imagined rollout 的幻觉和长时误差。
4. [World-VLA-Loop](./01_world_model_vla_rl/2602.06508_World-VLA-Loop.pdf)：world model 与 policy 迭代共演化。
5. [ProphRL](./01_world_model_vla_rl/2511.20633_ProphRL.pdf)：面向 flow action head 的 FA-GRPO。
6. [SafeDojo](./01_world_model_vla_rl/2606.20698_SafeDojo.pdf)：Lagrangian constrained GRPO，同时优化成功率和安全性。
7. [PRWM](./02_world_model_rl_and_imagination/2603.25685_PRWM.pdf)：RL 更新 world model 本体，而不是只更新 policy。
8. [Reward as An Agent](./02_world_model_rl_and_imagination/2606.19990_Reward-as-an-Agent.pdf)：DynDiff-GRPO、rollout diversification 与 reward hacking。

## 3. 核心：World Model + VLA RL

| 论文 | RL/post-training 机制 | World model 的作用 | 主要更新对象 | 相关性 |
|---|---|---|---|---|
| [WMPO](./01_world_model_vla_rl/2511.09515_WMPO.pdf) | On-policy GRPO | Pixel-based imagined environment | VLA policy | 最直接 |
| [Reinforcing Action Policies by Prophesying / ProphRL](./01_world_model_vla_rl/2511.20633_ProphRL.pdf) | Flow-action-GRPO + FlowScale | Prophet action-to-video simulator | Flow-based VLA policy | 最直接 |
| [World-Gymnast](./01_world_model_vla_rl/2602.02454_World-Gymnast.pdf) | Model-based GRPO，trajectory-level group advantage | Action-conditioned video world model，VLM 提供 reward | VLA policy | 最直接 |
| [World-VLA-Loop](./01_world_model_vla_rl/2602.06508_World-VLA-Loop.pdf) | GRPO + iterative post-training | 同时预测视频与 binary reward | VLA policy 与 world model 迭代更新 | 最直接 |
| [RISE](./01_world_model_vla_rl/2602.11075_RISE.pdf) | Imagined advantage + advantage-conditioned training | Dynamics model + progress value model | Robot policy | 直接，但不是标准 GRPO |
| [GigaBrain-0.5M*](./01_world_model_vla_rl/2602.12099_GigaBrain-0.5M.pdf) | RAMP，world-model-conditioned advantage learning | 联合预测未来与 value | VLA policy | 直接，但偏 RECAP 风格 |
| [WoVR](./01_world_model_vla_rl/2602.13977_WoVR.pdf) | Imagined-rollout GRPO | Keyframe-initialized rollout + policy/model co-evolution | VLA policy，随后共演化 world model | 最直接 |
| [AcceRL](./01_world_model_vla_rl/2603.18464_AcceRL.pdf) | Distributed asynchronous policy-gradient RL | 可插拔 world model，提高在线样本效率 | VLA policy | 系统方向 |
| [VLA-MBPO](./01_world_model_vla_rl/2603.20607_VLA-MBPO.pdf) | Flow-Noise PPO + chunk-level branched rollout | Multi-view UMM world model | VLA policy | 直接 PPO 对照 |
| [ViVa](./01_world_model_vla_rl/2604.08168_ViVa.pdf) | RECAP / advantage-conditioned refinement | Video-generative value model | VLA policy/value model | 奖励与价值建模 |
| [Sword](./01_world_model_vla_rl/2605.07288_Sword.pdf) | 在 WoVR/PPO/GRPO pipeline 中验证 RL post-training | Style-robust simulator + dynamic latent bootstrapping | 主要改 world model，继而改善 policy RL | 模拟器可靠性 |
| [RAW-Dream](./01_world_model_vla_rl/2605.12334_RAW-Dream.pdf) | GRPO + unreliable rollout filtering | Task-agnostic world model + frozen VLM reward | VLA policy | 最直接 |
| [SafeDojo](./01_world_model_vla_rl/2606.20698_SafeDojo.pdf) | Lagrangian constrained GRPO | Interactive video WM + success classifier + safety head | VLA policy | 最直接，安全 RL |
| [WorldSample](./01_world_model_vla_rl/2607.02431_WorldSample.pdf) | Actor-critic RL + Policy-Paced Learning | 从真实 rollout 生成 synthetic transitions | Policy 与 world model 闭环改进 | Real/synthetic loop |
| [TACO](./01_world_model_vla_rl/2607.02840_TACO.pdf) | Advantage-conditioned corrective post-training | Tactile-aware WM 生成局部纠错片段 | VLA policy | 接触任务纠错 |

## 4. 世界模型本体、imagined RL 与 simulator reliability

| 论文 | 方法重点 | RL 更新对象 | 与 WMPO 的关系 |
|---|---|---|---|
| [Coupled Local and Global World Models for Efficient First Order RL](./02_world_model_rl_and_imagination/2602.06219_Coupled-World-Models-First-Order-RL.pdf) | Local/global WM、differentiable first-order policy optimization | Policy | 不使用 GRPO，但属于 WM 内 policy optimization |
| [PRWM](./02_world_model_rl_and_imagination/2603.25685_PRWM.pdf) | Contrastive RL 修正 autoregressive multi-step rollout | World model 的 LoRA、UNet/action encoder | 重点从 policy 转向 world model post-training |
| [GIRL](./02_world_model_rl_and_imagination/2604.07426_GIRL.pdf) | Information-theoretic hallucination control | Policy/world-model RL pipeline | 通用 imagined RL 相邻工作 |
| [On Training in Imagination](./02_world_model_rl_and_imagination/2605.06732_On-Training-in-Imagination.pdf) | 分析 learned dynamics/reward error 对 imagined policy optimization 的影响 | 理论分析 | 理解 WMPO 类方法的 model bias |
| [MBDPO](./02_world_model_rl_and_imagination/2605.26282_MBDPO.pdf) | Model-Based Diffusion Policy Optimization | Diffusion policy | 非 GRPO，但直接结合 latent WM、search 与 policy optimization |
| [Policy-Aware Simulator Learning](./02_world_model_rl_and_imagination/2605.29032_Policy-Aware-Simulator-Learning.pdf) | Minimax simulator learning，抑制 policy exploitation | Simulator/world model | 理解 simulator 被 RL 利用的问题 |
| [Reward as An Agent for Embodied World Models](./02_world_model_rl_and_imagination/2606.19990_Reward-as-an-Agent.pdf) | DynDiff-GRPO + agentic reward verification + rollout diversification | Embodied world model / rollout behavior | 最接近“GRPO 更新 world model rollout 能力” |

## 5. VLA RL post-training 对照组

这些论文不一定使用 world model，但可以帮助比较 RL objective、credit assignment、action head 与训练系统。

| 论文 | 主要 RL 方法 | 建议关注点 |
|---|---|---|
| [What Can RL Bring to VLA Generalization?](./03_vla_rl_post_training/2505.19789_RL-for-VLA-Generalization.pdf) | PPO，并对比 DPO/GRPO | 论文结论认为 PPO 在其设置中优于 LLM-derived GRPO/DPO |
| [AutoVLA](./03_vla_rl_post_training/2506.13757_AutoVLA.pdf) | GRPO | 同时优化驾驶推理长度与轨迹质量 |
| [SimpleVLA-RL](./03_vla_rl_post_training/2509.09674_SimpleVLA-RL.pdf) | Modified GRPO | 真实/模拟环境交互式 rollout、稀疏 outcome reward |
| [VLA-R1](./03_vla_rl_post_training/2510.01623_VLA-R1.pdf) | RLVR + GRPO | reasoning 与 execution 的可验证奖励 |
| [SOP](./03_vla_rl_post_training/2601.03044_SOP.pdf) | RECAP + HG-DAgger | 多机器人在线 post-training 系统 |
| [Probabilistic Chunk Masking](./03_vla_rl_post_training/2605.16154_Probabilistic-Chunk-Masking.pdf) | GRPO | 只对成功/失败开始分化的 action chunks 回传梯度 |
| [PAPO-VLA](./03_vla_rl_post_training/2605.19580_PAPO-VLA.pdf) | Planning-aware GRPO | 用 causal importance 强化关键 planning actions |
| [Expert-Guided GRPO for VLA Aerial Navigation](./03_vla_rl_post_training/2606.02313_Expert-Guided-GRPO-VLA.pdf) | Expert-guided GRPO | 航空导航中的意图对齐与可验证 reward |
| [FlowPRO](./03_vla_rl_post_training/2606.05468_FlowPRO.pdf) | Reward-free proximalized preference optimization | Flow-matching VLA 的非标准 RL/post-training 对照 |
| [Z-1](./03_vla_rl_post_training/2606.31846_Z-1.pdf) | Task-wise GRPO | Shared-prefix rollout、trajectory branching、reward calibration |
| [PAC-ACT](./03_vla_rl_post_training/2607.09590_PAC-ACT.pdf) | Chunk-level actor-critic | ACT policy 的 online RL post-training 与 behavior prior |

## 6. 关键研究轴

阅读时建议重点比较下面六个维度：

| 维度 | 需要回答的问题 |
|---|---|
| Rollout 来源 | 真实机器人、传统 simulator，还是 learned world model？ |
| Policy update | GRPO、PPO、actor-critic、advantage-conditioned training，还是 preference optimization？ |
| Reward | Binary success、VLM/VLM-as-judge、value model、dense progress，还是 safety cost？ |
| World model update | 冻结、offline fine-tune、policy/model co-evolution，还是 RL 直接更新？ |
| Model bias 控制 | Keyframe reset、short branched rollout、uncertainty filter、dual-noise verification，还是真实数据回灌？ |
| Action distribution | Autoregressive tokens、Gaussian、flow matching、diffusion policy，还是 deterministic action chunk？ |

## 7. 与 LeWM / JEPA 的直接连接

对于 frozen LeWM，可以把上述框架写成：

$$
a_{t:t+H-1}^{(i)} \sim \pi_\theta(\cdot\mid z_t,z_g),
\qquad
\hat{z}_{t+H}^{(i)} = f_\phi(z_t,a_{t:t+H-1}^{(i)}),
$$

$$
R_i =
-\lambda_g\left\|\hat{z}_{t+H}^{(i)}-z_g\right\|_2^2
-\lambda_u U(\hat{\tau}_i)
-\lambda_s C_{\mathrm{smooth}}(a^{(i)})
+\lambda_e R_{\mathrm{env}}^{(i)}.
$$

最稳妥的实验递进是：

1. 当前 LeWM rollout rerank / CEM；
2. Frozen LeWM + imagined-rollout GRPO，仅更新 action policy；
3. 加入 uncertainty 或 rollout validity filter；
4. 混合少量真实环境 reward，检查 world-model exploitation；
5. 最后再研究 PRWM/World-VLA-Loop 风格的 world model 与 policy 共演化。

## 8. 说明

- 本集合以 **RL post-training** 为主，不包含只有 SFT、纯 world-model pretraining 或纯 MPC planning 的论文。
- `02_world_model_rl_and_imagination` 中少数论文属于理论或通用 MBRL 邻近工作，用于解释 model bias、simulator exploitation 和 imagined policy optimization。
- arXiv 编号已写入文件名；PDF 均已完成结构、页数和首页标题检查。
