# PIDM vs. BC 第四节阅读笔记：理论分析

## 0. 论文与本节主旨

论文：*When Does Predictive Inverse Dynamics Outperform Behavior Cloning?*

本地文件：[`papers/fair_ab_tradeoff_studies/01_planning_search_vs_policy/2601.21718_PIDM-vs-BC.pdf`](../papers/fair_ab_tradeoff_studies/01_planning_search_vs_policy/2601.21718_PIDM-vs-BC.pdf)

第四节最核心的结论不是“PIDM 一定优于 BC”，而是：

> PIDM 把未来状态 $s_{t+k}$ 当作能够解释动作多模态性的中间变量。它的收益来自“知道未来后动作不确定性下降”；它的代价来自预测未来状态产生的分布偏移、模型偏差和估计方差。PIDM 是否胜出，取决于收益能否覆盖这些代价。

本节分为两部分：

1. **Generalization Gap**：比较 BC 与 PIDM 的测试时动作预测误差。
2. **Sample Efficiency Gain**：比较二者达到相同参数估计误差所需的样本量。

---

## 1. 问题设定：从 BC 到 PIDM

### 1.1 BC 直接预测动作

行为克隆直接学习专家策略

$$
\pi^\star(a_t\mid s_t).
$$

在均方误差下，BC 的损失为

$$
\mathcal L_{\mathrm{bc}}(\pi_\mu)
=
\mathbb E_{(s_t,a_t)\sim\mathcal D,\,
\hat a_t\sim\pi_\mu(\cdot\mid s_t)}
\left[\ell(\hat a_t,a_t)\right].
$$

困难在于，同一个当前状态 $s_t$ 下可能存在多个合理动作。例如在岔路口，专家可能左转或右转。BC 只看到当前状态，不知道专家最终打算去哪里，因此必须直接拟合一个多模态动作分布。

### 1.2 PIDM 显式引入未来状态

PIDM 包含两个模块：

- 状态预测器 $p$：根据 $s_t$ 预测未来状态 $s_{t+k}$；
- 逆动力学策略 $\pi_\xi$：根据 $(s_t,s_{t+k})$ 预测当前动作 $a_t$。

状态预测器和 IDM 的训练目标分别为

$$
\mathcal L_{\mathrm{sp}}(\hat p)
=
\mathbb E
\left[
\ell(\hat s_{t+k},s_{t+k})
\right],
$$

$$
\mathcal L_{\mathrm{idm}}(\pi_\xi)
=
\mathbb E
\left[
\ell(\hat a_t,a_t)
\right].
$$

专家策略可以分解为

$$
\boxed{
\pi^\star(a_t\mid s_t)
=
\int_{\mathcal S}
p^\star(s_{t+k}\mid s_t)
\pi^\star_{\mathrm{idm}}(a_t\mid s_t,s_{t+k})
\,\mathrm d s_{t+k}.
}
\tag{4}
$$

其中：

- $p^\star(s_{t+k}\mid s_t)$ 是专家数据诱导的真实未来状态分布；
- $\pi^\star_{\mathrm{idm}}(a_t\mid s_t,s_{t+k})$ 回答：“当前在 $s_t$，若未来要到达 $s_{t+k}$，现在应采取什么动作？”

式 (4) 本质上是对未来状态这个隐变量进行边缘化。PIDM 不直接拟合边缘化后的复杂策略，而是先选出一个未来意图，再解决条件化后更简单的逆动力学问题。

---

## 2. 4.1 Generalization Gap：未来信息能降低多少误差

### 2.1 期望预测误差

记当前状态的边缘分布为 $d$。BC 的 expected prediction error 定义为

$$
\operatorname{EPE}(f)
\triangleq
\mathbb E_{s_t\sim d,\,a_t\sim\pi^\star(\cdot\mid s_t)}
\left[
(a_t-f(s_t))^2
\right].
\tag{5}
$$

IDM 在未来状态分布 $p$ 下的误差为

$$
\operatorname{EPE}(f;p)
\triangleq
\mathbb E_{s_t\sim d,\,
s_{t+k}\sim p(\cdot\mid s_t),\,
a_t\sim\pi^\star_{\mathrm{idm}}(\cdot\mid s_t,s_{t+k})}
\left[
(a_t-f(s_t,s_{t+k}))^2
\right].
\tag{6}
$$

分号后的 $p$ 强调：IDM 的测试误差取决于向它提供什么未来状态分布。

### 2.2 平方误差下的 Bayes 最优预测器

在平方误差下，最优点预测器是条件均值：

$$
f^\star_{\mathrm{bc}}(s_t)
=
\mathbb E[a_t\mid s_t],
\tag{7}
$$

$$
f^\star_{\mathrm{idm}}(s_t,s_{t+k})
=
\mathbb E[a_t\mid s_t,s_{t+k}].
\tag{8}
$$

作者定义在真实未来状态分布 $p^\star$ 下的误差差距：

$$
\Delta_{p^\star}
\triangleq
\operatorname{EPE}(f^\star_{\mathrm{bc}})
-
\operatorname{EPE}(f^\star_{\mathrm{idm}};p^\star).
\tag{9}
$$

若 $\Delta_{p^\star}>0$，说明在能够观察真实未来状态时，IDM 的动作预测误差更低。

### 2.3 定理 1：未来条件带来的不确定性下降

定理 1 给出

$$
\boxed{
\Delta_{p^\star}
=
\mathbb E_{s_t\sim d}
\left[
\operatorname{Var}_{s_{t+k}\sim p^\star(\cdot\mid s_t)}
\left(
f^\star_{\mathrm{idm}}(s_t,s_{t+k})
\right)
\right]
\ge 0.
}
\tag{10}
$$

它直接来自条件全方差公式：

$$
\operatorname{Var}(a_t\mid s_t)
=
\mathbb E
\left[
\operatorname{Var}(a_t\mid s_t,s_{t+k})
\mid s_t
\right]
+
\operatorname{Var}
\left(
\mathbb E[a_t\mid s_t,s_{t+k}]
\mid s_t
\right).
$$

三项分别表示：

- $\operatorname{Var}(a_t\mid s_t)$：BC 在只知道当前状态时面对的全部动作不确定性；
- $\mathbb E[\operatorname{Var}(a_t\mid s_t,s_{t+k})\mid s_t]$：知道未来以后仍然无法消除的动作噪声；
- $\operatorname{Var}(\mathbb E[a_t\mid s_t,s_{t+k}]\mid s_t)$：不同未来意图所造成的平均动作差异。

因此，PIDM 的理想收益正是“未来状态能够解释的那部分动作方差”。

PIDM 最适合如下数据结构：

$$
\text{相同 }s_t
\quad\longrightarrow\quad
\text{不同 }s_{t+k}
\quad\longrightarrow\quad
\text{明显不同的 }a_t.
$$

典型例子包括岔路口、绕障碍物的左右选择，以及从同一位置朝不同目标移动。

若所有未来意图对应的当前动作基本相同，则

$$
f^\star_{\mathrm{idm}}(s_t,s_{t+k})
\approx
f^\star_{\mathrm{bc}}(s_t),
$$

从而 $\Delta_{p^\star}\approx 0$，PIDM 没有结构性优势。

### 2.4 定理 1 没有证明什么

定理 1 只讨论：

- Bayes 最优预测器；
- 真实未来状态分布 $p^\star$；
- 平方误差下的点预测；
- population-level prediction error。

因此，它证明的是“额外的真实条件信息不会伤害最优预测器”，并没有证明实际训练得到的 PIDM 一定优于 BC。实际系统还需要预测未来，而未来预测误差会引入新的代价。

---

## 3. 实际 PIDM 的关键代价：未来状态分布偏移

训练 IDM 时，未来状态来自真实数据：

$$
s_{t+k}\sim p^\star(\cdot\mid s_t).
$$

测试时无法访问真实未来，只能使用状态预测器：

$$
\hat s_{t+k}\sim\hat p(\cdot\mid s_t).
$$

于是 IDM 的训练输入分布和测试输入分布不同，形成 PIDM 特有的 covariate shift。

### 3.1 Assumption 1：支持集包含关系

作者要求

$$
\boxed{
\operatorname{supp}\hat p(\cdot\mid s_t)
\subseteq
\operatorname{supp}p^\star(\cdot\mid s_t),
\qquad \forall s_t\in\mathcal S.
}
$$

这个假设保证状态预测器不会产生专家未来分布完全没有覆盖的状态，并使密度比

$$
w(s_t,s_{t+k})
\triangleq
\frac{\hat p(s_{t+k}\mid s_t)}
{p^\star(s_{t+k}\mid s_t)}
\tag{11}
$$

有定义。

这是一个较强的假设。现实中的神经网络状态预测器完全可能产生 off-support observation 或 latent state；一旦发生，后续的 density-ratio 分解不能直接覆盖这些点。

### 3.2 Proposition 1：分布改变不改变逐点 Bayes 解

在支持集假设下，对于任意位于公共支持集中的 $(s_t,s_{t+k})$，Bayes 最优 IDM 仍然是

$$
f^\star_{\mathrm{idm}}(s_t,s_{t+k})
=
\mathbb E[a_t\mid s_t,s_{t+k}].
$$

它不取决于测试时使用 $p^\star$ 还是 $\hat p$。分布偏移改变的是“哪些未来状态被赋予更高权重”，而不是每个输入点上的 Bayes 最优答案。

---

## 4. Corollary 1：把实际输赢拆成四项

作者进一步引入 i.i.d. 假设。令

$$
\mathcal D_n
=
\{(s_t,a_t)\}_{i=1}^{n}
$$

为 BC 数据集，令

$$
\mathcal D_m^{p^\star}
=
\{(s_t,a_t,s_{t+k})\}_{i=1}^{m}
$$

为 IDM 数据集。有限数据训练出的预测器分别记作 $\hat f_{\mu_n}$ 和 $\hat f_{\xi_m}$。

推论 1 得到

$$
\boxed{
\widehat\Delta_{\hat p}
=
\Delta_{p^\star}
+\delta+\beta+\gamma.
}
\tag{17}
$$

这是第四节最重要的公式。实际 PIDM 相对 BC 的优势，由一个结构收益和三个有限样本或分布偏移项共同决定。

### 4.1 $\Delta_{p^\star}$：未来条件的结构收益

$$
\Delta_{p^\star}\ge 0.
$$

它衡量未来状态对当前动作的解释力，是专家联合数据分布的性质，而不是某种训练算法的性质。

### 4.2 $\delta$：估计方差差距

$$
\delta
\triangleq
\mathbb E_{s_t\sim d}
\left[
\operatorname{Var}_{\mathcal D_n}
\left(\hat f_{\mu_n}(s_t)\right)
-
\mathbb E_{s_{t+k}\sim p^\star(\cdot\mid s_t)}
\left[
w(s_t,s_{t+k})
\operatorname{Var}_{\mathcal D_m^{p^\star}}
\left(\hat f_{\xi_m}(s_t,s_{t+k})\right)
\right]
\right].
\tag{12}
$$

它比较重复采样训练集并重新训练模型时，BC 与 IDM 输出的波动：

- $\delta>0$：IDM 估计更稳定，有利于 PIDM；
- $\delta<0$：IDM 的估计方差更大，不利于 PIDM。

IDM 输入维度更高，而且测试分布由 $\hat p$ 重加权，所以实际中 $\delta$ 不一定为正。

### 4.3 $\beta$：估计偏差差距

$$
\beta
\triangleq
b^2_{\hat f_{\mu_n}}
-
b^2_{\hat f_{\xi_m}}.
\tag{13}
$$

其中 BC 的平方偏差为

$$
b^2_{\hat f_{\mu_n}}
=
\mathbb E_{s_t\sim d}
\left[
\left(
\mathbb E_{\mathcal D_n}[\hat f_{\mu_n}(s_t)]
-f^\star_{\mathrm{bc}}(s_t)
\right)^2
\right],
$$

IDM 的平方偏差还需要按照 $w$ 对测试未来状态分布进行重加权。

- $\beta>0$：BC 偏差更大，有利于 PIDM；
- $\beta<0$：PIDM 偏差更大，不利于 PIDM。

如果状态预测器把 IDM 带到训练数据很少覆盖的位置，PIDM 的偏差往往会增加，因此作者预期很多情形下 $\beta<0$。

### 4.4 $\gamma$：不可约噪声的重加权效应

$$
\gamma
\triangleq
\mathbb E_{s_t\sim d,\,s_{t+k}\sim p^\star(\cdot\mid s_t)}
\left[
(1-w(s_t,s_{t+k}))
\operatorname{Var}(a_t\mid s_t,s_{t+k})
\right].
\tag{14}
$$

令

$$
\sigma^2(s_t,s_{t+k})
=
\operatorname{Var}(a_t\mid s_t,s_{t+k}),
$$

则可将 $\gamma$ 写成更直观的形式：

$$
\gamma
=
\mathbb E_{p^\star}[\sigma^2(s_t,s_{t+k})]
-
\mathbb E_{\hat p}[\sigma^2(s_t,s_{t+k})].
$$

因此：

- 如果 $\hat p$ 更偏向动作条件方差较低、容易判断的未来，$\gamma>0$；
- 如果 $\hat p$ 更偏向动作仍然模糊的未来，$\gamma<0$。

分布偏移并非必然伤害 PIDM。它可能偶然把质量集中在更容易预测的未来状态上，从而降低不可约误差、估计偏差或估计方差。

### 4.5 正文中的一个疑似符号笔误

论文正文说“$\hat p$ 放置更多概率质量”时，同时写了 $0<w<1$。但根据

$$
w=\frac{\hat p}{p^\star},
$$

$\hat p$ 比 $p^\star$ 放置更多质量应对应 $w>1$。因此，判断 $\gamma$ 的符号时应以上面的期望差公式为准。

### 4.6 实际 PIDM 胜出的条件

因为

$$
\widehat\Delta_{\hat p}>0
$$

才表示 PIDM 的预测误差低于 BC，所以实际胜出条件为

$$
\boxed{
\Delta_{p^\star}
>
-(\delta+\beta+\gamma).
}
$$

也就是说，未来状态带来的动作消歧收益，必须覆盖状态预测器和有限样本带来的总代价。

---

## 5. 4.2 Sample Efficiency Gain：PIDM 为什么可能更省样本

这一小节不再直接研究测试时动作误差，而是研究参数估计达到目标 MSE $\varepsilon$ 所需的样本数。

### 5.1 Assumption 3：正确设定

作者假设 BC 和 IDM 的策略类都能精确表达真实策略：

$$
\exists\mu^\star\in\Theta_\mu
\quad\text{s.t.}\quad
\pi_{\mu^\star}=\pi^\star,
$$

$$
\exists\xi^\star\in\Theta_\xi
\quad\text{s.t.}\quad
\pi_{\xi^\star}=\pi^\star_{\mathrm{idm}}.
\tag{18}
$$

这是 correct specification 假设。它排除了模型容量不足和不可消除的函数逼近误差，使 Fisher 信息能够锚定到真实数据生成过程。

### 5.2 有偏 Cramér-Rao 下界

记参数估计器的偏差为

$$
b_{\hat\mu_n}(\mu)
=
\mathbb E[\hat\mu_n]-\mu,
$$

其在真实参数处的导数为

$$
b'_{\hat\mu_n}(\mu^\star)
=
\left.
\frac{\partial b_{\hat\mu_n}(\mu)}{\partial\mu}
\right|_{\mu=\mu^\star}.
$$

IDM 参数 $\hat\xi_m$ 的定义类似。

BC 和 IDM 在最优参数处的单样本 Fisher 信息分别为

$$
F_{\mu^\star}
=
\mathbb E
\left[
\left(
\left.
\frac{\partial}{\partial\mu}
\log\pi_\mu(a_t\mid s_t)
\right|_{\mu=\mu^\star}
\right)^2
\right],
\tag{21}
$$

$$
F_{\xi^\star}
=
\mathbb E
\left[
\left(
\left.
\frac{\partial}{\partial\xi}
\log\pi_\xi(a_t\mid s_t,s_{t+k})
\right|_{\xi=\xi^\star}
\right)^2
\right].
\tag{22}
$$

有偏 Cramér-Rao 下界给出

$$
\operatorname{Var}(\hat\mu_n)
\ge
\frac{
\left(1+b'_{\hat\mu_n}(\mu^\star)\right)^2
}{nF_{\mu^\star}}.
$$

作者进一步假设两种估计器在大样本下都能渐近达到该下界，因此可以近似取等号。

结合

$$
\operatorname{MSE}
=
\operatorname{Var}
+
\operatorname{Bias}^2
$$

并要求两种方法都达到同一个参数 MSE $\varepsilon$，可以分别解出 BC 所需样本数 $n$ 和 IDM 所需样本数 $m$。

### 5.3 Theorem 2：样本效率比

定义

$$
\eta
\triangleq
\frac{n}{m}.
$$

定理 2 得到

$$
\boxed{
\eta
\approx
\frac{F_{\xi^\star}}{F_{\mu^\star}}
\frac{
\left(\varepsilon-b^2_{\hat\xi_m}(\xi^\star)\right)
\left(1+b'_{\hat\mu_n}(\mu^\star)\right)^2
}{
\left(\varepsilon-b^2_{\hat\mu_n}(\mu^\star)\right)
\left(1+b'_{\hat\xi_m}(\xi^\star)\right)^2
}.
}
\tag{23}
$$

其中：

- $\eta>1$：BC 需要更多样本，PIDM 更省样本；
- $\eta=1$：二者样本效率相当；
- $\eta<1$：BC 更省样本。

式 (23) 表明，样本效率不仅由 Fisher 信息决定，也受估计偏差及其局部导数影响。

---

## 6. 为什么 IDM 的 Fisher 信息不小于 BC

### 6.1 Assumption 4：局部共享参数化

作者要求在 $\xi^\star$ 的某个邻域内，BC 策略可以表示为 IDM 策略对未来状态的边缘化：

$$
\pi_{\Psi(\xi)}(a_t\mid s_t)
=
\int_{\mathcal S}
p^\star(s_{t+k}\mid s_t)
\pi_\xi(a_t\mid s_t,s_{t+k})
\,\mathrm d s_{t+k}.
\tag{24}
$$

参数映射还满足

$$
\Psi(\xi^\star)=\mu^\star,
\qquad
\Psi'(\xi^\star)=1.
$$

这不仅要求两个最优策略相同，还要求最优点附近的策略变化能够对齐。$\Psi'(\xi^\star)=1$ 固定了局部参数尺度，防止仅通过重参数化人为放大或缩小 Fisher 信息。

### 6.2 Assumption 5：正则性

作者要求可以交换对参数的微分和对未来状态的积分：

$$
\frac{\partial}{\partial\xi}
\mathbb E_{p^\star}
\left[
\pi_\xi(a_t\mid s_t,s_{t+k})
\right]
=
\mathbb E_{p^\star}
\left[
\frac{\partial}{\partial\xi}
\pi_\xi(a_t\mid s_t,s_{t+k})
\right].
\tag{25}
$$

### 6.3 Lemma 1：Fisher 信息不等式

在 Assumptions 3--5 下，作者证明

$$
\boxed{
F_{\xi^\star}\ge F_{\mu^\star}.
}
$$

直观上：

- IDM 观察 $(s_t,s_{t+k},a_t)$；
- BC 只观察 $(s_t,a_t)$；
- BC 相当于将 $s_{t+k}$ 边缘化；
- 边缘化一个可能有信息的变量不会增加 Fisher 信息。

更具体地，BC 的 score 可以写成 IDM score 在后验未来状态分布

$$
P_\xi(s_{t+k}\mid s_t,a_t)
=
\frac{
p^\star(s_{t+k}\mid s_t)
\pi_\xi(a_t\mid s_t,s_{t+k})
}{
\pi_{\Psi(\xi)}(a_t\mid s_t)
}
$$

下的条件期望：

$$
\frac{\partial}{\partial\xi}
\log\pi_{\Psi(\xi)}(a_t\mid s_t)
=
\mathbb E_{s_{t+k}\sim P_\xi(\cdot\mid s_t,a_t)}
\left[
\frac{\partial}{\partial\xi}
\log\pi_\xi(a_t\mid s_t,s_{t+k})
\right].
$$

由 Jensen 不等式

$$
\left(\mathbb E[X]\right)^2
\le
\mathbb E[X^2]
$$

即可得到边缘策略的 Fisher 信息不超过保留未来状态条件后的 Fisher 信息。这就是一种 missing information principle。

---

## 7. Corollary 2 与理论上的 fairness

作者希望从复杂的式 (23) 得到 $\eta\ge 1$。为此，他们对齐 BC 和 IDM 的 intrinsic bias：

$$
b^2_{\hat\mu_n}(\mu^\star)
=
b^2_{\hat\xi_m}(\xi^\star),
$$

$$
b'_{\hat\mu_n}(\mu^\star)
=
b'_{\hat\xi_m}(\xi^\star).
$$

这样式 (23) 中的偏差项抵消，得到

$$
\eta
\approx
\frac{F_{\xi^\star}}{F_{\mu^\star}}
\ge 1.
$$

因此 Corollary 2 的结论是：在这些条件下，PIDM 至少与 BC 一样 sample-efficient。

### 7.1 作者如何建立公平比较

理论比较通过以下条件进行对齐：

1. 两种方法拟合同一个专家联合数据分布。
2. 两种方法使用同一种平方损失。
3. 二者以同一个目标参数 MSE $\varepsilon$ 为标准。
4. IDM 的训练未来状态来自真实 $p^\star$，不把状态预测误差混入训练期 Fisher 信息。
5. BC 和 IDM 都被假设为正确设定，均可表达真实策略。
6. 局部共享参数化保证二者在最优策略附近描述同一族策略变化。
7. $\Psi'(\xi^\star)=1$ 对齐局部参数尺度。
8. Corollary 2 进一步假设两种估计器的 intrinsic bias 和 bias derivative 相等。
9. 两种估计器都被假设能渐近达到各自的有偏 Cramér-Rao 下界。

经过这些对齐，剩余差异主要是“是否保留未来状态信息”，于是 Fisher 信息比较才具有明确含义。

### 7.2 这种 fairness 的边界

这些条件是理论上的控制变量，而不是现实中自动成立的事实：

- **相等 intrinsic bias 是假设，不是证明结果。**
- **参数 MSE 依赖参数化。** 它不一定等价于动作误差或任务成功率。
- **PIDM 还需训练状态预测器。** 该模块的数据和计算成本没有进入 $n/m$。
- **样本内容并不完全相同。** BC 样本是 $(s_t,a_t)$，IDM 样本是 $(s_t,a_t,s_{t+k})$。
- **结论是渐近的。** 小样本下未必达到 Cramér-Rao 下界。
- **支持集假设很强。** 实际预测器可能产生 off-support future state。
- **轨迹通常不是 i.i.d.。** 附录认为在平稳、遍历和充分 mixing 的 Markov 过程中，全方差信息收益仍成立，Fisher 信息也会随轨迹长度近似线性增长，但严格形式需要使用整条轨迹的联合似然。

因此，Corollary 2 最准确的解读是：

> 当表达能力、局部参数尺度、固有偏差和渐近估计效率都对齐后，保留未来状态比将其边缘化携带更多 Fisher 信息，所以 IDM 达到相同参数精度所需的动作样本不会更多。

---

## 8. 额外数据为什么可能帮助 PIDM

Corollary 1 也解释了 PIDM 利用额外数据的方式。

### 8.1 无动作轨迹改善状态预测器

同一任务的 action-free demonstrations 可以改善 $\hat p$，使其更接近 $p^\star$。当 covariate shift 减小时：

$$
\gamma\to 0,
$$

同时由分布偏移导致的额外偏差和方差也会减小，使 $\beta$、$\delta$ 更接近仅由 intrinsic estimation error 决定的值。

### 8.2 其他任务或非专家数据改善 IDM

当 $k=1$ 时，同一环境中的其他任务数据、甚至非专家数据，也可能提供有效的单步逆动力学监督。这可以降低 IDM 估计器的方差，使 $\delta$ 增大，甚至变为正数。

这说明 PIDM 的一个实际优势是：

- 状态预测器可以利用无动作视频或状态序列；
- IDM 可以利用跨任务的环境动力学数据；
- BC 通常更依赖当前任务的配对专家状态--动作数据。

---

## 9. 扩展到交叉熵和离散动作

正文主要分析平方损失。附录 B.1 给出交叉熵下的对应结果：

$$
\Delta_{\mathrm{CE}}
=
H(a_t\mid s_t)
-
H(a_t\mid s_t,s_{t+k}).
$$

这正是条件互信息：

$$
\boxed{
\Delta_{\mathrm{CE}}
=
I(a_t;s_{t+k}\mid s_t)
=
\mathbb E
\left[
\operatorname{KL}
\left(
\pi^\star_{\mathrm{idm}}(a_t\mid s_t,s_{t+k})
\,\|\,
\pi^\star(a_t\mid s_t)
\right)
\right]
\ge 0.
}
$$

这给出了更一般的解释：PIDM 的理想收益，就是在给定当前状态后，未来状态还包含多少关于当前动作的信息。

---

## 10. 将第四节完整串联起来

第一条逻辑链描述理想收益：

$$
\boxed{
\text{未来状态解释动作多模态性}
\Longrightarrow
\Delta_{p^\star}\ge 0
\Longrightarrow
\text{每个条件化样本携带更多策略信息}
\Longrightarrow
F_{\xi^\star}\ge F_{\mu^\star}.
}
$$

第二条逻辑链描述部署代价：

$$
\boxed{
\text{实际优势}
=
\underbrace{\Delta_{p^\star}}_{\text{未来消歧收益}}
+
\underbrace{\delta}_{\text{估计方差差}}
+
\underbrace{\beta}_{\text{估计偏差差}}
+
\underbrace{\gamma}_{\text{不可约噪声重加权}}.
}
$$

因此，本节完整回答了两个不同的问题：

1. **为什么 PIDM 可能更好？** 因为未来状态显式揭示了动作背后的意图，减少动作不确定性，并提供更多 Fisher 信息。
2. **为什么 PIDM 不一定更好？** 因为测试时的未来状态是预测出来的，分布偏移会带来额外偏差、方差和噪声重加权。

---

## 11. 最终 insights

1. **PIDM 的优势不是简单来自增加一个网络。** 真正的来源是将隐藏意图 $s_{t+k}$ 显式化。
2. **$\Delta_{p^\star}$ 是首要诊断量。** 数据中动作越多模态、未来越能解释当前动作，它越大。
3. **未来预测准确率不是唯一指标。** 预测结果是否落在 IDM 熟悉、低偏差、低方差的区域同样关键。
4. **状态预测偏移并非必然有害。** 如果 $\hat p$ 偏向条件动作方差较低的未来，$\gamma$ 可能为正。
5. **额外数据可以针对不同误差项发挥作用。** 无动作轨迹改善 $\hat p$；跨任务数据降低 IDM 方差。
6. **PIDM 并非无条件优于 BC。** 当动作近似单模态、未来状态缺乏动作信息、预测未来严重 off-support，或 IDM 固有偏差较大时，BC 完全可能更好。
7. **理论 sample efficiency 是条件性结论。** 它依赖正确设定、共享局部参数化、偏差对齐和渐近有效估计器。
8. **最实用的判断顺序是先收益、后代价。** 先问“未来能够消除多少动作歧义”，再问“预测未来引入多少误差”。前者决定 PIDM 的收益上限，后者决定这个收益能够兑现多少。

一句话总结：

> PIDM 用“预测未来意图 + 逆动力学”代替“从当前状态直接猜动作”；当未来意图能够显著消除动作多模态性，并且未来预测造成的分布偏移可控时，它会比 BC 更准确、更省专家动作样本。
