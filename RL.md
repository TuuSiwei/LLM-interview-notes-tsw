<h1 id="FcD0v">策略梯度</h1>
目标：最大化所有 trajectory 的 return 的期望。

1. trajectory 的概率，主要取决于`1.` policy 对 action 的选择`2.`状态转移

$ p_\theta(\tau) = p(s_1)p_\theta(a_1|s_1)p(s_2|s_1, a_1)p_\theta(a_2|s_2)p(s_3|s_2, a_2) \cdots= p(s_1) \prod_{t=1}^{T} p_\theta(a_t|s_t) p(s_{t+1}|s_t, a_t)
 $

2. `return的期望`=`每条trajectory的return` * `每条trajectory的概率`

$ \bar{R}_\theta = \sum_\tau R(\tau)p_\theta(\tau) $

3. 由于需要最大化 return 的期望，因此可以使用梯度上升来更新 policy。推导梯度：

$ \nabla \bar{R}_\theta = \sum_\tau R(\tau) \nabla p_\theta(\tau) \\
= \sum_\tau R(\tau) p_\theta(\tau) \frac{\nabla p_\theta(\tau)}{p_\theta(\tau)} \\
= \sum_\tau R(\tau) p_\theta(\tau) \nabla \log p_\theta(\tau) $

前两项可以视为期望，改写为抽样 N 条 trajectory

$ \nabla \bar{R}_\theta = \sum_\tau R(\tau) \nabla p_\theta(\tau) \\
= \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} R(\tau^n) \nabla \log p_\theta(a_t^n | s_t^n) $

如何计算 log 项？连乘取 log 后变为连加，最终只与包含 policy 的一项有关

$ p_\theta(\tau) = p(s_1)p_\theta(a_1|s_1)p(s_2|s_1, a_1)p_\theta(a_2|s_2)p(s_3|s_2, a_2) \cdots= p(s_1) \prod_{t=1}^{T} p_\theta(a_t|s_t) p(s_{t+1}|s_t, a_t)
 $

$ \nabla \log p_\theta(\tau) = \nabla \left( \log p(s_1) + \sum_{t=1}^{T} \log p_\theta(a_t|s_t) + \sum_{t=1}^{T} \log p(s_{t+1}|s_t, a_t) \right) \\
= \nabla \log p(s_1) + \nabla \sum_{t=1}^{T} \log p_\theta(a_t|s_t) + \nabla \sum_{t=1}^{T} \log p(s_{t+1}|s_t, a_t) \\
= \nabla \sum_{t=1}^{T} \log p_\theta(a_t|s_t) \\
= \sum_{t=1}^{T} \nabla \log p_\theta(a_t|s_t) $

最终结果

$ \nabla \bar{R}_\theta = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} R(\tau^n) \nabla \log p_\theta(a_t^n | s_t^n) $

4. 定义 loss

$ loss = -\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} R(\tau^n) \log p_\theta(a_t^n | s_t^n) $

训练时：$ \theta_{new}=\theta_{old}-lr*(loss对\theta的梯度)=\theta_{old}+lr*\bigtriangledown\bar{R} _{\theta} $

直观理解：如果一个 trajectory 的 return 为正，就强化这个 trajectory 中每个 state 采取对应 action 的概率

5. on policy

策略梯度收集多个 trajectory，更新 policy，循环往复。（收集数据的 policy 和待更新的 policy 是同一个）

缺点：采集数据时间较长，训练较慢。

<h1 id="jGQie">PPO</h1>
1. 改进 return

`Q1.`当前 state 做出的 action 应该只影响以后的 reward(而不是整条 trajectory 的 return)，同时影响应该逐渐衰减。

`修改方案`将return修改为当前时间步到结束的reward之和，并引入逐渐衰减的权重

`Q2.`可能所有 action 的 reward 都是正值，这样选择的所有 action 都会被提升，训练效果较慢，即使所有 action 的 reward 都是正值，也需要区分出它们的相对性

`修改方案`为所有 action 的 reward 设置一个 baseline(见后文)

原先：

$ \nabla \bar{R}_\theta = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} R(\tau^n) \nabla \log p_\theta(a_t^n | s_t^n) $

修改后：

$ \nabla \bar{R}_\theta = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} (R_t^n - B(s_n^t))\nabla\log P_\theta(a_n^t|s_n^t) $，其中$ R_t^n=\sum_{t'=t}^{T_n} \gamma^{t'-t} r_{t'}^n  $

2. 价值函数与优势

`1`action value$ Q_{\theta}(s,a) $：直接使用神经网络估计出在 state s 下做出 action a 的 return 的期望

`2`state value$ V_{\theta}(s) $：使用神经网络估计在 state s 下的 return 的期望

`3`advantage$ A_{\theta}(s,a)=Q_{\theta}(s,a)-V_{\theta}(s) $：在 state s 下做出 action a 相比其他 action 的优势，可以体现出 action 间的相对性

因此将梯度替换为：

$ \nabla \bar{R}_\theta = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_{\theta}(s,a)\nabla\log P_\theta(a_n^t|s_n^t) $

3. 优势的推导

`1`根据 immediate reward 展开 action value，并更新 advantage

$ Q_\theta(s_t, a) = r_t + \gamma * V_\theta(s_{t+1}) $

$ A_\theta(s_t, a) = r_t + \gamma * V_\theta(s_{t+1}) - V_\theta(s_t) $

`2`再根据 immediate reward 将 t+1 时刻的 state value 递推一步

$ V_\theta(s_{t+1}) \approx r_{t+1} + \gamma * V_\theta(s_{t+2}) $

`3`因此，advantage 可以根据不同数量的 immediate reward 进行展开

$ A_\theta^1(s_t, a) = r_t + \gamma * V_\theta(s_{t+1}) - V_\theta(s_t) $

$ A_\theta^2(s_t, a) = r_t + \gamma * r_{t+1} + \gamma^2 * V_\theta(s_{t+2}) - V_\theta(s_t) $

$ 

A_\theta^3(s_t, a) = r_t + \gamma * r_{t+1} + \gamma^2 * r_{t+2} + \gamma^3V_\theta(s_{t+3}) - V_\theta(s_t) \\
\vdots  $

$ A_\theta^T(s_t, a) = r_t + \gamma * r_{t+1} + \gamma^2 * r_{t+2} + \gamma^3 * r_{t+3} + \cdot + \gamma^T * r_T - V_\theta(s_t) $

影响：采样步数越多，轨迹间的方差越大，偏差越小；采样越少，越多的 state value 由神经网络得到，轨迹间的方差越小，偏差越大。

`4` GAE 函数，对所有时间步的结果都进行采样，并分配权重

$ A_\theta^{GAE}(s_t, a) = (1 - \lambda)(A_\theta^1 + \lambda * A_\theta^2 + \lambda^2 A_\theta^3 + \cdots) $

因此将梯度替换为：

$ \nabla \bar{R}_\theta = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_\theta^{GAE}(s^t_n, a^t_n)\nabla\log P_\theta(a_n^t|s_n^t) $

4. 训练 state value 网络

与 policy 共享权重，policy 最后一层输出 action 的概率分布，而 state value 网络最后一层输出一个值作为 state value 即可，label 是 discounted return

5. on policy 与 off policy

on policy：使用 policy 采集数据，更新 policy (同一个 policy)

off policy：使用参考 policy 采集数据，来更新需要训练的 policy

6. 重要性采样

$ E(f(x))_{x \sim p(x)} = \sum_{x} f(x) * p(x) \\
= \sum_{x} f(x) * p(x) \frac{q(x)}{q(x)} \\
= \sum_{x} f(x) \frac{p(x)}{q(x)} * q(x) \\
= E\left(f(x) \frac{p(x)}{q(x)}\right)_{x \sim q(x)} \\
\approx \frac{1}{N} \sum_{n=1}^{N} f(x) \frac{p(x)}{q(x)}_{x \sim q(x)} $

要求`f(x)`在分布`p(x)`下的期望，可以转化为`f(x)p(x)/q(x)`在分布`q(x)`下的期望，并转化为采样 N 条记录的均值

---

可以将上述思想替换到策略梯度中：将原 policy$ 
P_\theta(a_n^t|s_n^t) $视为`p(x)`，其余项为`f(x)`，`q(x)`为参考 policy$ 
P_{\theta'}(a_n^t|s_n^t) $，则梯度替换为：

$ \nabla \bar{R}_\theta = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_\theta^{GAE}(s^t_n, a^t_n)\nabla\log P_\theta(a_n^t|s_n^t) \\
= \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_{\theta'}^{GAE}(s_n^t, a_n^t) \frac{P_\theta(a_n^t|s_n^t)}{P_{\theta'}(a_n^t|s_n^t)}\nabla\log P_\theta(a_n^t|s_n^t) $

(log 项是梯度项，不需要更换为参考）展开 log 项可得

$ = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_{\theta'}^{GAE}(s_n^t, a_n^t) \frac{P_\theta(a_n^t|s_n^t)}{P_{\theta'}(a_n^t|s_n^t)}\frac{\nabla P_\theta(a_n^t|s_n^t)}{P_\theta(a_n^t|s_n^t)} \\
= \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_{\theta'}^{GAE}(s_n^t, a_n^t) \frac{\nabla P_\theta(a_n^t|s_n^t)}{P_{\theta'}(a_n^t|s_n^t)} $

最后的 loss 表示为

$ loss=- \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_{\theta'}^{GAE}(s_n^t, a_n^t) \frac{\nabla P_\theta(a_n^t|s_n^t)}{P_{\theta'}(a_n^t|s_n^t)} $

7. 两个分布的差距不能过大

`1`kl 散度约束

$ loss_1=- \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} A_{\theta'}^{GAE}(s_n^t, a_n^t) \frac{\nabla P_\theta(a_n^t|s_n^t)}{P_{\theta'}(a_n^t|s_n^t)}+\beta kl(P_{\theta'},P_{\theta}) $

`2`截断过大的更新

$ Loss_{2} = -\frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T_n} min(A_{\theta'}^{GAE}(S_n^t, a_n^t)\frac{P_{\theta}(a_n^t|S_n^t)}{P_{\theta'}(a_n^t|S_n^t)}, clip(\frac{P_{\theta}(a_n^t|S_n^t)}{P_{\theta'}(a_n^t|S_n^t)}, 1-\epsilon, 1+\epsilon)A_{\theta'}^{GAE}(S_n^t, a_n^t)) $

clip：比值在`1-ε`到`1+ε`之间，返回本身；小于返回`1-ε`；大于返回`1+ε`

<h1 id="KLTCT">GRPO</h1>
见[deepseek & qwen](https://www.yuque.com/u39172896/orbyov/msl7dzgymmsk2r1w)

<h1 id="lez1a">GSPO</h1>
见[deepseek & qwen](https://www.yuque.com/u39172896/orbyov/msl7dzgymmsk2r1w)

