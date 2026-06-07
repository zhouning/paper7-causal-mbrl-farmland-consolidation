这是一份以《Nature Machine Intelligence》或《Nature》正刊等全球顶级学术期刊评审专家（Reviewer）视角撰写的同行评审报告。该评审以**极其严谨、客观且富有建设性**的基调进行，旨在帮助你们在投稿前拔高论文的学术深度与严谨性。

***

# Review Report

**Manuscript Title:** Dreaming in Embedding Space: Model-Based Reinforcement Learning for Farmland Consolidation via Learned Environment Dynamics
**Target Journal:** Nature Machine Intelligence / NeurIPS

## 总体评价 (General Assessment)

本研究针对空间规划（农田平整）中强化学习环境模拟成本极高、容易发生奖励利用（Reward Exploitation）和策略漂移的痛点，提出了一种基于模型的强化学习（MBRL）框架。作者通过构建轻量级神经状态转移模型替代真实环境，并创新性地引入了“因果奖励校准（Causal Reward Calibration）”机制。

**本文具有极高的应用价值和显著的方法论创新。** 将计算成本从 A100 GPU 上的 8-12 小时骤降至普通 CPU 上的 2 小时以内（提速近一个数量级），同时还提升了最终的规划质量（坡度降低率提升 17.8%），这对于将 DRL 真正推向大规模现实世界空间规划具有里程碑式的意义。此外，发现“引入高维基础模型特征（GeoFM）虽能提升转移模型精度，但会降低最终策略质量”的客观规律，为 MBRL 领域的“目标不匹配（Objective Mismatch）”问题提供了非常生动的真实世界实证。

然而，要达到《Nature Machine Intelligence》等顶刊的发表标准，本文在**统计严谨性、因果校准的基线对比、标题概念的准确性以及泛化能力验证**上仍存在几个必须解决的核心问题。以下是详细的肯定与改进建议。

---

## 主要优势与创新点 (Major Strengths and Innovations)

1. **突破计算瓶颈的工程与学术双重贡献**：在真实、超大规模（13个乡镇，5.2万个地块）的空间规划任务中成功落地 MBRL，摆脱了对高端 GPU 的依赖，这种计算效率的量级提升对可持续人工智能（Green AI）和资源受限的规划部门极具吸引力。
2. **因果奖励校准（Causal Reward Calibration）的巧妙引入**：利用倾向得分匹配（PSM）来估计行动质量对奖励的真实平均处理效应（ATT），以此修正神经网络对奖励差异的系统性高估（5.4倍）。这是一种将因果推断与强化学习结合以解决“奖励骇客（Reward Hacking）”问题的极具启发性的方法。
3. **反直觉但深刻的实证发现（GeoFM 实验）**：作者诚实地报告了更精确的环境动态模型（加入 GeoFM embedding 后）反而导致策略退化的现象，并给出了合理的理论解释（静态特征污染与“有益的过拟合”）。这提升了科学讨论的深度。

---

## 核心问题与重大改进建议 (Critical Concerns and Major Revisions)

为确保能够在顶级期刊成功发表，建议作者在修改稿中务必解决以下几个致命弱点：

### 1. 统计显著性严重不足 (Statistical Significance and Seed Count)
* **评审意见**：对于一篇目标为顶刊的文章，实验部分仅使用 $n=3$ 个随机种子（Seeds）来证明策略的优越性是**绝对不可接受的**。在表1和表3中，由于样本量过小，作者甚至无法报告 $p$-value。$n=3$ 极易受到偶然性（Lucky seeds）的影响。
* **修改建议**：考虑到本文的核心贡献之一是“在 CPU 上不到 2 小时即可完成训练”，增加随机种子数量在计算上是完全可行的。**必须将主要实验和消融实验的种子数增加至至少 10 个（推荐 15-20 个）**，并报告严格的统计检验结果（如 Welch's t-test 或 Mann-Whitney U test），以证明校准后的 MBRL 显著优于无校准版和无模型（Model-free）基线。

### 2. 因果校准机制的论证存在漏洞 (Mechanistic Justification of Causal Calibration)
* **评审意见**：作者通过因果推断计算出一个全局标量 $\alpha = 0.185$，并将其乘在所有预测奖励上。审稿人会立刻提出一个极其尖锐的问题：**这与我通过网格搜索（Grid Search）或者启发式手动设置一个奖励缩放因子（Reward Scaling Factor, $\lambda \in [0.1, 1.0]$）有何区别？** 如果简单地将奖励缩小 5 倍也能达到相同的效果，那么“因果推断”的必要性就会被削弱。
* **修改建议**：必须在消融实验中增加一组**“启发式奖励缩放（Heuristic Reward Scaling）”基线**（例如尝试 $\alpha = 0.1, 0.2, 0.5, 1.0$）。你需要证明：通过因果推断**直接计算出的 $\alpha=0.185$ 恰好等于（或优于）昂贵的超参数搜索所能找到的最优值**。这将极大地增强该方法的说服力——即因果推断提供了一种“免调参（Tuning-free）”的、具有物理意义的校准手段。
* **进阶建议**：目前的 $\alpha$ 是一个全局常数。如果能探讨基于状态依赖（State-dependent）的动态校准因子 $\alpha(s)$，将使该机制更具顶刊级别的技术深度。

### 3. 标题与核心方法的概念错位 (Misalignment between Title and Core Concept)
* **评审意见**：论文标题为 “Dreaming in Embedding Space”，这会让读者立刻联想到 DreamerV3 等在隐空间（Latent Space / Embedding Space）学习环境动态的方法。然而，作者在第 2.1 节明确指出：“*rather than jointly learning a latent state representation... we operate directly in the environment's native observation space*”。这说明转移模型是直接在**原始观测空间（Observation Space）**进行的预测（尽管对输入进行了 encode），并未在降维的连续隐空间中进行“梦境”生成。
* **修改建议**：目前的标题具有一定的误导性，可能会激怒极其熟悉 Dreamer 架构的审稿人（他们会认为你在蹭热度）。建议将标题修改为更准确的形式，例如：
  * *Dreaming in Observation Space: Model-Based Reinforcement Learning for Farmland Consolidation*
  * *Causally Calibrated Environment Dynamics for Efficient Spatial Planning*

### 4. 分布偏移与泛化能力的缺失 (Distribution Shift and Out-of-Distribution Generalization)
* **评审意见**：转移模型是由“随机策略”和“贪心策略”收集的数据（12,000 条轨迹）训练的。在 PPO 训练后期，智能体大概率会探索到这两种策略未曾覆盖的状态空间（Out-of-Distribution, OOD）。文章目前缺乏对模型在这种分布偏移下表现的探讨。
* **修改建议**：
  1. 在 Section 5.3 中补充分析：在 PPO 训练的最后 10% step 中，转移模型对当前状态的预测误差（MSE）是否显著上升？
  2. **（极具杀伤力的加分项）** 零样本泛化测试（Zero-shot Generalization）：如果将在璧山区（Bishan District）训练的转移模型，直接用于另一个完全未见过的区县（或另外几个乡镇）进行策略训练，效果如何？如果能证明该转移模型具备空间泛化能力，这将是冲击《Nature Machine Intelligence》的重磅武器。

### 5. 空间溢出效应与独立性假设 (Spatial Spillover Effects in the Transition Model)
* **评审意见**：在转移模型的架构中，作者预测了被选地块的残差 $\Delta x_{a_t}$，并强制假设其他地块不变（$\hat{x}_{b, t+1} = x_{b, t}$ for $b \neq a_t$）。在空间规划中，地块之间的连通性（Contiguity）和百亩方（Baimu fang）通常涉及多个相邻地块的拓扑变化。强制其余地块状态不变的假设是否过于强烈？它是否破坏了空间连通性的真实动态？
* **修改建议**：在论文中必须明确讨论这一假设（Independence Assumption）的合理性。如果环境底层引擎每次 Action 只真正改变选定 block 内的 parcel 属性，那么这个假设是精确的；如果 Action 会改变 block 之间的边界或连通性指标，那么忽略溢出效应（Spillover）会带来怎样的误差？应在 Discussion 中予以充分回应。

---

## 次要问题与写作建议 (Minor Points and Presentation)

1. **图表呈现**：在表格 1 (Table 1) 中，目前缺少 $p$-value。请在扩大种子数量后，增加一列统计检验的显著性标记（如 $*, **, ***$）。
2. **术语一致性**：在摘要中提到 Transition model 拥有 237K 参数，而在 Section 4.2 结尾处写的是 236,958，请在正文中统一表述或说明“约237K”。
3. **因果推断细节补充**：在 Section 4.4 中，使用梯度提升树（GBDT）计算倾向得分（Propensity score）。建议在附录（Appendix）中补充匹配前后的协变量平衡性检验图（Covariate Balance Plot / Love Plot），这是发表因果推断相关工作的标准流程，能极大增强评审专家对你 ATT 估计准确性的信任。
4. **文献引用**：建议补充 MBRL 在离散/组合优化空间（Discrete/Combinatorial Action Spaces）中应用的最新相关文献，目前引用的 MuZero 是经典，但在组合优化（如 TSP, VRP 或 Spatial Planning）中利用 MBRL 的最新讨论略显不足。

---

## 总结论 (Conclusion)

**Verdict: Revise and Resubmit (Major Revision)**

这篇论文触及了当前强化学习在空间规划中应用的最痛点（算力成本高、容易过拟合奖励模型），其提出的 “轻量级转移模型 + 因果奖励校准” 框架不仅思路新颖，且在真实的庞大系统上展现出了卓越的性能。

只要作者能够：
1. **老老实实将随机种子（Seeds）增加到 10个以上以确立统计显著性**；
2. **通过与网格搜索的对比，证明因果校准不仅“有效”且“免调参”**；
3. **修正具有误导性的标题，并补充倾向得分匹配的严谨性图表**；

本文将是一篇非常扎实、具有极高影响力的顶刊佳作。预祝修改顺利，期待在顶刊上看到这项成果！