# RL Fair Toolkit (可插拔 RL 公平性重加权工具)

这个仓库提供一个**可插拔的 Reinforcement Learning (RL) 工具**，用于在**不泄露任何原始数据/字段信息**的前提下，对训练数据做**组级别（group-level）重采样 / 重加权**，从而在 **最大化公平性** 的同时尽量 **最小化准确率/性能损失**。

> ✅ 本仓库 **不包含任何数据集**  
> ✅ 不硬编码任何 “敏感属性字段名”（例如 race、gender 等）  
> ✅ 你只需要在本地运行时提供列映射（features/label/protected cols），即可复用同一套 RL 逻辑

---

## 1) 这个工具做什么？

给定一个表格数据集（CSV / DataFrame）：

- **X**：特征列（feature_cols）
- **y**：标签列（label_col）
- **g**：受保护属性列（protected_cols；可多个，自动形成 intersectional groups）

工具会让一个 RL Agent 在训练循环中反复尝试不同的 **group sampling weights**：

- 每一步选择一个 action：对某个 group 的权重做 `+delta` 或 `-delta`
- 通过这些权重对训练集重采样（保持总样本量不变）
- 训练一个基础分类器（默认 Logistic Regression，可替换）
- 在验证集上评估：
  - performance（默认 AUC，也可用 accuracy）
  - disparity（默认 Demographic Parity diff，也可用 Equal Opportunity diff 或 ABROCA）
- reward：把 “disparity 变小” 作为主要目标，同时惩罚 “性能低于 baseline”

最终输出：**一组最优 group weights**，可用于你后续的正式训练/实验 pipeline。

---

## 2) 安装

```bash
pip install -r requirements.txt
# 或者
pip install -e .
```

---

## 3) 最快上手：合成数据 Demo

```bash
python examples/quickstart_synthetic.py
```

这个 demo 使用 `p0/p1` 作为受保护列（纯占位符），不会涉及任何现实世界敏感属性。

---


## 4) 指标说明

- `perf_metric`
  - `auc`：ROC-AUC（binary / multiclass ovo）
  - `acc`：Accuracy

- `disparity_metric`
  - `dp`：Demographic Parity diff（各组预测为正的比例差）
  - `eo`：Equal Opportunity diff（各组 TPR 差）
  - `abroca`：ABROCA-style disparity（仅适用于 **二元 protected attribute**；如果你的 groups>2 会自动回退到 EO）

---

## 5) 隐私与“不要泄露字段名”的建议

本工具本身不会打印或 hard-code 任何字段名/敏感属性名，但你**运行时**仍需在 `DatasetSpec` 里提供列名。若列名本身敏感：

- ✅ 推荐：把真实列名放在 `configs/local_*.yaml`，并在 `.gitignore` 里忽略
- ✅ 推荐：对外分享时用 `p0/p1/...` 这样的占位符示例
- ❌ 不要：把真实数据或包含敏感列名的配置直接提交到 GitHub

---

## 6) 目录结构

```
rl_fair_toolkit/
  rl_fair/                 # 核心库
    metrics/               # fairness/performance metrics
    models/                # sklearn model wrapper
    rl/                    # env + DQN agent + training loop
  examples/                # demo 与脚本
  tests/                   # 最小单测
```

---

## 7) 免责声明

- 这是一个**研究工具**，并不保证在所有数据/任务上都能改善公平性或保持性能。
- 公平性指标与“公平”的定义依赖场景与政策/伦理语境；请结合你的研究问题选择合适指标与阈值。

---



---

If you want, you can publish this as a standalone pip package after adding CI (pytest) and basic documentation. 
If you use this toolkit in your research or applications, please cite: Zhang, F., Xing, W., Li, C., & Jiang, Y. (2026). Fair AI in educational predictions: A multi-group fairness approach using reinforcement learning. The Internet and Higher Education, 101074.
