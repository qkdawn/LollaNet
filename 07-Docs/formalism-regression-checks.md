---
type: evaluation_method
status: active
updated: 2026-07-16
---

# Formalism Regression Checks

[`06-Scripts/check_model_formalisms.py`](../06-Scripts/check_model_formalisms.py) 为所有 `formalism: true` 的模型卡运行一组**窄关系回归检查**。

```bash
python 06-Scripts/check_model_formalisms.py
python 06-Scripts/check_model_formalisms.py --json
```

## 它检查什么

每张形式化卡都必须有且仅有一个显式检查定义。检查同时验证：

1. 卡片中的一个有区分度的公式、算法或边界标记仍然存在；
2. 与它对应的最小数值、符号或逻辑关系仍可复算；
3. 检查结果绑定的是卡片写下的**窄形式关系**，而不是一个泛泛的主题标签。

例如，并联可靠度检查独立并联系统的关系式；Little 定律检查 `L=XW` 的数值账本但不把它错误地当作“自动定位瓶颈”的方法；量纲检查只排除单位不一致，不能推出因果机制。

## 它不检查什么

该脚本**不能**证明：

- 来源正文是否真的支持完整核心主张；
- 公式在所有前提外仍成立；
- 跨域转译、组织类比或现实因果机制成立；
- 某模型使回答、行动或长期结果改善；
- 独立评审、实验/观察复现或真实任务 30/90 天复盘已经发生。

因此它是卡片质量层中的可执行回归门，而不是模型治理或现实有效性的证据。正式状态、版本和 proposal 仍遵循既有治理流程。

## 维护规则

新增、删除或改为 `formalism: true` 的模型卡时，必须同步更新检查表；脚本会因漏定义、过期定义、卡片公式标记丢失或示例失败而失败。若改变的是完整机制或适用边界，应先走相应的模型 proposal，而不是只修改检查数据。
