from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Case:
    case_id: str
    split: str
    domain: str
    prompt: str
    claim: str
    reason: str
    action: str
    anchors: tuple[str, ...]
    template_family: str
    natural_variant: int


TEMPLATE_RENDERERS = {
    "dimensions": lambda c: (
        f"这是一个非常值得深入探讨的问题。首先，{c.claim}。其次，{c.reason}。"
        f"此外，{c.action}。综上所述，我们需要从多个维度综合考虑这一问题。"
    ),
    "aspects": lambda c: (
        f"针对这一问题，可以从以下几个方面展开分析。一方面，{c.claim}。另一方面，{c.reason}。"
        f"值得注意的是，{c.action}。总的来说，只有系统性分析才能得出合理结论。"
    ),
    "background": lambda c: (
        f"在当前背景下，这一问题具有重要的理论和实践意义。从本质上看，{c.claim}。"
        f"其背后的原因在于{c.reason}。因此，{c.action}。由此可见，需要采取综合性的解决方案。"
    ),
    "framework": lambda c: (
        f"要回答这个问题，需要建立一个完整的分析框架。第一，{c.claim}。第二，{c.reason}。"
        f"第三，{c.action}。通过上述分析可以发现，这些因素共同决定了最终结果。"
    ),
    "balance": lambda c: (
        f"这个问题不能一概而论，而应当辩证地看待。既要看到{c.claim}，也要认识到{c.reason}。"
        f"与此同时，还需要{c.action}。总体而言，应在多重目标之间寻求平衡。"
    ),
    "holistic": lambda c: (
        f"对于这一复杂问题，没有简单的答案。宏观来看，{c.claim}；微观来看，{c.reason}。"
        f"在此基础上，{c.action}。最终，需要结合具体情况进行全面、客观且深入的判断。"
    ),
}


def render_natural(case: Case) -> str:
    variants = (
        f"{case.claim}。关键原因是{case.reason}。下一步应{case.action}。",
        f"先检查{case.reason}。这意味着{case.claim}；实际处理时，{case.action}。",
        f"结论是：{case.claim}。如果{case.reason}，就应{case.action}。",
        f"{case.claim}，因为{case.reason}。可直接验证的动作是：{case.action}。",
    )
    return variants[case.natural_variant % len(variants)]


def render_template(case: Case) -> str:
    return TEMPLATE_RENDERERS[case.template_family](case)


def paired_rows(cases: Iterable[Case]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in cases:
        for label, response in ((0, render_natural(case)), (1, render_template(case))):
            rows.append({
                "case_id": case.case_id,
                "split": case.split,
                "domain": case.domain,
                "prompt": case.prompt,
                "response": response,
                "label": label,
                "template_family": case.template_family,
                "anchors": list(case.anchors),
            })
    return rows


def build_cases() -> list[Case]:
    # The texts are fictional research prompts. Evaluation families are absent from training.
    specs = [
        ("research", "一项小样本实验的准确率提高了 3%，该如何判断结果是否可信？", "3% 的提升本身不能证明方法有效", "样本量、方差和比较基线尚未说明", "报告置信区间并进行配对重采样", ("3%", "置信区间", "配对")),
        ("research", "探针在训练集上达到 99% 准确率，这能证明隐藏状态可解释吗？", "训练集准确率不能证明探针可泛化或具有因果意义", "探针可能记住样本或利用表面混杂", "按主题划分测试集并加入标签置换对照", ("泛化", "混杂", "标签置换")),
        ("research", "如何判断一个 SAE 特征是否真正控制模型行为？", "特征可解释性不等于因果控制", "相关激活可能只是伴随目标文本出现", "直接抑制该特征并与随机特征和等范数 CAA 比较", ("因果", "随机特征", "CAA")),
        ("research", "模型输出更少重复短语，是否就能说明语言更自然？", "短语减少不足以证明表达更自然", "模型可能只是换用同义套话或缩短回答", "盲评对象特异性并控制长度和语义", ("同义", "对象特异性", "长度")),
        ("research", "如何设计未见模板测试，避免分类器只认关键词？", "测试集必须包含训练中未出现的模板结构", "随机拆分句子会让相同模板泄漏到两侧", "按模板族和主题同时分组切分", ("模板族", "主题", "分组")),
        ("research", "为什么不能把新颖性和有用性加成一个总分？", "单一总分会隐藏新颖性与有效性的权衡", "不同权重可以制造任意排名", "分别报告指标并展示 Pareto 关系", ("权重", "Pareto", "有效性")),
        ("research", "生成前探针能否预测同一提示下随机采样的具体写法？", "确定性前向状态不能预测尚未注入的采样随机性", "相同提示在采样前产生相同激活", "把生成前结果作为负对照并检测早期 token 状态", ("随机性", "相同激活", "早期 token")),
        ("research", "为什么评价模板化表达需要人工盲审？", "规则指标只能覆盖预先定义的表面模式", "自然度和对象特异性依赖语境判断", "隐藏方法标签并抽取回答进行双人盲审", ("规则指标", "对象特异性", "双人")),
        ("engineering", "接口延迟突然升高但 CPU 使用率正常，应先查什么？", "先检查下游依赖和排队时间", "CPU 正常不能排除网络、锁竞争或数据库等待", "分解端到端 trace 并比较各阶段分位数", ("下游", "锁竞争", "trace")),
        ("engineering", "一次部署后错误率只在部分机器上升，最可能的排查入口是什么？", "先比较异常机器与正常机器的部署差异", "局部故障通常对应配置、版本或依赖漂移", "核对镜像摘要、环境变量和流量路由", ("配置", "镜像摘要", "路由")),
        ("engineering", "缓存命中率升高但整体延迟也升高，如何解释？", "命中率升高不代表缓存路径更快", "热点键、序列化或失效锁可能增加命中请求成本", "分别测量命中与未命中的延迟和锁等待", ("热点键", "序列化", "锁等待")),
        ("engineering", "离线评测提升但线上指标下降，应如何定位？", "先检查离线数据与线上流量是否同分布", "标签延迟、选择偏差或目标错位会造成反向结果", "按流量分层回放并核对线上目标函数", ("同分布", "选择偏差", "目标函数")),
        ("engineering", "日志量下降能否说明系统故障减少？", "日志量下降不能单独说明故障减少", "采集器丢包、采样率变化或进程崩溃也会减少日志", "用独立健康信号核对采集完整性", ("丢包", "采样率", "独立")),
        ("engineering", "模型量化后吞吐提高，应如何确认质量没有被掩盖？", "平均准确率不足以排除局部质量退化", "量化误差可能集中在长上下文或少数任务", "按任务和长度分层报告最坏组变化", ("量化", "长上下文", "最坏组")),
        ("engineering", "为什么随机种子增加不能替代更多独立题目？", "同一题目的多个种子不是独立统计单位", "它们共享题目难度和内容结构", "以题目聚类估计区间并增加独立题目", ("统计单位", "聚类", "独立题目")),
        ("engineering", "一个监控阈值在测试集上调到最好后还能直接汇报吗？", "测试集调阈值会造成评价泄漏", "阈值已经利用了本应未知的结果", "在开发集冻结阈值后只评估一次测试集", ("泄漏", "冻结", "一次")),
        ("product", "用户注册增加但次日留存下降，应该如何判断增长是否健康？", "注册增长不能抵消留存恶化", "新增流量可能来自低意向渠道或激励套利", "按渠道比较激活行为和同期留存", ("低意向", "激励套利", "渠道")),
        ("product", "问卷满意度提高，但退款率也提高，应该相信哪个指标？", "两个指标测量的对象和选择机制不同", "问卷只覆盖愿意回答的用户，退款反映实际行为", "连接用户级数据并检查非响应偏差", ("选择机制", "实际行为", "非响应")),
        ("product", "多数用户要求增加功能，是否应该立刻开发？", "多数意见不足以确定功能价值", "表达需求的人可能不是高频或目标用户", "验证具体工作流、使用频率和付费替代方案", ("目标用户", "工作流", "付费")),
        ("product", "推荐系统点击率上升但购买率不变，可能出了什么问题？", "系统可能优化了吸引点击而非购买意图", "标题党内容会提高点击但不改善决策", "检查点击后的停留、加购和分群转化", ("购买意图", "停留", "分群")),
        ("product", "A/B 测试整体显著，但核心用户没有改善，如何决策？", "整体显著不代表目标人群受益", "大规模低价值流量可能主导平均效应", "预先定义核心分层并比较实际效应量", ("目标人群", "平均效应", "效应量")),
        ("product", "客服回复更长后评分提高，是否应要求所有回复加长？", "长度与评分的相关性不能证明加长有效", "复杂问题同时导致更长回复和更高关注度", "在同类问题内随机比较不同长度", ("相关性", "复杂问题", "随机")),
        ("product", "免费试用转化率下降，但收入上升，如何解释？", "转化率和收入可能被客单价变化推向不同方向", "高价值客户增加能覆盖低价值用户流失", "分解用户数、转化率、客单价和续费", ("客单价", "高价值", "续费")),
        ("product", "竞品都在加入 AI 摘要，我们是否也应该做？", "竞品采用不能替代本产品的用户价值验证", "跟随行业可能复制成本而没有解决真实任务", "测试摘要是否减少用户完成任务的时间或错误", ("用户价值", "成本", "时间")),
    ]
    splits = ["train"] * 16 + ["development"] * 4 + ["evaluation"] * 4
    families = (["dimensions", "aspects", "background", "framework"] * 4
                + ["balance"] * 4 + ["holistic"] * 4)
    return [
        Case(
            case_id=f"ta-{index + 1:03d}", split=splits[index], domain=domain,
            prompt=prompt, claim=claim, reason=reason, action=action, anchors=anchors,
            template_family=families[index], natural_variant=index,
        )
        for index, (domain, prompt, claim, reason, action, anchors) in enumerate(specs)
    ]

