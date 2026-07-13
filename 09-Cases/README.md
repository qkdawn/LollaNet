---
type: case_library
title: Public Case Library
updated: 2026-07-12
---

# Public Case Library

`09-Cases/` 保存基于公开可核验来源整理的真实案例。它不是私人项目证据区，也不是全文资料库。

## 收录契约

每个案例放在 `09-Cases/<domain>/`，并至少包含：

```yaml
---
type: case
case_name: <案例标题>
domain: <domain>
created: <YYYY-MM-DD>
updated: <YYYY-MM-DD>
date_observed: <YYYY-MM-DD or unknown>
as_of: <YYYY-MM-DD>
privacy: public
provenance: public-sourced
verification_status: sourced
public_case: true
source_refs:
  - <可核验 URL、出版物或精确文献定位>
related_models: []
---
```

- `source_refs` 不得为空，且应定位到案例事实所依据的底层来源。
- `date_observed` 表示事件发生或被观察的日期；未知时明确写 `unknown`，不得用文件创建日冒充。
- `as_of` 表示资料最后核验到的日期；创建当天完成核验时可以与 `created` 相同。
- 正文必须分别记录“事实”“推断”“时代条件”；来源作者的解释与整理者的推断也要区分。
- 时代条件至少检查与结论相关的制度、技术、激励、参与者和测量口径变化。
- 引用、定位并概括必要内容，不复制受版权保护的全文；遵守来源条款和适用许可。
- 公开可见不等于个人数据可无条件收录。只保存判断所需的最小信息，并确认目的相关与合法性；敏感或无关个人细节不收录。

## 不属于这里的内容

- 包含用户个人信息、未公开项目或公司、私人对话、内部文件或专有数据的案例，写入 `private/projects/<project_id>/evidence/cases/`。
- 纯虚构示例写入 `08-Examples/cases/`，并标记 `example: true` 与 `fictional: true`。
- 只有表面相似、无法回引来源或不能区分事实与推断的材料，不作为公共真实案例。

## 创建方式

三个案例模式互斥：

```bash
python 06-Scripts/create_case.py "<公开案例标题>" --public-case --source "<可核验来源>"
python 06-Scripts/create_case.py "<私人案例标题>" --project <project_id>
python 06-Scripts/create_case.py "<虚构示例标题>" --public-example
```

同一公共案例需要多个来源时，重复传入 `--source`。
使用 `--observed-on <YYYY-MM-DD>` 记录事件日期，未知时保持 `unknown`；使用 `--as-of <YYYY-MM-DD>` 记录资料核验截止日。
