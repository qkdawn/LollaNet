# LollaNet Workbench

Next.js + TypeScript 前端和 Go Chi 只读 API。模型来自公开 Markdown 的 legacy 快照，Behavior Mode 是独立实验目录，当前没有正式模型与曲线关联。

## 启动

需要 Node.js 22+ 和 Go 1.24+。从仓库根目录打开两个终端：

```powershell
cd frontend
npm install
npm run snapshot
npm run dev
```

```powershell
cd backend
go mod download
go run .
```

前端：http://127.0.0.1:3000；API：http://127.0.0.1:8080/api/meta。

环境变量：前端 `PORT`（默认 3000）、`API_ORIGIN`（默认 http://127.0.0.1:8080）；后端 `API_ADDR`（默认 127.0.0.1:8080）、`LIBRARY_PATH`（默认 data/library.json，相对于后端工作目录）。端口被占用时调整两个服务的配置。

`npm run snapshot` 从公开的 `01-Models/` 读取模型，并接入 `content/model-readings/` 中独立维护的试读文章，输出 `backend/data/library.json`。快照保留 ID、版本、完整二级章节、来源链接和结构化正式关系；不把 related/conflicts 自动转换为关系，不读取私人目录。曲线分类在 `lib/modes.ts`，公式受控实现位于 `lib/curves.ts`。曲线来源状态为待核验，图像为无量纲示意，不是实际观测；文章来源的核验范围逐条标注。

## API

- `GET /api/meta`：来源、生成时间、模型及曲线计数。
- `GET /api/categories`：三级分类树与条目数。
- `GET /api/models?q=&category=&status=&kind=`：中文/英文/定义子串搜索和分类、状态、类型过滤。
- `GET /api/models/{id}`：完整模型详情；未知 ID 返回 404。
- `GET /api/modes`、`GET /api/modes/{id}`：Behavior Mode 分类与变体。
- `GET /api/associations`：独立多对多关联，初始为空。

后端 repository 接口隔离 JSON 存储；后续接 PostgreSQL 不改变上述接口。当前不会运行数据库迁移或写入正式模型。

## 模型试读文章

原创讲解位于 `content/model-readings/`，已覆盖当前公开快照的全部 88 个模型，包括数学、物理、化学、工程和商业判断。文章从具体例子展开，再解释机制与容易误用的地方。批次与模型 ID 记录在 `content/reading-batches.json`，编辑复核记录见 `content/EDITORIAL_REVIEW.md`。每篇使用 YAML 记录 modelId、modelVersion、独立稿件 revision、title、summary 和可定位的 sources；正文与技术附录由 `<!-- technical-appendix -->` 分隔。正文引用使用 `#source-<id>`，跳至文末来源。

快照脚本将文章加入对应模型的可选 `reading` 字段。版本不匹配、重复绑定、丢失附录或引用不存在的来源时，生成失败。原始 definition、sections、sources 和正式关系完整保留；当前 88 个模型都有 reading 字段，未来没有阅读稿的条目仍可展示原始资料。Go API 校验绑定版本，搜索包含新版摘要和标题。单元测试检查已发布批次完整性及正文 1200–2000 中文字，浏览器测试逐篇覆盖新版、原始资料，并用缺少阅读稿的响应验证回退。

有阅读稿的模型默认显示新版文章，列表使用新版摘要，原始资料通过页签切换。公式使用 remark-math 与 KaTeX，关闭 trusted TeX 命令且不解析原始 HTML。阅读稿尚不代表正式模型变更；更新文章后需重新生成快照并重启 Go 服务。

## 动力学实验

Behavior Mode 分为“时间曲线”和“动力学实验”两个页签。时间曲线保留七类、十六个变体；动力学实验包含平衡与稳定性、极限环与自激振荡、多稳态与吸引域、分岔与临界转变、滞回与路径依赖、延迟动力学、受迫响应与共振、耦合与同步八个专题。

实验目录及受控方程位于 `lib/dynamics.ts`，轨迹在浏览器内计算，独立于 Go 模型快照及正式模型关联。各实验提供初始条件、参数、预设、重置、时间曲线与适合该系统的相图、势能图、吸引域、分岔图或频率响应图。小球实验支持 t=4 时施加速度扰动；分岔图与频率响应图可点击调参数，吸引域可点击选起点。参数与扰动状态保存在 URL 的 `settings`、`kick` 中，刷新、分享和预设历史可恢复实验。

常微分方程使用固定步长 RK4；Hopf 半径采用解析解；延迟方程采用历史插值与梯形步进。吸引域以 21×17 网格、35 时间单位估计，尚未稳定或停在势垒处的格子显示灰色。图中明确区分理论长期分支、稳态响应及有限时间轨迹。参考来源逐项标注已读取范围或待核验线索。所有坐标均为无量纲教学示意，没有新增正式模型与曲线关系。

直接入口：`http://127.0.0.1:3000/?view=curves&atlas=dynamics`。

本次验证：TypeScript 检查、生产构建、18 项前端单元测试与 Go 测试通过。Playwright 的 6 项动力学检查和 4 项原有工作台检查通过，覆盖桌面/移动端、全部八个专题、参数与重置、公式、扰动、图上选点、URL 恢复和跨页导航；已人工核对目录、多稳态与共振的桌面/移动端截图。数值测试包含参数边界、解析解对照、延迟步长细化及锁相条件；这些验证适用于所实现的教学方程，不等同于真实系统验证。

## 验证

```powershell
npm run snapshot
npm test
npm run typecheck
npm run build
npx playwright install chromium
npm run test:e2e
```

浏览器测试需要前后端服务已启动；默认使用本机 Chrome，`BROWSER_CHANNEL=chromium` 可改为下载的 Chromium，自定义地址使用 `APP_URL`。后端执行 `go test ./...`。生产前端执行 `npm run build` 后 `npm start`。
