# test7-OCR

将未命中题库的数学、物理题图转换为可验证、可编辑、可重新排版的结构化题目，并输出宽度固定为 800 px、高度随内容增长的 PNG。

这已经不只是 Gold 制作工程。当前包含两个明确分离的入口：

- **实际 OCR 产品页**：`http://127.0.0.1:8765/ocr/`，上传或粘贴一张原题后，真实调用结构化 OCR、原图直裁、800 px 排版和安全门禁；通过后可下载 PNG，发现实际问题时可一键加入哈希绑定的回归队列；
- **数据与成品验收台**：`http://127.0.0.1:8765/review/`，用于 Gold、手写 Mask、自建数据以及批量候选成品的后台质检，不是日常 OCR 输入页。裸地址 `http://127.0.0.1:8765/` 默认进入实际 OCR 产品页；旧的 `/?ui=...` 链接仍兼容后台验收。

产品上传采用 SHA-256 内容寻址去重，输入和结果分别保存在 `artifacts/product_inputs/`、`artifacts/product_runs/live/`，与 Gold、训练集和 `artifacts/runs/` 基准测试完全隔离。每次只保留原图一份、最终 PNG、结构化 JSON、原图直裁资产和报告，不保存逐阶段截图。

## 当前唯一 OCR 金标来源

```text
D:\robot\Documents\ASK\datasets\gaokao_physics_300_v1_0_1
```

此前的 `testzip_math_physics_300_v1` 已停止作为正式 OCR Gold，只保留为历史和真实拍照/手写安全回归素材。当前主金标为 300 道高中物理题，来源是项目自有结构化事实和确定性渲染，因此 300/300 可用于来源派生的逐字 OCR 正式评分。

当前默认端到端候选为 `gaokao_physics_full300_v72_formula_line_arbitration_candidate`：

- 300/300 成功生成，0 运行失败；
- 规范文字、题干、选项、公式语义、视觉裁剪、来源完整性、800 px 宽度和延期 PNG 回读均为 100%；
- LaTeX 泄漏和静默关键错误均为 0；
- 延期回读后为 261 PASS、39 REVIEW；在线平均耗时 `3.397s`、P95 `3.888s`、最大 `4.868s`，300/300 单题低于 5 秒，正式机器评分门槛全部通过；
- 300/300 最终 PNG 独立回读通过，后台回读用时 `293.758s` 且不计入在线 SLA；
- 产品时延配置会把两次 Edge+KaTeX 尝试限制在当前请求剩余的端到端预算内。v72 中 300/300 使用正式 Edge+KaTeX、0 降级；`GKP-PHY-0258` 的一次真实重试最终为 `4.868s`；
- v67 统一了流水线与验证器对“公式区域”的判定，v70 把公式 token 共识和正文标点共识拆开。v72 进一步只核验独立 OCR 中真正含公式信号的行：必须选中了独立 OCR、主公式识别有至少 2 个明确损坏信号而独立结果为 0、至少有 2 个公式行、公式行最低置信度不低于 `0.95` 且均值不低于 `0.965`，并且没有未知 LaTeX 命令，才允许确认公式；整页普通文字的原阈值没有放宽。`GKP-PHY-0140/0153` 因此从 REVIEW 转为 PASS，`FORMULA_UNVERIFIED` 从 2 降为 0；没有使用 Gold 答案决定产品状态。相对 v70 行为回归为 0，300/300 规范文字及最终 PNG SHA-256 完全一致。

这些数字只证明自建合成物理分布，不能外推到真实斜拍、阴影、折痕或手写遮挡。最终 PNG 人工验收与 OCR Gold 分开保存；发布器仍要求当前成品具备哈希绑定的人工 PASS。

旧 Testzip 目前只承担非 Gold 的真实拍照/手写安全回归。最新全量运行 `realworld_testzip_full298_v60_visual_option_recovery_non_gold` 为 298/298 完成、237 REVIEW/61 REJECT、0 运行失败，在线平均 `2.723s`、P95 `3.687s`，297/298 低于 5 秒；298 张成图全部 800 px，233 个实际保存的题内裁图通过完整性检查，版面审计 0 异常。141 个可执行最终 PNG 回读通过，3 个低覆盖样本在更严格的当前回读规则下失败，其余 154 个因手写、拍照风险或独立硬错误跳过。v61 没有降低 `0.72` 阈值，而是把 `\sqrt`、`\underline{\hspace{...}}` 和 `\cdots` 转换为真正可读的根号、填空线和省略号：两条伪失败覆盖率由 `0.690/0.711` 恢复至 `0.925/0.952`，真正缺正文的 `TZIP-MATH-T03-Q02` 仍以 `0.667` 失败。该数据没有规范 OCR Gold、干净原稿配对和明确来源许可，只能证明真实扫描工程路径与安全门禁，不能计算准确率或替代 410 张自建拍照数据。

针对 v54 的 12 个 `QUESTION_NUMBER_MISSING` 硬拒绝，v56 已用当前 OCR 对原始图片逐张重跑：只允许从与当前保留题图空间相邻的高置信度印刷图注恢复数字主题号，禁止按样本 ID 或文件名猜测，也不自动恢复容易把 `III` 误读成 `II` 的分部后缀。6 题的题号已恢复并实际写入最终 PNG，不能只存在于 JSON；其中 3 题由 REJECT 安全降为 REVIEW，另 3 题因拍照、手写或未核验裁框仍保持 REJECT。剩余 6 题缺少可靠像素证据，继续硬拒绝。12 张成图均为 800 px，版面和文件完整性审计通过，3 张可执行的最终 PNG 回读全部通过；证据见 `artifacts/realworld_question_number_v56.json`。

v57 又对 4 个双栏选项异常样本执行了当前 OCR 重跑。只有 `TZIP-MATH-T06-Q03` 同时具备高置信度独立 B/D 标签、上下两行各左右一个高置信度公式、两列严格对齐、A/C 无冲突且四项内容均已存在，因此只补回丢失的 A/C 标签，不猜内容；最终成图显示完整 A–D。其他 3 题因内容或几何证据不完整保持原状。恢复题仍受 `OPTION_LABEL_RECOVERED_UNVERIFIED` 和“拍照页＋未核验裁图”硬门禁约束，没有因修复而静默放行。4/4 均低于 5 秒且宽度为 800 px，版面和完整性审计通过；证据见 `artifacts/realworld_option_grid_v57.json`。

v59 对 7 个缺失题内图样本执行了带编号图注的自适应裁图回归。只有 `TZIP-MATH-T08-Q03` 同时具备与当前题号一致的印刷图注和紧邻上方图形，因此从当前原图直接裁出完整函数图；其余 6 题没有足够像素证据，保持拒绝而不猜图。v61 随后关闭了两条公式命令名造成的回读伪失败，同时保留一条真实低覆盖失败。相关运行与审计记录保存在 `artifacts/runs/realworld_visual_missing_7_v59_adaptive_caption_tight_non_gold/` 和 `artifacts/realworld_readback_v61.json`。

## 图片处理硬规则

题目内的几何图、受力图、电路图、函数图、表格和其他插图必须从原题图片直接裁切并等比缩放。禁止重新绘制，禁止使用生成模型补图；裁框触边、标签不完整或混入手写时进入人工复核。

## 入口

- [实际 OCR 产品页（服务启动后）](http://127.0.0.1:8765/ocr/)
- [后台数据与成品验收台（服务启动后）](http://127.0.0.1:8765/review/)
- [完整工作流](WORKFLOW.md)
- [自建数据操作说明](docs/SELF_BUILT_DATASET.md)
- [Gold 任务清单](data/splits/gaokao_physics_ocr_gold300_manifest.jsonl)
- [Gold 当前统计](data/splits/gaokao_physics_ocr_gold300_summary.json)
- [Gold 状态](data/gold/gaokao_physics_300/STATUS.json)
- [复核任务格式](schemas/ocr_gold_review.schema.json)
- [正式 Gold 格式](schemas/ocr_gold.schema.json)
- [最终 PNG 人工验收格式](schemas/output_quality_review.schema.json)
- [410 行自建数据采集计划](templates/self_built_collection_plan.csv)
- [私有采集工作区工具](scripts/prepare_self_built_collection_workspace.py)
- [管线配置](configs/pipeline.json)
- [测试集注册表](configs/datasets.json)
- [外部评测集许可登记](configs/external_datasets.json)
- [外部回归基线](artifacts/external/BASELINE.json)
- [外部冻结诊断](artifacts/external/DIAGNOSIS.json)
- [本机数据角色与重复图片审计](artifacts/local_dataset_inventory.json)
- [Testzip 全量 298 题真实扫描回归](artifacts/realworld_full298_v54_product_v8.json)
- [Testzip 题号像素证据恢复回归](artifacts/realworld_question_number_v56.json)
- [Testzip 双栏选项几何恢复回归](artifacts/realworld_option_grid_v57.json)
- [最新物理 Gold v72 成熟候选证据](artifacts/gaokao_physics_v72_formula_line_arbitration.json)
- [Testzip v61 最终 PNG 回读修复证据](artifacts/realworld_readback_v61.json)

## 当前可运行能力

- 本机 PP-DocLayoutV2 + PP-FormulaNet 结构化文字/公式识别；连接拒绝、超时或服务重启中的 503 会按有界指数退避重试并在结果中留痕，`auto` 模式仍可降级到 RapidOCR/ONNX；坏图片、HTTP 400 等真实输入错误不会被重试掩盖；
- 结构化服务启动时使用五张程序生成的非数据集合成页，以“批量后逐张”的实际请求形状预热文字、公式和图形分支，不读取冻结 Gold；服务同时启用与 MinerU CLI 一致的稳定 cuDNN 运行策略；
- 题干和 A–D 选项结构化；数学正文中的 `A、B`、`OB.` 等变量标记不会抢占后面的真实 A–D 选项组；
- `question_document_v2` 为每个识别文字/公式块记录原图坐标、类型、置信度和处理状态；被裁掉的相邻题和手写块也以 `ignored + 原因` 留在台账中，缺少或存在未决来源记录的结果不能自动 `PASS`；
- 本地 KaTeX + Edge 无头渲染，真正排版分数、根号、上下标和集合符号；产品配置中的多次渲染尝试共享请求剩余的 5 秒端到端预算，避免浏览器偶发卡顿叠加；浏览器配置目录与截图路径均使用绝对路径，避免命令返回成功但临时 PNG 写入错误目录；
- 产品 v8 将重型 PNG 独立回读固定为哈希绑定的第二阶段：在线阶段完成 OCR、原图直裁、800 px 排版和轻量门禁后返回“后台复检中”，下载保持锁定；后台验证正文覆盖率、选项标签和 LaTeX 泄漏，网页轮询完成状态，只有最终 PASS 才由服务端开放下载。批处理可用 `verify-run` 恢复同一状态机；
- “如图”等视觉线索触发自动图框候选；图框内资产始终重新从当前原图直接裁切，绝不重绘；
- 800 px 固定宽度、动态高度 PNG；
- `PASS / REVIEW / REJECT` 质量门禁；公式未独立校验、拍照风险和候选图框均不能静默通过，确认检测到手写时硬拒绝；无法找到可靠题内图框时直接拒绝并要求补标，禁止用整题截图兜底；
- 对“求：”后与左侧分问并列的右侧手写解答列执行保守剔除；手写覆盖印刷文字或原题插图时不臆测修复，保持 `REJECT`；
- 如果输入图片的 SHA-256 精确命中一条已经由两名不同人员核验的“带手写照片—干净原稿”配对，流水线可改用该干净原稿继续识别和直接裁图；未核验、配对任务被改动、哈希变化或审核身份不完整时立即停止替换。该机制不做涂抹、修补或重绘，并在 `report.json` 中同时记录原始输入、实际处理源、双哈希、配对任务哈希和两名审核者；
- 检测完整选项组之后侵入的下一道印刷题，裁去后题并进入题界复核，不把后题公式误判为手写；
- 对跨页或大行距造成的上下叠字，只有每个字符都能与独立同行块对齐时才拆回对应行；对被文字检测拆散的上下分式，仅重跑原图中的紧凑公式区域。低置信度选项末尾疑似鼠标指针形成的单字母下标，必须经两次紧裁文字复核一致后才移除；缺失的首个 A 选项标签也只在 B–D 行距一致且两个原图紧裁结果都高置信度读成 A 时恢复；这些自动修正全部保留专用警告并强制人工确认；
- 题图分组对右侧长图注保留较大安全边距，但上下边距保持紧凑，防止补全图注时把下一题吞入裁图；拍照风险页还会检查直接裁出的题图，只有同一图块出现至少两个低于 0.65 的短文字识别结果时才作为手写证据硬拒绝，约 0.69 的正常印刷图标签不会因此误拒绝；
- 同一组选项图片会依据原图中的行、列空间顺序排序；当每个图块都有唯一且紧邻的 A–D 标签时，把图块与标签合并成一个原图直接裁框，原样保留选项对应关系，不把标签拆成正文，也不分别重排图块；
- 罗马分卷题号只在原图题号紧裁区执行直接二次识别，不再对整行重复文字检测；保留高置信度纠错能力，同时降低交互时延；
- 已知站点水印从正文和来源台账中移除；句尾答案槽、待填公式或选项疑似丢失时不能自动 `PASS`；
- 命令行 JSON 固定使用 UTF-8 输出，Windows 旧代码页不会让已成功生成的 Unicode 数学结果在打印报告时误报失败；
- 每个样本保存结构化 JSON、成品 PNG、直接裁图资产和质检报告；不复制输入原图，不保存日常中间图；
- 正式评测锁：Gold 数量、验证方法或预测覆盖不满足当前评测配置时拒绝输出正式准确率；人工制作的 Gold 必须双审，源生成 Gold 必须明确限制为来源派生基准。
- 正式发布锁：只有完整门禁为 `PASS`、公式已独立验证、PNG 回读通过且全部插图框已人工确认的样本才能进入发布目录。
- 全量批处理加入端到端时延熔断：至少处理 20 题后，最近 5 题中若有 3 题超过 5 秒，立即中止并记录窗口、题号和资源状态；该运行不会被复核站列为可发布版本。

当前已形成可运行的端到端成熟候选 `gaokao_physics_full300_v72_formula_line_arbitration_candidate`，实际产品接口为 `product_ocr_v8`，完整自动测试为 `383 passed`。它已经满足当前来源派生 300 题的机器门禁和逐题 5 秒目标，但还不是可外推到真实拍照分布的 `v1.0` 正式验收版。当前 39 个自动 REVIEW 不能由人工 PASS 覆盖；261 个自动 PASS 也必须完成哈希绑定的最终 PNG 人工验收后才能进入选择性发布包。精确手写分割、成对干净原稿和 410 张真实自建数据仍是完整产品发布阻塞项。外部网上测试集已接入隔离回归，不能替代上述正式 Gold。

## 安装和运行

工程已经建立本地 `.venv`。如需在另一台机器重建：

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements-ocr.txt
.\.venv\Scripts\python.exe -m pip install -e .
npm install --ignore-scripts
```

`npm install` 只安装锁定为 `0.16.11` 的本地 KaTeX 排版资源；当前机器也可自动发现已有离线资源。大型 MinerU 模型和 GPU 环境不提交 Git。

日常使用直接启动完整产品（会先确认结构化 OCR 服务，再启动网页）：

```powershell
.\scripts\start_product_ocr.ps1
```

日常重复启动会复用版本兼容的本机服务；修改 Python、配置或前端代码后使用 `.\scripts\start_product_ocr.ps1 -Restart` 强制重载同版本进程，避免页面继续运行旧代码。

随后打开 `http://127.0.0.1:8765/ocr/`。上传、拖入或粘贴图片后，页面会显示原图和实际保存的 800 px PNG、端到端用时、输出尺寸、识别引擎和门禁问题。只有 `PASS` 才开放“下载最终 PNG”；`REVIEW/REJECT` 仍可查看诊断图，但不会被误当成成熟成品。结果 ID 会写入网址，因此刷新页面后仍可恢复本次结果。正常结果无需逐题重复确认；只有发现实际错误时点击“发现问题”，勾选错误类型并加入 `artifacts/product_feedback/` 回归队列。反馈绑定当时的原图与成图 SHA-256，不修改 Gold；一旦登记问题，当前结果的下载会锁定，等待后续生成新版本。

启动工程自带的结构化 OCR 服务（只绑定本机回环地址；若 8240 已有健康且版本匹配的服务则直接复用）。首次启动会额外运行五张非数据集合成页，先批量再逐张预热实际请求形状，不接触冻结 Gold：

```powershell
.\scripts\start_structured_ocr.ps1
```

处理一张新图片：

```powershell
.\scripts\run_pipeline.ps1 -Image "D:\path\question.png" -SampleId "demo-001"
```

若要从命令行复现网页产品所用的安全时延配置，显式使用 `--runtime-profile product-v7-safe`；批次汇总会记录该配置名。对 Testzip 等非 Gold 数据只能检查运行稳定性、安全状态和行为漂移，不能据此计算正式 OCR 准确率：

```powershell
.\.venv\Scripts\python.exe -m test7_ocr batch `
  --manifest "D:\path\manifest.jsonl" `
  --dataset-root "D:\path\dataset" `
  --provider structured `
  --runtime-profile product-v7-safe `
  --run-id realworld-regression
```

对当前测试集做一次管线连通性检查（预标注不是 Gold，所有结果至少为 `REVIEW`）：

```powershell
.\.venv\Scripts\python.exe -m test7_ocr batch `
  --provider eval-prelabel `
  --allow-eval-prelabel `
  --limit 10
```

长批次中断后按样本 ID 和来源哈希安全续跑：

```powershell
.\.venv\Scripts\python.exe -m test7_ocr batch --provider auto --run-id <原运行ID> --resume
```

全量运行前只做磁盘空间预检，不加载 OCR 模型：

```powershell
.\.venv\Scripts\python.exe -m test7_ocr batch --storage-preflight-only
```

预检使用近期样本产物大小的 P95 乘以 1.5 估算新产物，并额外保留 2 GiB 安全空间；断点续跑只计算尚无有效报告的样本。结果写入 `artifacts/STORAGE_PREFLIGHT.json`，空间不足时批次会在加载模型前停止。

结果保存在 `artifacts/runs/<run_id>/<sample_id>/`：

```text
output.png                 800px 动态长图
question_document.json     结构化题目、来源哈希、裁图坐标
assets/*.png               从原题直接裁切的图形/表格
report.json                状态、问题码、耗时和产物哈希
```

在线阶段会严格保留 `elapsed_seconds` 作为用户等待时延。若独立 PNG 回读来不及在 5 秒预算内完成，结果保持 `REVIEW`，然后运行可恢复的发布前复核：

仅修改排版模板时，可以复用既有结构化识别文档、重新核对原图哈希并按记录坐标重裁视觉资产，不必再次调用 OCR 模型：

```powershell
.\.venv\Scripts\python.exe -m test7_ocr rerender `
  artifacts\runs\<source_run> --run-id <layout_run>
```

`rerender` 产物会明确写入 `measurement.mode=render_only`、`model_invoked=false` 和 `default_review_eligible=false`。它只能验证排版变化，不能用其耗时声称 OCR 达到 5 秒，也不能替代正式 Gold 准确率或自动成为网页默认基线。

```powershell
.\.venv\Scripts\python.exe -m test7_ocr verify-run artifacts\runs\<run_id>
```

复核会先核对当前原图和成品 PNG 的 SHA-256，防止串题或文件被替换，再原子更新每题的 `question_document.json` 和 `report.json`，生成 `deferred_readback_summary.json` 与 `index.finalized.jsonl`。这部分耗时记录在 `post_verification`，不篡改在线耗时；重复执行会跳过已经通过的样本。首轮只缺选项标签时会对同一不可变 PNG 做一次 1.5 倍彩色复读；若仍缺标签，再对同一 PNG 的无损灰度派生图做一次独立像素复读。只有三次均不能确认完整标签时才硬拒绝，灰度图不会替换或改写最终 PNG。仅由旧复读问题造成的 `REJECT` 允许重新验证，其他硬拒绝仍跳过。已经确定含手写或拍照风险的诊断样本会被跳过，仍不能发布。

`report.json` 还会单列 `pipeline_initialization`。`elapsed_seconds` 表示长期服务已预热后的单题交互耗时；首次启动 OCR/回读模型的耗时不能混入热请求数字，也不能省略，批次汇总会同时给出 `pipeline_initialization_seconds` 和 `first_processed_cold_equivalent_seconds`。

`artifacts/runs` 已被 Git 忽略。日常迭代不复制原始图片、不保存中间图，正式发布只选择性归档；因此 GitHub 不会被大量 PNG 填满。

运行自动测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
```

正式评分会优先读取 `index.finalized.jsonl`，并在配置要求的 Gold 数量未完成时主动锁定：

```powershell
.\.venv\Scripts\python.exe -m test7_ocr evaluate artifacts\runs\<run_id>
```

外部网上测试集（OmniDocBench、UniMER-Test、MathVision-Wild）只能走隔离评测，禁止写入正式 Gold 准确率，也不得用来调 PASS/REVIEW 阈值。WarpDoc/RealDAE 在许可登记为 `approved` 之前，下载脚本会直接失败。

冻结桶必须从 Hugging Face 实数据转换，**不要**再跑 `scripts/run_external_harness_baseline.py`：该脚本会用夹具覆盖 `data/splits` 里的冻结 jsonl。夹具冒烟报告只留在 `artifacts/external/*/harness_smoke/`。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\scripts\fetch_external_dataset.py omnidocbench
.\.venv\Scripts\python.exe .\scripts\fetch_external_dataset.py unimer_test
.\.venv\Scripts\python.exe .\scripts\fetch_external_dataset.py mathvision_wild
.\.venv\Scripts\python.exe .\scripts\prepare_external_manifests.py omnidocbench
.\.venv\Scripts\python.exe .\scripts\prepare_external_manifests.py unimer_test
.\.venv\Scripts\python.exe .\scripts\prepare_external_manifests.py mathvision_wild
.\.venv\Scripts\python.exe -m test7_ocr batch --manifest data\splits\omnidocbench_exam_zh.jsonl --dataset-root data\external\omnidocbench --runtime-profile product-v7-safe --dataset-role external_eval_only --run-id omnidocbench_exam_zh_v1_external --resume
.\.venv\Scripts\python.exe -m test7_ocr external-eval artifacts\runs\omnidocbench_exam_zh_v1_external --gold data\splits\omnidocbench_exam_zh_gold.jsonl --dataset-id omnidocbench --output artifacts\external\omnidocbench\frozen_v1\report.json
.\.venv\Scripts\python.exe -m test7_ocr batch --manifest data\splits\unimer_test_stratified400.jsonl --dataset-root data\external\unimer --runtime-profile product-v7-safe --dataset-role external_eval_only --run-id unimer_test_stratified400_v1_external --resume
.\.venv\Scripts\python.exe -m test7_ocr external-eval artifacts\runs\unimer_test_stratified400_v1_external --gold data\splits\unimer_test_stratified400_gold.jsonl --dataset-id unimer_test --output artifacts\external\unimer_test\frozen_v1\report.json
.\.venv\Scripts\python.exe -m test7_ocr batch --manifest data\splits\mathvision_wild_frozen.jsonl --dataset-root data\external\mathvision_wild --runtime-profile product-v7-safe --dataset-role external_eval_only --run-id mathvision_wild_frozen_v1_external --resume
.\.venv\Scripts\python.exe -m test7_ocr external-eval artifacts\runs\mathvision_wild_frozen_v1_external --gold data\splits\mathvision_wild_frozen_gold.jsonl --dataset-id mathvision_wild --output artifacts\external\mathvision_wild\frozen_v1\report.json
.\.venv\Scripts\python.exe .\scripts\assemble_external_baseline.py
.\.venv\Scripts\python.exe .\scripts\diagnose_external_runs.py
```

诊断摘要见 `artifacts/external/DIAGNOSIS.json`。它只解释 REJECT 码和崩溃类型，**禁止**把 CER、覆盖率或 REJECT 比例写进 `pipeline.json`、提示词或 PASS/REVIEW 规则。崩溃复测可对既有 jsonl 用 `--sample-id` / `--resume` 只跑失败样本；UniMER 公式裁图走清单字段 `scoring_mode: formula_only`（见 `unimer_test_smoke20.jsonl`），高考 300 整题不得使用该模式。`formula_only` 在版面框不可靠时对整图做 MFR，并用 layout/渲染再认做自洽核验（不读 Gold）；UniMER `formula_match` 仍非正式 Gold。迭代记录见 `artifacts/external/ITERATION_LOG.json`。

先完成上述 `verify-run`，并在本机复核页对实际最终 PNG 给出人工 PASS，再发布同时通过自动门禁和人工成品验收的结果：

```powershell
.\.venv\Scripts\python.exe -m test7_ocr release artifacts\runs\<run_id>
```

`REVIEW`、`REJECT`、公式未核验、手写污染、候选裁图框、PNG 回读失败、人工成品验收缺失或人工 FAIL 都会锁住对应样本。人工 PASS 不能覆盖任何自动安全门禁。发布包会包含哈希绑定的 `human_output_review.json` 审计记录，不复制输入原图，并再次逐像素验证裁图确实来自记录的实际处理源坐标；若使用双审干净原稿，还会重新验证输入/干净源双哈希、配对任务内容及两个人员签核。

## 一键检查

在 PowerShell 中运行：

```powershell
.\scripts\run_self_built.ps1
```

默认只输出一屏关键进度，包括 Gold、410 张真实采集、手写 Mask、题界、原稿配对、冻结状态和下一计划任务。需要查看各检查器的完整 JSON 时使用：

```powershell
.\scripts\run_self_built.ps1 -Detailed
```

严格检查 410 张真实采集、题界、成对原稿—照片和至少 100 张手写 Mask 的数量、隐私与双审冻结门禁：

```powershell
.\scripts\run_self_built.ps1 -RequireVerified
```

该命令现在还会同时检查手写像素 Mask 队列。候选数量会随人工成品验收中的“手写可见”证据实时变化；当前选出 22 张冻结候选，另会把“自建数据登记”中新加入的每张 `handwriting_capture` 自动同步为隔离 Mask 任务。当前仍是 `0/22` 已双人核验，距离至少 100 张的目标还缺 78 张。原图不复制，两类样本均禁止训练；只有实际绘制合法像素类别、核对原图哈希并由两名不同人员签核后，才计入完成数。可单独检查：

```powershell
.\.venv\Scripts\python.exe .\scripts\prepare_handwriting_mask_eval.py --check
```

发布前硬检查使用 `--check --require-complete`；在当前 `0/22` 状态下按设计失败，不能把空任务或自动差分伪装成真值。

历史 Testzip 人工复核草稿位于 `data/annotations/testzip_math_physics_ocr_gold/drafts`。该目录默认不提交 Git，只保留审计轨迹和安全回归用途；它不能再晋升或覆盖当前高考物理 300 题主 Gold。

启动本机复核页面：

```powershell
.\scripts\start_gold_review.ps1
```

页面固定绑定 `127.0.0.1`。日常产品默认打开 `http://127.0.0.1:8765/ocr/`，后台验收固定打开 `http://127.0.0.1:8765/review/`。验收台默认入口是“最终 PNG 验收”：按 `run_id` 并排显示原题和流水线实际保存的 800 px PNG，人工逐项确认语义、公式渲染、LaTeX 源码、手写、原图裁图、相邻题/噪声、截断、填空线/单位和清晰度。PASS 要求全部问题均被明确标记为不存在；FAIL 必须记录至少一项实际问题。验收记录单独保存在 `data/annotations/output_quality_reviews/<run_id>/`，并同时绑定原图和成品 SHA-256，不会写入或污染 OCR Gold。

“金标制作”是独立入口。它会核验原图 SHA-256，锁定来源和候选预标注字段，并支持在 1280×720 原图上拖框标记印刷文字、原题插图、手写和忽略区域。候选区默认只显示无 LaTeX 命令的可读预览；点击“采用候选作为编辑起点”后，服务端会把异常公式分隔符清理并拆成普通题干和 A–D 选项输入框，不需要编辑 JSON。规范转写提供本地 KaTeX 视觉预览，原始源码仅在“高级”区域显示，不能被当作最终效果验收。“已验证”状态要求一名标注员、两名不同复核员、人工公式状态、文字框和六项检查全部完成；选择题还必须至少有两个标签唯一且内容非空的选项。

页面默认只列出自动未拒绝、确实可进入人工验收的成图；自动 `REJECT/FAIL` 不再抢占首页，而是保留在“自动 REJECT”专项筛选中进行失败原因审计。页面还支持按人工结论、旧版曾 FAIL、当前图片与旧 FAIL 相同、无历史记录等条件筛选，以及风险优先/样本编号排序、样本搜索、上一题/下一题、保存进度、确认结论并下一题、`Ctrl+S`、`Ctrl+Enter`、角色编号记忆和进度 CSV 导出。任务列表会直接标出“旧 FAIL”及“同图仍曾 FAIL”，方便先关闭真实历史问题。最终验收页把十项人工检查分成“文字与语义、公式、原题裁图、手写/杂质/版面”四组，并为每项给出具体核对方法；自动风险或历史 FAIL 只在组内置顶并标记重点，不会替人工勾选。人工 PASS/FAIL 不再使用独立下拉框，而是由十项人工选择确定：任一项有问题即为 FAIL，十项全部明确正常才为 PASS，其余均为待完成，从而避免结论与问题清单互相矛盾。

页面可单图放大，也可打开全屏并排对照，以 50%–300% 缩放并按相对位置联动滚动长图。全屏对照底部可直接逐项判定，支持 `1=正常`、`2=有问题`、左右方向键切换，不必反复关闭大图。默认只进入最新全量运行，不再被浏览器记住的旧运行覆盖；显式网址中的 `run` 仍可用于历史审计，并在页面醒目标为历史版本。自动 `REJECT/FAIL` 及当前 PNG 与历史人工 FAIL 哈希完全相同的结果不会进入普通人工队列，前者只供定位失败，后者必须重新生成，服务端也拒绝重复提交结论。“上一题/下一题”在筛选队列内循环切换；只有一题或没有题时显示明确原因。直接打开本地 `review_ui/index.html` 会自动跳转到 `http://127.0.0.1:8765/review/`，避免绕过服务端。

页面同时显示真正需要人工验收的数量、自动拒绝数、同图待重生数和服务器记录状态；点击“确认结论并下一题”后会保留明确的服务器保存回执。就绪区实时显示原图、800 px 成图、检查完成度、验收人和派生结论五项状态。未完成时“保存当前进度”会写入 `pending`，只有双图加载成功、成图宽度为 800 px、十项检查完成且填写验收人时，才能确认结论。尚未保存的勾选、结论和备注也会保存为浏览器本机草稿；只有运行、题号、原图哈希、成品哈希和 ETag 全部相同时才恢复。人工 PASS 不能覆盖自动拒绝。“最终 PNG 验收”可导出流水线状态、人工结论、问题类型、时延、尺寸和双哈希；“金标制作”可导出标注进度。命中双审干净原稿的结果会在列表标为“双审干净原稿”，自动报告显示输入哈希到干净源哈希的转换；左侧仍保留原始输入用于人工对照。“自建数据登记”可直接下载与批量导入格式完全一致的 410 行采集计划，并显示私有采集目录的真实缺图、格式、隐私声明和可增量预检行数。

当前 `UI 34 / review_workstation_v32 / product_ocr_v8` 会自动升级旧进度缓存，并列出全部当前可执行的计划任务；主高考物理 Gold 队列和 Testzip 历史输出复核可在同一服务中解析，但不会混合为同一个 Gold。产品固定入口为 `/ocr/`，测试与验收固定入口为 `/review/`；旧的根路径查询链接会保留参数并重定向到 `/review/`，两个页面也以明确标题和互链标出各自职责。选定采集任务后会锁定角色、学科、设备、条件、命名和干净原稿配对，支持拖入图片或调用设备相机。登记图片优先以硬链接连接内容寻址隔离区和计划目录，避免重复占用空间。首题加载前显示明确载入提示；服务重启或版本切换时网页会自动重连，原图或成品 PNG 瞬时加载失败会自动重试三次。网页首先服从服务端 `default_eligible` 资格，再比较完成度、数量和时间，因此慢批次、资源冲突诊断或不完整批次不会因更新时间较新而抢占默认验收版本。纯排版运行标为“仅排版候选”，不得计入 5 秒在线指标。人工成品结论仅在原图哈希一致且解码后的 RGB 显示像素逐点完全一致时沿用；PNG 压缩或元数据变化不制造重复验收，任何真实像素变化仍恢复为待验收。

第三个入口“手写 Mask 标注”同时读取人工证据筛选的冻结测试任务（当前 14 个），以及从自建登记自动生成的带手写照片隔离任务，并在列表明确区分“冻结测试”和“自建隔离”。页面提供六类像素笔刷、大小和透明度调节、覆盖层开关、逐笔撤销、像素统计、备注、状态筛选和进度 CSV。保存时由服务端把浏览器画布规范化为单通道 `L` 模式 PNG，重新计算像素类别数量和 SHA-256，并使用内容寻址文件名保留可追溯版本；空白 Mask、错误尺寸、彩色值、非法类别和来源哈希变化都会被拒绝。标注员提交后必须由另一名复核员完成五项检查，修改已签 Mask 会自动清除签核并退回草稿。两类任务都禁止训练，自建任务在 D0-E 冻结前也禁止评测。

第四个入口“自建数据登记”用于补充 50 份干净原稿、200 张手机照片、100 张带手写照片和 60 页题界样本。图片登记前必须明确学科、学段、原稿组、来源许可和隐私检查；手机照片与带手写照片必须关联已登记的同一份干净原稿，并记录匿名设备编号和拍摄条件。原图以 SHA-256 内容寻址，相同字节只保存一次，默认写入 `D:\robot\Documents\ASK\datasets\test7_ocr_self_built_v1\sources`；登记记录进入 `quarantine`，`training_allowed=false`、`evaluation_allowed=false`，不会因上传成功就变成 Gold，也不会随项目 Git 上传。

大量图片不需要逐张在网页登记。把 `templates/self_built_intake_batch.csv` 复制到待导入图片所在目录，填写后先运行只读预检；只有整张表的路径、图片格式、配对、设备、条件、来源、隐私和分发权限全部通过，才显式加入 `--apply`。CSV 内的拍照行可以排在原稿行之前，导入器会先解析整批依赖并在实际登记时先写原稿。相同图片字节、越界路径和预检后发生变化的图片都会拒绝；成功后自动建立 Mask、拍照配对和题界复核任务。

如果按完整目标一次采集，可直接使用 `templates/self_built_collection_plan.csv`，或在网页“自建数据登记”中下载。它预先分配 410 行：数学/物理各 25 份干净原稿，每份 4 张普通拍照和 2 张带手写拍照，另有单双栏整页各 30 张；300 张拍照均衡分配给 `phone-a/b/c`，并覆盖斜拍、阴影、折痕/曲面、模糊、正拍和不均匀光。计划不会预填隐私确认，真实图片放入约定路径后仍必须逐行填写 `privacy_checked` 和 `contains_personal_data`。`run_self_built.ps1` 会先验证该模板未损坏。

不要把真实采集图片放进工程的 `templates` 目录。运行下面的命令会在数据集目录建立独立的私有采集工作区，复制一份可编辑的 410 行工作清单并创建全部图片目录；再次运行只刷新就绪报告，绝不会覆盖已经填写的 CSV：

```powershell
.\.venv\Scripts\python.exe .\scripts\prepare_self_built_collection_workspace.py
```

默认位置是 `D:\robot\Documents\ASK\datasets\test7_ocr_self_built_v1\collection`。真实图片按工作清单放进其中的 `images` 子目录；总体报告保存为 `collection_readiness.json`，全部缺图路径保存为 `collection_missing_files.csv`。系统还会生成 `collection_ready_batch.csv`：只包含图片有效、隐私声明完成且干净原稿依赖同时就绪的行，可直接用于边采集边执行只读预检；人工编辑的全量工作清单不会被覆盖。只有 410 张图片齐全、格式有效、每行 `privacy_checked=true`、`contains_personal_data=false` 且分发权限明确后，`ready_for_import_preflight` 才会变为 `true`。这些图片与登记后的内容寻址源图都不进入 Git。

```powershell
# 默认只检查，不写入图片或登记记录
.\.venv\Scripts\python.exe .\scripts\import_self_built_batch.py D:\待采集批次\batch.csv

# 查看预检输出无误后，才执行真实登记
.\.venv\Scripts\python.exe .\scripts\import_self_built_batch.py D:\待采集批次\batch.csv --apply
```

正式登记报告保存到私有的 `data/annotations/self_built_intake/batches/BATCH-*.json`，记录清单哈希、逐行图片哈希、生成的资产 ID、已登记/新登记状态和失败原因，但不复制图片。重复执行同一批次是幂等的，不会产生第二份源图；默认预检不创建任何文件。

第五个入口“拍照配对 Gold”同时显示干净原稿和手机照片/带手写照片。上传时指定的配对只生成候选任务；标注员必须确认同一页、内容边缘完整、质量可用、条件标签、隐私和手写状态，另一名复核员再完成五项检查。普通照片必须确认无手写，`handwriting_capture` 必须确认存在手写；不合格照片写明原因后拒绝。来源双哈希、角色和签核均由服务端保护，当前真实进度为 `0/50` 份原稿、`0/200` 普通照片对、`0/100` 带手写照片对。

第六个入口“整页题界 Gold”只处理在第四个入口登记为 `boundary_page` 的真实整页图片。标注员在原图上框出至少两道完整题，填写题号、单双栏、父题和跨页关系并排列阅读顺序；另一名复核员完成五项检查后签核。来源图、题框和签核均受哈希与 ETag 保护，修改已签题框会使签核失效。当前真实登记数和核验数仍为 0，不会使用现有单题测试集或合成页面冒充题界 Gold。

可独立检查登记文件、配对关系和采集目标：

```powershell
.\.venv\Scripts\python.exe .\scripts\check_self_built_intake.py
```

加入 `--require-targets` 后会要求全部数量、至少 3 台设备以及斜拍、阴影、折痕/曲面、模糊或压缩条件齐备；当前没有登记真实图片，因此严格检查按设计失败。

整页题界 Gold 另有独立检查器：

```powershell
.\.venv\Scripts\python.exe .\scripts\check_question_boundary_gold.py
```

加入 `--require-targets` 后会同时要求双人核验完成的 60 页、300 道题、30 页单栏和 30 页双栏；它只接受 `boundary_page` 隔离登记及其人工题框，不会自动生成 Gold。拍照配对使用 `scripts/check_capture_pair_gold.py`，严格模式要求 50 份已核验原稿、200 对普通照片、100 对带手写照片、至少 3 台设备及必要拍摄条件。

D0-E 冻结器会联合检查 OCR Gold、Mask、题界、拍照配对、来源许可与隐私、文档组隔离、跨数据集 SHA-256 泄漏和文件哈希。日常只生成就绪报告；全部目标和双审完成后才允许创建不可覆盖的版本：

```powershell
.\.venv\Scripts\python.exe .\scripts\freeze_self_built_dataset.py --require-targets
.\.venv\Scripts\python.exe .\scripts\freeze_self_built_dataset.py --version test7-self-built-v2026.09.06-01
.\.venv\Scripts\python.exe .\scripts\freeze_self_built_dataset.py --verify-version test7-self-built-v2026.09.06-01
```

冻结包位于本机私有 `data/frozen/<version>`，只复制必要的 Mask Gold 和结构化清单，不复制原始试卷或拍照图片；每个文件都写入 SHA-256 和大小，版本目录一旦存在即拒绝覆盖。当前就绪状态写入 `data/splits/self_built_freeze_readiness.json`。`run_self_built.ps1 -RequireVerified` 现在执行 OCR Gold、手写 Mask、题界 Gold、拍照配对 Gold、原始采集和 D0-E 冻结就绪六类门禁，并在最后汇总所有未完成项。

最终验收页还可直接筛选“最终 PNG 回读异常”“图片裁剪/图框风险”“手写/拍照风险”和“处理超过 5 秒”，列表会标出 `PNG 回读异常` 与 `超 5 秒`；默认风险排序先列未验收项，再依次考虑同图旧 FAIL、历史 FAIL、自动拒绝、PNG 回读、手写、裁图和时延风险。

## 当前完整行为基线（2026-09-07）

当前网页默认运行是 `official_full298_v58_hybrid_gated`。它使用 PP-DocLayoutV2/PP-FormulaNet 提供版面与公式框、RapidOCR 读取印刷文字，并继续执行原图直接裁图、来源台账、KaTeX/安全降级排版和最终 PNG 回读。唯一测试集的 298 个非重复样本均成功生成，0 个运行错误；延期回读后为 228 个 `REVIEW`、70 个 `REJECT`。默认人工队列只包含 228 个未被自动拒绝的结果。

v58 的 298 张 PNG 全部为 800 px 宽，动态高度 133–1907 px。174 题包含题内视觉资产，共 245 个裁图，245/245 均来自本次实际处理原图的记录坐标；重绘与整题截图兜底均为 0。未发现显示态 LaTeX 源码；7 个最终结果仍带 PNG 回读低覆盖或选项告警并保持复核/拒绝。题号缺失、A–D 选项不完整、确认手写、无法找到必要图框和 PNG 回读丢选项均采用封闭失败。

v58 热请求平均 `2.823s`、P50 `2.906s`、P95 `3.868s`、最大 `4.703s`，298/298 小于 5 秒；初始化 `1.645s`，首题冷启动等效 `5.282s`。前一完整轮 `official_full298_v55_hybrid_gated` 也达到 298/298 小于 5 秒、P95 `3.856s`、最大 `4.706s`，说明当前热请求目标已连续两轮满足；冷启动仍略高于 5 秒，不能声称所有启动场景都已达标。

v42 是首个混合识别全量候选：298/298 成功，但出现 80 个相对 v17 的行为差异，人工抽查发现干净双分式被误删、上一题选项残留、页尾水印乱码、手写填数表格误入复核等问题，因此从未晋升。v58 已针对这些失败增加窄范围修复与门禁：打印态双层分式不再被当作手写；带低置信数字的规则网格硬拒绝；图旁游离手写数字、上一题单个 D 选项和长数字水印从成品移除。定向运行 `targeted_v57_safety_fixes_final` 的 6/6 均小于 5 秒，干净样本保持 `REVIEW`，两张确认含手写的样本保持 `REJECT`。

当前自动测试为 211 项全部通过。v55→v58 行为比较只有 3 项被标记：干净等式由 `REJECT` 恢复为 `REVIEW`、手写幻方由 `REVIEW` 改为 `REJECT`，以及一张始终为 `REJECT` 的样本因公式识别非确定性产生文本/公式数差异；未出现自动拒绝被无解释放宽的其他样本。

## 最新换行排版诊断（2026-09-07）

最终渲染层现将扫描/OCR 产生的普通物理换行恢复为连续段落，再按 800 px 成品的实际可用宽度重新流式换行；A–D 选项、`（1）（2）` 分问、章节标题、带圈序号和实验部分标题仍保留强制断行。中文禁则避免逗号、句号和右括号落在行首，也避免把短英文/数字词元随意拆开。规范文本、OCR 块、来源坐标和裁图坐标保持原样，变化只发生在最终 PNG/HTML 渲染层。

选项区进一步按最终 704 px 内容宽度自适应：能够完整容纳的短选项使用两列，长选项继续单列，任何选项都不会为了填满一行而拆开公式。最初的 4 列诊断在 `TZIP-MATH-T01-Q03` 上被最终 PNG 回读检出 B 标签缺失，因此 4 列方案已被否决并保留为失败记录 `targeted_v74_option_layout_4`。收紧到最多两列后，`targeted_v75_option_pack_max2_q03` 的 A–D 回读 1/1 通过；当前 Q01 的两列版本为 `800×258`，相对单列的 341 px 明显减少空白。只有标签连续为 A–D 前缀、无重复且每项内容非空的结构才允许进入选项网格；异常结构按完整题文安全排版。最终全量分布为 47 道两列、251 道单列或无选项网格。

纯排版全量候选 `layout_full298_v88_compact_readable_gated` 已完成 298/298 重渲染和最终 PNG 复读，0 运行错误。相对 v66，52 张变短、246 张不变、0 张变高，总高度减少 4347 px，最大单题减少 126 px；298 张均为 800 px 宽，298/298 规范显示文本逐字一致。171 题的 235 个视觉资产全部按当前原图记录坐标重新直裁，0 重绘；248 个 KaTeX 输出的截图前可见源码探针均为 0，其余 50 个安全文字输出的显示文本也不含反斜杠或 `$` 源码。加入灰度独立复读后，两张公式选项图的彩色 OCR 误拒绝被像素证据纠正；最终状态为 235 条 `REVIEW`、63 条 `REJECT`，142 个被实际复读的成品通过、3 个保守保留低覆盖告警，未检出 LaTeX 源码。该运行复用了 v66 的识别结果，因此仍是排版候选、不是 OCR 性能或准确率基线；正式 OCR Gold 仍为 `0/298`，网页默认继续使用 v58。

后续候选 `layout_full298_v89_visual_cue_readback_gated` 收紧了“必须含题图”的语义判定：孤立的“函数图像向右平移”“在平面直角坐标系中”“函数图象在某点的切线”等抽象文字不再被误判为缺图，只有“如图、图中、下列图像”等明确指向可见图片的表达才触发缺图硬门禁。全量比较中 298/298 显示文本一致、298/298 PNG 像素哈希与 v88 一致，只有 7 个已核对的纯文字样本从误拒绝恢复为 `REVIEW`；真实附图题没有放宽。最终 PNG 复读处理 147 张，144 张通过、3 张保守低覆盖、0 执行失败、0 LaTeX 源码命中；最终状态为 242 条 `REVIEW`、56 条 `REJECT`。它仍是纯排版/门禁候选，不替代正式 OCR Gold 或端到端时延基线。

结构恢复候选 `layout_full298_v90_structure_recovery_gated` 只在 A–D 连续序列证据充分时恢复漏掉句点或被识别为逗号的选项标签，并保守恢复题首“数字 + 中文题干”中漏掉的题号标点。两个干净样本恢复为完整 A–D 结构；含拍照手写或未核验题图的样本新增硬门禁，不能因结构恢复进入普通人工队列。全批为 244 条 `REVIEW`、54 条 `REJECT`，显示文本没有被改写。

最新排版候选 `layout_full298_v91_adaptive_wrap_gated` 增加自适应段落换行：2–6 行短题干采用均衡换行，较长题干采用防孤行换行；选项仍使用保守的漂亮换行，不强制两端对齐，也不把公式从中间硬拆。相对 v90，148 张 PNG 的断行位置变化，298/298 显示文本逐字一致、298/298 状态一致、298/298 宽度为 800 px，行为比较回归数为 0；248 张使用 KaTeX、50 张使用安全文字渲染、0 张降级渲染。最终 PNG 回读为 144 张通过、3 张因低覆盖继续保守复核、0 LaTeX 源码命中、0 执行失败；235/235 视觉资产仍为当前原图直接裁切。它是纯排版候选，不能替代 OCR Gold 或端到端 5 秒成绩，网页默认仍保留 v58。

排版审计候选 `layout_full298_v95_balanced_metrics_gated` 按浏览器最终字形框统计每一实际行，并按 KaTeX 的顶层关系式分段测量长公式；自然段或分问的末行不作为“题干内部短行”误报。248 个 KaTeX 成品中，内部行占用宽度最低为 `0.5176`、P05 为 `0.6004`，低于 50% 的候选为 0；另 50 个安全文字成品不提供浏览器行框，单独列为未测量而不是伪造分数。该批次与 v94 的 298/298 PNG 哈希、显示文本和状态全部一致，0 降级渲染，行为比较回归为 0；147 份最终 PNG 回读证据只在输出哈希完全相同时复用。可用 `scripts/audit_layout_quality.py` 重复生成审计记录，当前记录位于 `artifacts/diagnostics/v95_layout_quality_audit.json`。它仍是纯排版质量证据，不是正式 OCR Gold 或端到端 5 秒成绩。

统一排版候选 `layout_full298_v99_unified_edge_visual_options_gated` 将原来 50 张 Pillow 纯文字成品也纳入 Edge/KaTeX HTML 排版，使 298/298 都能接受相同的字体、换行、源码探针和实际行框审计。审计为 298 可测、0 缺测、0 降级、内部行最低填充 `0.5176`、低于 50% 为 0。对同时满足“四个 A–D、五个同排原图裁块、前三项无文字、末项为视图方向说明”的图片选择题，系统保守恢复为一张题图和四个带标签的图片选项；所有图片仍来自原图直接裁切，0 重绘。无任何视觉资产支撑的空选项会新增 `OPTION_CONTENT_MISSING` 硬拒绝，因此 `TZIP-MATH-T10-Q01` 从 REVIEW 收紧为 REJECT；其余 297 题状态不变，298/298 显示文本和裁图数量不变。延期回读共 147 张：144 通过、3 张低覆盖继续复核、0 执行失败、0 LaTeX 源码；当前仍是纯排版候选，不替代正式 OCR Gold 或端到端时延成绩。

换行修正候选 `layout_full298_v100_pretty_wrap_gated` 不再把 2–6 行题干强制均衡，而让普通正文的非末行尽量排满、仅允许自然段末行留白；多行联立式、分段式后若仍有正文，则在完整公式后另起一行，避免把阅读顺序挤到高公式右侧。相对 v99，Q01 的内部行填充由 `0.6588` 提升到 `0.8471`；全批内部行 P05 由 `0.6122` 提升到 `0.8190`，低于 70% 的题由 31 道降到 2 道。298/298 均为 Edge/KaTeX、宽 800 px，0 降级、0 KaTeX 错误、0 可见 LaTeX 源码；文本、公式数、裁图数和自动状态相对 v99 均为零回归。独立成品回读没有新增低覆盖，仅原有 2 张继续 REVIEW。该运行复用既有识别结果，所以仍是纯排版候选，不可作为 OCR 准确率或 5 秒端到端成绩。

完整诊断运行 `official_full298_v66_reflow_quality_gated` 生成 298/298、0 运行错误；离线回读后为 232 条 `REVIEW`、66 条 `REJECT`。298 张 PNG 全部宽 800 px，高度 `133–1739px`；171 题的 235 个视觉资产全部为当前原图直接裁切，重绘与整题兜底为 0，显示态 LaTeX 源码告警为 0。与 v58 比较，全部样本中 183 题变短、76 题不变、39 题变高，高度差中位数为 `-41px`；仅看最终 `REVIEW` 队列则为 137 题变短、64 题不变、31 题变高，同样中位数 `-41px`。最大变化样本已目视抽查，长题干能自然占满行宽，选项、分问及原题裁图保持分离。

v66 的服务时延明显抖动：平均 `5.929s`、P50 `5.267s`、P95 `10.499s`、最大 `29.333s`，仅 142/298 小于 5 秒；因此它只是排版诊断版，明确不得替换 v58 默认基线或用于发布。定向运行 `targeted_v67_reflow_review_gate_5` 进一步覆盖当前 Q01、长物理题、多裁图题、实验题和相邻题污染；发现“当前题号出现在上一题某个选项后续行”时新增 `ADJACENT_QUESTION_EMBEDDED_IN_OPTION` 硬拒绝，污染样本已从人工队列排除。正式 OCR Gold 仍为 `0/298`，所有行为数字均不是准确率。

复核服务的默认运行排序同时加入资格门禁：只有已完成离线回读、报告数与批次尝试数一致、0 运行失败且所有样本均低于 5 秒的运行才优先成为默认；不合格诊断版仍可显式选择。服务重启后的 `/api/runs` 已验证首项仍为 `official_full298_v58_hybrid_gated`，不会因为 v66 更新时间较新而误切换。

管线关闭时现在显式回收最终 PNG 回读工作进程；拍照题的多裁图手写检查改为紧凑货架式拼图，避免用最大裁块尺寸填满每个网格单元。定向运行 `targeted_v69_compact_visual_collage_2` 为 2/2 `REVIEW` 且全部低于 5 秒，平均 `4.107s`、P95 `4.826s`、最大 `4.906s`。这证明目标慢样本的局部优化有效，但两题结果不能代替全量性能验收。

随后全量诊断 `official_full298_v70_reflow_worker_clean` 生成 298/298、0 运行错误，离线回读后为 231 条 `REVIEW`、67 条 `REJECT`；但平均 `7.146s`、P95 `13.090s`、最大 `57.861s`，仅 113/298 小于 5 秒。处理顺序每 25 题分段后，前 25 题为 25/25 达标，51–75 题降至 18/25，76–100 题降至 7/25，最后 23 题为 0/23；主要增长位于结构化识别服务耗时。运行期间另发现 8260 端口上存在 `test6` 的本地公式 OCR 服务并占用同一块 GPU，因此 v70 的时延只能作为资源竞争诊断，不能作为独占性能结论。分段报告位于 `artifacts/diagnostics/v70_latency_windows.json`。

v66→v70 行为比较另发现 `TZIP-PHYS-T03-Q07` 的逃逸速度公式从正确的根式退化为普通文本，虽然自动状态仍是 `REVIEW`，但仍属于真实质量回退。因此 v70 同时因性能环境不干净和公式行为回退而不得晋升。批处理现新增本机资源冲突保护：正式运行前和运行中每 10 题探测配置中的冲突服务；启动前命中则返回 `BLOCKED_RESOURCE_CONFLICT`（退出码 4），不加载模型也不创建候选目录；中途命中则保留已完成样本并将批次标为 `ABORTED`。只有显式 `--allow-resource-conflict` 才能做诊断运行，带 `ALLOWED_CONFLICT` 的结果永远没有网页默认资格。真实探针已确认 8260 活跃时 `resource_guard_probe_v71` 被启动前阻断且未创建目录；没有停止或修改该服务。

该公式退化的直接原因是：版面模型先把根号分式拆成上下两行，紧裁公式重识别在资源竞争中超时，旧门禁只保留了普通 `REVIEW`。现在只要候选分式紧裁重识别报错、超时或仍得不到完整公式，就产生 `STACKED_FRACTION_REFINEMENT_FAILED` 硬拒绝，不再进入普通人工验收队列。实题诊断 `targeted_v72_fraction_fail_closed` 成功恢复完整 `ν′=√(2Gm/R)`，输出 `800×785`，热链路 `4.964s`，离线最终 PNG 回读 1/1 通过，成品 SHA-256 与 v66 的正确版本一致；该单题只验证修复，不作为全量晋升证据。

行为探针清单位于 `data/splits/hybrid_formula_regression_probe.jsonl`。比较两个运行可执行：

```powershell
.\.venv\Scripts\python.exe .\scripts\compare_behavior_runs.py `
  artifacts\runs\<baseline_run> artifacts\runs\<candidate_run> `
  --output artifacts\runs\<candidate_run>\behavior_comparison.json
```

该比较只能发现相对旧运行的文本、公式数、裁图数、状态或时延变化，不判断旧结果是否正确，也不能替代 OCR Gold。`hybrid` 已完成两轮 298 题全量行为运行，但正式 OCR Gold 仍为 `0/298`，因此不得把行为结果宣传为识别准确率或成熟版本验收。

结构化 OCR 服务现在必须同时返回本项目标识、精确协议版本和精确引擎名；旧版或其他服务即使占用了 8240 端口也会明确报错，不再被启动脚本复用，也不会静默降级为另一种识别结果。每题报告新增服务端模型耗时、请求往返耗时、工程后处理耗时和重试次数。当前项目服务的小样诊断确认三题的规范文本、裁图数、状态和 PNG 哈希与 v17 完全一致，但首次新题平均仍为 `7.392s`，其中结构化服务识别平均约 `5.834s`；因此该改动解决的是结果来源和可诊断性，不代表 5 秒目标已经达成。

若 8765 端口上已经运行本工程的 300 题复核服务，启动脚本会直接复用，不会创建重复进程。

旧工程的 233 个插图结果只用于反推出“候选坐标”。当前接受了 134 个高相关候选，保存在 `data/splits/testzip_visual_bbox_candidates.jsonl`；运行时仍从当前原题重新裁切，旧插图像素不会进入成品，且候选框在人工确认前不能算 Gold。

## 历史排版成熟候选（v139，2026-09-08）

`layout_full298_v139_mature_safe_candidate_gated` 是当前排版与安全处理的成熟候选。它从稳定的 v127 出发，把最新人工反馈收敛成一组窄范围、可审计的通用修正：选项标签旁的低置信光标伪影、严格两栏 A/B/D 中缺失的 C 标签、完整 A–D 后的低置信相邻题尾巴、γ 射线伪数字、能级转移方向、热量 Q 伪下标、T₅/Tₛ 混淆、分号误作下标 i、上下堆叠分母，以及导函数题中由同一行定义和坐标共同证明的漏撇号。另有两项原图直裁修复：统计题把被扁平 OCR 的列联表和临界值表恢复为第二个表格裁片；三联物理图依据甲/乙/丙标签行扩展右边界并裁除顶部几像素残字。所有规则都依赖题意、邻接结构、原图坐标和 OCR 置信度，不按样本 ID 写死。

本轮 298/298 成功生成、0 运行错误，最终为 234 `REVIEW`、64 `REJECT`；全部为 800 px 宽、动态高度 133–1563 px。298/298 使用 Edge/KaTeX，0 安全备用排版、0 KaTeX 错误、0 可见 LaTeX 源码。实际行框审计通过：内部行填充最低 `0.5882`、P05 `0.8229`、中位数 `0.9490`，没有低于 50% 的异常短内部行。全批 171 道题含 232 个视觉资产，逐个与当前测试集原图坐标重裁并做像素哈希校验，0 错配、0 重绘。相对 v127 恰好只有 12 张预期成品变化，其余 286 张 PNG 的 SHA-256 完全一致。

保守行为比较器累计报告 3 个警报，均保留为已人工目视核实的预期修复：`TZIP-MATH-T07-Q01` 与 `TZIP-MATH-T07-Q08` 在恢复完整 A–D 后由 `REJECT` 变为 `REVIEW`；`TZIP-MATH-T03-Q11` 因用原图表格裁片替代扁平表格文字而出现文本长度变化和视觉资产增加。没有发现未解释的额外变化。延期回读累计 140 题通过，只有 `TZIP-MATH-T04-Q06` 与 `TZIP-PHYS-T06-Q06` 两个既有低覆盖结果继续保守保留；92 个拍照风险和 64 个自动拒绝样本按策略跳过。

当时的复核工作台为 `UI 32 / review_workstation_v28`。v139 安全沿用 143 项仍有效结论（118 PASS、25 FAIL）；99 个 REVIEW 因成品变化、证据变化或旧结论不足重新开放确认，56 个未继承结论的自动 REJECT 仅供失败审计。此次修复的导函数、统计表和三联图均不再继承对应旧 FAIL。该历史节点全套自动测试为 271 项通过。

该节仅保留历史迭代证据。Testzip 后来已停止作为正式 OCR Gold，当前唯一正式来源派生基准见文首的高考物理 300 题；v139 的纯排版耗时也不能替代端到端 5 秒证据。真实自建采集在该节点以及当前都仍是 `0/410`。题图始终只从当前原图直接裁切；手写与必要印刷线条重叠、无法证明可无损分离时继续安全拒绝，不做像素修补或生成式擦除。

端到端候选 `official_full298_v113_mature_safe_gated` 在 8260 的其他公式 OCR 服务和一个 test5 后台 Python 任务同时存在时留下 108/298 题记录，其中 34 题超过 5 秒、最大 `11.302s`，已主动停止且不得晋升。相同 5 题在重启本项目 8240 服务前后分别平均 `9.446s` 与 `9.605s`，证明当前慢点不是 8240 长时间运行衰减；把本地文字 OCR 从 8 线程降为 4 线程后反而增至 `11.132s`，配置已恢复为 8 线程。要生成新的正式端到端成熟版，必须先得到无同类资源冲突的运行窗口。批处理现在会用滚动窗口自动熔断持续超时，避免再次跑完整套后才发现不合格。

## 最新人工反馈闭环候选（v151，2026-09-10）

`layout_full298_v151_idempotent_candidate_gated` 接续已全部审核的 v139，把 16 个最新人工 FAIL 收敛为通用、坐标和上下文联合约束的修复：移除屏幕工具栏、下一题残片、页码和页脚；补回答题横线、被黑三角误识别的加速度答案位和重力加速度公式答案位；消除 `mc²` 后重复公式；规范三角形角名和联立式左对齐；把图甲/图乙、图1/图2、带数字或罗马分图号的图注保留在原图裁片中，同时恢复正文中被旧规则误删的图号引用。没有按样本 ID 写死，也没有重画或修补题图像素。

全量 298/298 生成成功，状态仍为 234 `REVIEW`、64 `REJECT`；全部宽 800 px，高度 133–1594 px。298/298 使用 Edge/KaTeX，0 降级、0 KaTeX 错误、0 可见 LaTeX 源码。版面审计最低内部行填充 `0.5882`、P05 `0.8229`、中位数 `0.9490`，低于 50% 的候选为 0。171 道题的 232 个视觉资产逐个与当前测试源坐标复裁并核对 RGB 像素哈希，232/232 一致，0 缺失、0 重绘。

相对 v139 共有 21 张预期 PNG 变化：16 张是本轮被修复的旧 FAIL，另 5 张是自动 REJECT 中补全图注或分图标签像素的安全变化；其余 277 张文件 SHA-256 完全一致。行为比较器的两个长度漂移告警分别来自补回答题横线和删除卷尾页脚，已保留在审计记录中。为防止反复迭代时裁框继续向邻图扩张，又用 v150 作为输入完整重跑出 v151；v151 与 v150 的 298/298 PNG SHA-256 完全相同，文本最小/平均相似度均为 1.0，行为回归为 0。

网页会继承 201 个 PASS 和 25 个仍有效 FAIL，只重新开放 16 个实际修复的可验收成品；另有 56 个无人工结论的自动 REJECT 只供失败审计。自动测试 281 项全部通过。机器记录位于 `artifacts/runs/layout_full298_v151_idempotent_candidate_gated/maturity_summary.json`，排版和行为审计位于同一目录。当前版本仍是纯重排候选，平均渲染 `1.240s`、P95 `1.293s`、最大 `4.388s`，这些数字不含 OCR，不能用于声称端到端低于 5 秒。正式 OCR Gold、自建真实数据、手写 Mask 和无资源冲突的全量端到端复测仍未完成，因此状态保持 `MATURE_SAFE_CANDIDATE_NOT_RELEASED`。
