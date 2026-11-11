# Test6 本地工程结构说明

本文从“服务如何协作、一次题目如何流转、目录和关键文件分别做什么”三个角度说明 `picture/test6`。工程的目标是：上传一张数学或物理题图，尽快先显示本地生成的原题图，再逐步生成五档解析图。

## 1. 整体架构

本地运行时有四个服务：

```text
浏览器 / 硬件设备
        │
        ▼
统一门户 8230（鉴权、转发、状态、结果下载、OCR复核页）
   ├── 数学服务 8220（排队、数学处理流水线、六图发布）
   ├── 物理服务 8210（排队、物理处理流水线、六图发布）
   └── 结构化OCR 8240（仅本机回环，数学和物理共享）
                    │
                    └── PP-DocLayoutV2 + PP-FormulaNet

数学/物理流水线 ──按需──> DashScope 多模态模型
```

- 8230 是用户和硬件应访问的统一入口。
- 8210、8220、8240 是内部服务，默认只监听 `127.0.0.1`。
- 第一张题图完全由本地生成，不等待大模型。
- 五档解析使用模型自然语言，经本地结构校验、答案脱敏和图片渲染后输出。

## 2. 一道题的处理流程

1. 门户接收图片，验证大小、格式、鉴权和幂等键。
2. 门户根据用户选择把任务转发到数学或物理服务。
3. 学科服务创建任务目录，按设备身份轮转排队，再交给可配置的多条独立流水线并发处理。
4. Test5 预处理完成 EXIF 转正、旋转、透视、纠偏和光照归一化。
5. RapidOCR 识别普通文字；结构化 OCR 补充公式、分式、根号、上下标和版式关系，完成后发布第一张可读 OCR 原题图。
6. 融合器只在候选结果通过异常检查且确实增加语义信息时采用结构化 OCR；否则保留普通 OCR。
7. 如果题目包含必须保留的图形、几何图、电路图、坐标图或表格，保守裁剪并与 OCR 文字一起交给模型。
8. 模型返回结构化的五档自然语言；本地执行格式、完整性和答案泄漏检查。
9. 本地渲染为固定六张 800 px 灰度 PNG，并按生成顺序持续发布。
10. 门户每隔约 600 ms 获取状态，因此网页按 `1/6 → 6/6` 渐进显示。

## 3. 输出边界与公式治理

工程把数据分为两类：

### 内部机器数据

内部 OCR、模型响应、诊断 JSON 和任务审计文件允许保留 LaTeX，例如 `\frac{1}{2}`。这是为了避免把二维公式过早压平成不可靠的普通文字。

### 对外展示数据

以下内容不得直接包含 LaTeX/Markdown 控制码：

- 数学、物理后端的公共任务状态；
- 统一门户返回给浏览器或硬件的 `analysis_audit`；
- 网页“模型分析文字（检查用）”；
- OCR 复核网页和复核图片；
- 六张最终设备图片。

公共模块 `presentation_guard.py` 负责将分式、根式、上下标、希腊字母和关系符转换为可读形式，并递归检查公共数据中是否仍有 `\frac`、`$...$`、代码围栏等内部标记。数学后端、物理后端和门户都执行这项检查；即使某一层以后被误改，下一层仍会阻止原始标记直接输出。

最终图片使用更完整的富公式渲染器，能够真正绘制分数线、根号和上下标；`presentation_guard.py` 主要保护网页文字、接口文字和轻量复核图片。

## 4. 顶层目录和文件

```text
test6/
├── math/                       数学服务、流水线、渲染和测试
├── physics/                    物理服务、流水线、渲染和测试
├── portal/                     统一网页/硬件入口
├── benchmark/                  OCR、时延、正确率和人工复核工具及报告
├── deploy/                     nginx、systemd 和服务器安装脚本
├── tests/                      跨服务合同测试
├── release/                    本地构建的发布压缩包（运行产物）
├── .local/                     PID、日志、临时测试结果（运行产物）
├── presentation_guard.py       全工程公共文字出口治理
├── multi_device_scheduler.py   单服务器多设备公平队列和并发消费者
├── benchmark_multi_device.py   多设备真实并发提交与分阶段时延报告
├── provision_device_tokens.py  批量生成设备凭据和服务器端令牌哈希
├── formula_ocr_service.py      常驻结构化公式/版式 OCR 服务
├── structured_ocr_client.py    数学/物理调用结构化 OCR 的安全回环客户端
├── start_local_math.ps1        启动四个本地服务
├── stop_local_math.ps1         按 PID、端口和进程身份安全停止四个服务
├── start_local_mobile.ps1      在局域网额外开放统一入口
├── stop_local_mobile.ps1       关闭局域网入口
├── build_release.py/.ps1       生成发布包和文件完整性清单
├── verify_release.py           验证发布包路径和 SHA-256 完整性
├── requirements*.txt           运行/测试依赖声明
├── api_keys.env.example        环境变量示例，不包含真实密钥
├── README.md                   工程入口说明
├── HARDWARE_API.md             硬件调用合同
├── MULTI_DEVICE_DEPLOYMENT.md  100台设备单服务器并发配置和压测门槛
├── LATENCY_OPTIMIZATION.md      时延组成和优化记录
└── VERSION                     发布版本号
```

## 5. 数学与物理目录

`math` 和 `physics` 的结构基本镜像，使两门学科共享相同的服务合同，同时保留各自的提示词、校验规则和渲染细节。

### `math/source` 与 `physics/source` 的核心文件

| 文件 | 作用 |
|---|---|
| `search_machine_app.py` | FastAPI 学科服务入口；管理按设备公平轮转的多消费者队列、健康检查、上传、轮询和图片下载；只返回经过裁剪的公共任务字段。 |
| `search_machine_pipeline.py` | 一道题的总调度器；连接预处理、OCR、模型调用、校验、五档内容和六图发布。 |
| `test5_preprocess.py` | 当前主预处理/OCR实现；负责方向、透视、纠偏、光照、文字识别、题目区域、配图裁剪、文字重排及语义恢复。 |
| `*_guidance_pilot.py` | 学科提示词、模型结构化输出解析、校验与历史评测逻辑；也是大量学科规则的来源。 |
| `*_guidance_renderer.py` | 把题目和五档解析排版成 800 px 灰度图片，调用富公式排版能力。 |
| `result_image_store.py` | 通用富文本/公式绘制、图片元数据、原子发布和结果图库能力。 |
| `device_result_renderer.py` | 检查并生成固定六图合同，确保宽度、颜色模式、DPI、文件名和顺序正确。 |
| `five_tier_guidance.py` | 选择模型生成的学霸、中上、中等、中下、学渣五档内容；兼容旧任务的本地投影。 |
| `compact_guidance.py` | 定义紧凑模型返回合同和五档字段映射，降低模型输出量与等待时间。 |
| `streaming_images.py` | 识别流式 JSON 已完成前缀，支持第一张图和后续结果按顺序发布。 |
| `*_guidance_schema.json`、`compact_guidance_schema.json`、`schema.json` | 模型输出及内部结构的 JSON Schema。 |
| `数学题解析*.md`、`物理题解析*.md` | 完整版与紧凑版提示词。 |
| `search_machine_static/index.html` | 直接访问学科端口时使用的简易测试页；正式使用建议走 8230 门户。 |
| `preprocessor/` | 较早的通用预处理适配层及兼容接口；当前主路径以 `test5_preprocess.py` 为准。 |
| `models/` | 本地文字检测、方向分类等 ONNX 模型。 |
| `test_*.py` | 学科规则、OCR、裁图、渲染、接口和历史缺陷回归测试。 |

### `math/device_runs` 与 `physics/device_runs`

每个上传任务会形成独立目录，通常包含：

- 原始上传图片；
- 预处理和裁图片段；
- 六张最终 PNG；
- `job.json` 状态和内部模型/OCR审计数据。

这些目录是运行数据，不属于发布源代码。

## 6. 门户目录

| 文件/目录 | 作用 |
|---|---|
| `portal/source/test4_portal_app.py` | 统一网关。负责登录会话、设备令牌/SN、上传限制、设备级幂等提交、数学/物理转发、结果合同复核、公共字段清洗、并发时延字段和 OCR 报告下载。文件名保留 `test4` 是为了兼容旧启动方式，实际服务版本为 Test6。 |
| `portal/source/static/index.html` | 当前 8230 测试网页，显示学科状态、上传、流式六图、时延和经过治理的模型分析文字。 |
| `portal/source/.state/` | 门户幂等记录和会话状态数据库，属于运行数据。 |

门户不会把后端任务目录、绝对路径、私有错误详情或内部 OCR 原文直接转发给客户端。

## 7. 结构化 OCR 文件

| 文件 | 作用 |
|---|---|
| `formula_ocr_service.py` | 常驻加载 PP-DocLayoutV2 与 PP-FormulaNet；按版面顺序组合普通文字和公式；只允许本机调用。 |
| `structured_ocr_client.py` | 把增强题图发送到 8240；限制为回环地址，设置超时；失败时让主流程自动退回 RapidOCR。 |
| `presentation_guard.py` | 内外数据边界。内部保留 LaTeX，对外转成可读文字并检测遗漏的源码标记。 |

结构化 OCR 不是大模型答题服务，不会推导答案；它只恢复题面文字和公式结构。当前采用风险分流：普通高置信题只做快速 OCR；题干、选项、公式或置信度出现可观察风险时才请求结构化 OCR；候选结果必须保留已有 A-D 选项和题干上下文，仍有缺口才允许整页补救。

## 8. `benchmark` 目录

这里保存测试工具及历史报告，不参与在线请求主路径。

| 文件 | 作用 |
|---|---|
| `ocr_accuracy_300.py` | 逐题串行运行 300 图 OCR，记录时延、置信度、结构指标并生成复核图。 |
| `recompute_semantic_metrics.py` | 不重新调用 OCR，用当前生产规则重算已保存结果的文字结构与语义包指标。 |
| `build_full_ocr_audit_queue.py` | 根据一次 OCR 结果生成 300 题人工复核队列。 |
| `build_full_semantic_audit_report.py` | 汇总人工判定，形成完整率报告和可筛选网页。 |
| `build_structured_ocr_review.py` | 生成新旧 OCR 对照网页；网页只嵌入经过展示治理的文字。 |
| `build_semantic_incomplete_view.py` | 单独汇总被判定语义不完整的题目。 |
| `benchmark_local_formula_ocr.py` | 评测结构化公式 OCR。 |
| `retest_semantic_sample.py` | 对固定人工样本重新运行当前 OCR 并生成对照页。 |
| `apply_full_audit_decisions.py` | 应用人工审核决定。 |
| `verify_semantic_evidence_coverage.py` | 检查每题是否保留完整原始像素证据。 |
| 各日期结果目录 | 保存当次 CSV、JSON、Markdown、HTML 和图片；属于测试证据，不是在线服务代码。 |

原始 `results.jsonl`、`details.csv` 等机器审计文件可以保留 LaTeX；直接面向人的 HTML 和 PNG 必须经过公共展示治理。

## 9. `deploy`、`release` 与运行脚本

- `deploy/nginx/`：HTTP/HTTPS 反向代理模板，对外只暴露统一门户。
- `deploy/systemd/`：Linux 上数学、物理、门户和清理任务的服务定义。
- `deploy/scripts/`：安装、预检查、健康检查和历史任务清理。
- `build_release.py`：排除密钥、日志、缓存、任务结果和虚拟环境，再打包源代码。
- `RELEASE_MANIFEST.sha256`：发布文件的内容哈希清单。
- `.local/local_math_processes.json`：本地四服务的 PID、端口、程序路径和启动时间，用于防止误停其他进程。
- `.local/logs/`：各服务标准输出和错误日志。

## 10. 配置与安全边界

- 实际密钥位于工作区根目录的 `api_keys.env`，不应复制到报告、发布包或聊天内容。
- `api_keys.env.example` 只说明变量名和格式。
- 8240 只允许回环访问；结构化 OCR 客户端拒绝远程主机地址。
- 门户对浏览器使用会话 Cookie，对硬件使用 Bearer Token。
- 公共任务接口只提供六图元数据、公开时延和经过清洗的分析文字。
- 内部异常的路径和堆栈只写服务器日志，对用户返回友好错误。

## 11. 修改功能时通常从哪里入手

| 需求 | 主要文件 |
|---|---|
| OCR文字、方向、透视、题目区域 | 两学科的 `test5_preprocess.py` |
| 几何图、电路图、图表裁剪 | `test5_preprocess.py` 中的 illustration/figure 检测与裁剪逻辑 |
| 模型请求内容、提示词、时限和模型路由 | `search_machine_pipeline.py`、`compact_guidance.py`、学科提示词 |
| 五档内容要求和答案脱敏 | `*_guidance_pilot.py`、`five_tier_guidance.py`、对应 schema |
| 最终图片排版和公式显示 | `*_guidance_renderer.py`、`result_image_store.py` |
| 网页、硬件 API、鉴权和公共输出 | `portal/source/test4_portal_app.py`、`portal/source/static/index.html` |
| LaTeX/Markdown 文字泄漏 | `presentation_guard.py` 以及各公共出口的调用点 |
| 首图和总时延 | `search_machine_pipeline.py`、`streaming_images.py`、`LATENCY_OPTIMIZATION.md` |
| 正确率/完整率复测 | `benchmark/ocr_accuracy_300.py` 和人工复核构建脚本 |

## 12. 当前输出治理保证

当前标准路径采用三层保护：

1. 数学或物理服务在构造公共 `analysis_audit` 时转换公式源码并检查残留标记；
2. 统一门户重新转换和检查后端文字，避免单个后端绕过；
3. 网页使用 `textContent` 显示受控文字，最终图片使用本地富公式渲染，不执行模型返回的 HTML。

因此，后续题目只要通过标准服务和报告脚本输出，就不会再把 LaTeX 控制码作为普通文字显示。内部审计文件仍保留原始公式表示，便于定位识别错误和重新渲染。
