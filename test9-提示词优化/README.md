# Test9 提示词优化实验版

从 `../test6` 复制，原工程不变。先阅读 [优化与对比说明](PROMPT_OPTIMIZATION.md)。

新增：网页支持初一至高三的12个年级学期选择，详见 [年级学期提示词](SEMESTER_PROMPTS.md)。

本地启动：`powershell -ExecutionPolicy Bypass -File .\start_local_math.ps1`。
默认数学9120、物理9210、门户9130、公式9240；移动门户9131。
沿用原工程的共享OCR环境和工作区密钥读取方式，未复制私钥、虚拟环境、运行记录或提示词数据库。

以下为继承的Test6参考文档，其中旧端口、发布路径及部署脚本尚未迁移，不能直接用于Test9部署。

# Test6 流式物理数学搜题机

Test6 的本地开发、发布构建和 ECS 生产环境统一固定为 **CPython 3.14.6**。
项目根目录的 `.python-version` 供 `uv` 和兼容工具自动选择版本；服务器
安装脚本会安装并校验精确的 3.14.6，不使用系统默认 Python 运行业务。

Test6 从 Test4 派生，输出一张本地 OCR 原题图和五张分层解析图，共六张 800 px 灰度 PNG。第一张由完整 Test5 OCR 在本地识别、重排并保留必要插图；不再发布无 OCR 的快速预览图，模型也不参与第一张图。模型返回经过答案脱敏约束的结构化五档自然语言，本地负责校验、统一排版和六图渲染。历史任务缺少分档内容时，仍可用公共骨架在本地补齐。

完整目录、数据流和关键文件作用见 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)。

约 100 台设备接入一台服务器的公平调度、并发参数和压测门槛见
[MULTI_DEVICE_DEPLOYMENT.md](MULTI_DEVICE_DEPLOYMENT.md)。

若目标机器为 64 vCPU、128 GiB 的 `ecs.c9a.16xlarge`，直接使用的实例和环境变量清单见
[ECS_C9A_16XLARGE_CONFIGURATION.md](ECS_C9A_16XLARGE_CONFIGURATION.md)。

题目分类存储的 JSON Schema、示例记录和字段说明见
[storage_schema/README.md](storage_schema/README.md)。

## 公共输出治理

内部 OCR 和模型审计继续保留 LaTeX，以保存分式、根号和上下标结构；网页、硬件接口、OCR 复核页及最终图片统一经过 `presentation_guard.py` 或富公式渲染器。数学后端、物理后端和门户会逐层检查公共分析文字，阻止 `\frac`、`$...$`、代码围栏等内部标记直接显示。

## 流式行为

1. 上传题图经过 Test5 的 EXIF 转正、自动旋转、题面/透视校正、纠偏和光照归一化。
2. Test5 执行 RapidOCR、题目选择和白底重排；数学/物理可选用常驻的本地版式与公式 OCR（PP-DocLayoutV2 + PP-FormulaNet）恢复分式、指数和上下标。清晰普通题只走快速识别；检测到题干缺失、公式密集、选项不全或低置信度时才调用结构化 OCR，仍有结构缺口时才做整页补救。语义恢复会清理重复题块、修复可观察的选项标签，并对表格、几何图、电路图和高密度公式保留来源像素；服务不可用或超时时自动保留 RapidOCR。
3. 完整可读的 OCR 重排题图转换为 800 px、灰度 `L`、9 cm DPI 的 `01_question_knowledge.png`，原子写入任务目录并立即出现在任务 API 的 `images` 数组中。
4. 模型请求使用紧凑合同返回学霸、中上、中等、中下、学渣五档自然语言：简单题使用 Flash，复杂证明/几何/多问/多图题才使用 Max；OCR 清晰且无图时只发送主图，低 OCR 或带图时才附加增强图和裁片。模型不会生成、改写或覆盖第一张图。
5. 服务器校验五档内容并在本地确定性渲染。默认不给模型输出设置 token 上限；达到调用时限时优先复用已经完整到达的分档内容，缺失部分才由本地安全回退补齐，不执行额外模型修复。
6. 门户每 600 ms 获取一次状态，并保留已经显示的图片，因此页面按 `1/6 → … → 6/6` 渐进更新。

第一张图完全由本地 OCR 生成；模型传输失败或模型修复不会撤回它，最终完成态仍必须通过完整六图合同。

## 固定六图合同

1. `01_question_knowledge.png`：Test5 本地 OCR 识别、重排并保留必要插图的原题图
2. `02_guidance_elite.png`：学霸，只问 1～3 个问题，不给知识点、公式或方向
3. `03_guidance_upper.png`：中上，只点知识域，不展开应用
4. `04_guidance_middle.png`：中等，只点破每小问的题眼，不给第二步
5. `05_guidance_lower.png`：中下，给编号解题骨架，关键计算或论证留白
6. `06_guidance_novice.png`：学渣，给公式卡和兜底详解，最终结果仍由学生填写

图片继续保持 PNG、宽 800 px、灰度 `L`、显示宽度 9 cm、高度随内容自动变化。

## 服务结构

```text
统一门户 8230
   ├── 8210 物理
   ├── 8220 数学
   └── 8240 本地版式/公式 OCR（仅回环，数学物理共享）
```

三个端口默认只应绑定回环地址，由 nginx 对外提供 80/443。Test6 优先读取 `TEST6_*` 环境变量，并兼容读取旧的 `TEST4_*` 名称，便于从 Test4 平滑迁移。示例配置见 `api_keys.env.example`。

## 本地启动

本地脚本保留旧文件名以兼容现有使用方式，但现在会同时启动物理、数学、统一门户和常驻公式 OCR：

```powershell
.\start_local_math.ps1 -PhysicsPort 8210 -MathPort 8220 -PortalPort 8230 -FormulaPort 8240
```

`start_local_mobile.ps1` 只负责额外开启局域网页面，并会同时检查物理和数学后端；任一学科不可用时不再进入“单学科可用”的降级状态。

需要配置 `DASHSCOPE_API_KEY` 和至少 24 位的 `TEST6_ACCESS_TOKEN`。若沿用原脚本设置的 `TEST4_ACCESS_TOKEN`，兼容回退仍然有效。

## 时延分析与验收

300 题纯本地串行 OCR 平衡优化报告见 [新版 OCR 报告](benchmark/ocr_accuracy_300_balanced_final_20260903/LATEST_OCR_REPORT.md)，逐题原图与新旧 OCR 可在本地门户 `/ocr-audit` 查看。平均本地 OCR/预处理时延由 9.438 秒降至 5.268 秒（下降 44.2%），P95 由 18.847 秒降至 12.378 秒；自动纯文字结构通过 298/300，文字与必要原图组合的语义包通过 300/300。自动门槛只用于筛查，不能替代人工语义完整率。

详细设计见 [LATENCY_OPTIMIZATION.md](LATENCY_OPTIMIZATION.md)。公共骨架、模型路由和 16 秒预算的 15 题真实串行结果见 [最新低时延报告](benchmark/P_DATA_math_serial_compact_routing_20260827/LATENCY_REPORT.md)；上一轮本地修复优先对比见 [旧模型时延报告](benchmark/P_DATA_math_serial_qwen38max_localrepair_20260826/LATENCY_REPORT.md)，双阶段首图结果见 [快速首图报告](benchmark/P_DATA_math_serial_fastfirst_20260827/LATENCY_REPORT.md)。硬件接口保持 `v1` 兼容，见 [HARDWARE_API.md](HARDWARE_API.md)。
