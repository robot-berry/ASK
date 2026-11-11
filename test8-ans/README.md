# test8-ans

这是一个 Qwen-Max 数学、物理题五层解析实验工程。输入是上游已经准备好的题面文字，本工程不执行 OCR。它保留模型第一次回复原文，在展示副本上完成最终答案脱敏和五张 800px PNG。严格结构、相关性和 20 秒指标用于评测报告，不再阻止可展示的回复出图。

## 当前能力

- 按数学/物理选择两份原始提示词；
- 单题、普通 JSONL 批量和冻结 testzip 评测；
- 原样保存第一次模型回复和解析视图，另生成只供展示的答案脱敏副本；
- 对 Qwen 重复输出的 `s` 键做无损收集，只修复 JSON 传输结构，不改句子；
- 严格 JSON 结构检查和轻量本地相关性筛查；
- 按题面所求量删除明确终值、最终选项、最终区间、方向、判断结论和证明收尾；
- 模型、工作流、上游 OCR、估算端到端时延分开记录；
- 固定 20 题冒烟通过后才允许 298 题完整评测；
- 只有无法解析、`l1`～`l5` 缺少整层可展示内容，或脱敏后仍有明确答案嫌疑时才阻止出图；
- 句数、字数、字段细节、相关性、脱敏置信度和 20 秒只记录为质量告警，PNG 落盘时间仍计入时延。

## 初始化

要求 Python 3.11 或更高版本：

```powershell
Copy-Item .env.example .env
```

在 `.env` 中填写 `DASHSCOPE_API_KEY`。如果本工程没有 `.env`，也会兼容读取工作区根目录已有的 `api_keys.env`，且不会把密钥写入结果。先运行不联网检查：

```powershell
python -m test8_ans doctor
```

## 单题

离线预览请求，不消耗模型额度：

```powershell
python -m test8_ans preview --id demo-1 --subject math --stage senior --question "解方程：2(x-1)=6。"
```

调用一次模型：

```powershell
python -m test8_ans run --id demo-1 --subject math --stage senior --question "解方程：2(x-1)=6。"
```

默认不自动修复回答。仅在非基准调试中需要第二次修复调用时，显式添加 `--repair`。

## testzip 分阶段评测

实际冻结数据集目录是：

`D:\robot\Documents\ASK\datasets\testzip_math_physics_300_v1`

默认读取 test7-OCR 已经准备好的 298 题 `question_document.json` 作为题面文字来源，但不会在本工程中运行 OCR，也不会读取或发送答案 PDF。数据集的学段元数据为 unknown，因此以下示例中的 `senior` 只是操作员覆盖值，正式运行前需要确认。

先离线核对单题输入：

```powershell
python -m test8_ans benchmark --phase probe --stage senior --dry-run
```

配置密钥后运行单题真实探针：

```powershell
python -m test8_ans benchmark --phase probe --stage senior
```

探针稳定后运行固定 20 题：

```powershell
python -m test8_ans benchmark --phase smoke --stage senior
```

运行按数学/物理及 Test 组均衡抽取的 50 题样本：

```powershell
python -m test8_ans benchmark --phase sample --stage senior --limit 50
```

每批会额外生成 `comparison_artifacts`：原题图片、50 张模型原始回答截图、所有可展示题目的五层解析图、逐题分阶段时延 CSV 和三栏对比网页。审核页会明确区分“出图状态”和“严格评测告警”。更新本地校验或渲染规则后，可直接重放已经保存的模型回答，不会再次调用模型：

审核页支持逐题标记通过/有问题、问题分类、严重程度和文字备注。为了让反馈真正进入迭代工作流，建议用本地审核服务打开页面：

```powershell
python -m test8_ans review --batch-dir "runs\batches\..."
```

然后访问命令输出的 `http://127.0.0.1:<自动选择的端口>/overview.html`。每次点击“保存本题反馈”都会同时更新批次目录中的 `review_feedback.json`、`review_feedback.csv` 和 `review_feedback_summary.json`。仍可手动导出/导入 JSON。直接以 `file://` 打开时只能使用浏览器临时存储，不建议作为正式审核入口。

```powershell
python -m test8_ans reprocess-benchmark --benchmark-summary "runs\batches\...\benchmark_summary.json"
```

只有 smoke 汇总的晋级门槛通过，才能启动完整 298 题。把实际汇总路径传入：

```powershell
python -m test8_ans benchmark --phase full --stage senior --promotion-summary "runs\batches\...\benchmark_summary.json"
```

默认模型晋级要求仍是结构、相关性、20 秒时延通过率以及综合通过率分别达到 95%，答案脱敏置信度必须 100% 通过。这是“是否晋级模型”的严格评测，不是“是否生成图片”的前置条件。时延从题面文字进入本工程开始，直到五张 PNG 全部落盘；不包含 OCR，门槛可在 `.env` 中调整。

## 结果文件

每题独立目录中的主要文件：

- `model_output.txt`：第一次模型回复原文，是基准主记录；
- `provider_response.json`：第一次百炼完整响应；
- `answer.json`：成功解析后的原始 JSON 视图，不做脱敏改写；
- `display_answer.json`：删除明确最终答案后的展示副本；
- `answer_redaction.json`：脱敏规则、字段路径、删减字符数和原文哈希，不抄录被删答案；
- `transport_normalization.json`：重复 `s` 键或字符串 `s` 的无损数组化记录；
- `validation.json`：结构、相关性、答案脱敏和时延门禁；
- `metadata.json`：模型、提示词哈希、状态和请求 ID。
- `images_800px/l1.png`～`l5.png`：自动生成的五张脱敏解析图。
- `model_answer_screenshot/model_output.png`：不改写模型原始回答的 800px 对比截图。

审核服务保存的主要文件：

- `comparison_artifacts/review_feedback.json`：逐题原始审核反馈；
- `comparison_artifacts/review_feedback.csv`：便于表格检查的同步版本；
- `comparison_artifacts/review_feedback_summary.json`：已审核数、问题数、严重程度、问题类型频次和逐题备注汇总。

批量目录中的 `benchmark_summary.json` 汇总通过率和 P50/P90/P95/P99。由于该冻结集目前声明人工核验标准答案数量为 0，报告不会输出“正确率”。

## PNG 渲染

安装可选渲染依赖：

```powershell
pip install -e .[render]
```

使用任意已保存的评测汇总：

```powershell
python -m test8_ans render --benchmark-summary "runs\batches\...\benchmark_summary.json" --output-dir "runs\rendered-smoke"
```

渲染器只读取 `display_answer.json`，不会读取 `answer.json`。每题自动生成 `l1.png`～`l5.png`，对外标题依次为“学霸解析、中上解析、中等解析、中下解析、基础解析”；宽度固定 800px，高度随内容增长。公开图会把 LaTeX 控制符转换为 Unicode/可读公式文本，禁止 `$`、反斜杠命令和上下标花括号原样显示；正式设备版仍建议接入 KaTeX 并增加渲染回读。

单题检查可直接渲染一个已经通过展示安全门禁的运行：

```powershell
python -m test8_ans render-run --run-dir "runs\..." --output-dir "runs\...\images_800px"
```

## 测试

```powershell
python -m unittest discover -s tests -v
```

完整的门禁定义、数据边界和待确认指标见 [WORKFLOW.md](WORKFLOW.md)。
已执行的线上探针和当前晋级状态见 [EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md)。
