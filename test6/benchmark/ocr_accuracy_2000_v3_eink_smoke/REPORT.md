# 2题当前本地OCR全量串行统计

- 数据集：`D:\robot\Documents\ASK\datasets\testzip_math_physics_2000_v3_eink`
- 完成：2/2，失败 0题
- 执行方式：单进程逐题串行；未调用答题大模型。
- OCR组成：当前 Test6 RapidOCR + 本机结构化公式 OCR。
- 标准题干文本：0题。
- 说明：无人工标准题干，以下置信度和自动结构通过率不等于真实文字正确率。

## 总体

- OCR运行成功率：100.00%
- 非空文字率：100.00%
- 自动结构通过率：100.00%
- 自动判为结构不完整：0题
- 数据集已标记图片质量问题：0题（需与OCR缺失分开判断）
- 平均字符加权置信度：96.94%
- 结构化OCR采用率：0.00%
- 单题时延均值：2523.153 ms
- 单题时延 P50/P90/P95/P99：2523.153 / 3047.574 / 3113.127 / 3165.569 ms
- 全量墙钟耗时：6.504 秒
- 人工复核抽样：2题，见 `manual_audit_pages/` 和 `manual_audit.csv`。
- 自动不完整复核队列：0题，见 `semantic_review/manual_audit_pages/`。

## 分科

- mathematics：2/2成功，结构通过率 100.00%，平均置信度 96.94%，时延均值/P95 2523.153 / 3113.127 ms

## 分组语义结构通过率

- mathematics/seed_v2：2/2，100.00%

## 自动语义缺陷类型

- `stem_missing`：检测到选项但题干缺失
- `option_sequence_incomplete`：选择题选项序列不完整
- `choice_options_missing`：存在选择题语义但选项缺失
- `subquestion_gap`：小问编号不连续
- `nested_subquestion_incomplete`：出现第(1)问但第(2)问缺失
- `table_row_incomplete`：表格关键数据行不完整
- `formula_gap`：已知高风险公式位置出现缺口
- `symbol_damage`：出现损坏字符或连续问号
- `too_short`：识别文字过短，无法构成完整题意

### 本次计数


## 质量标记

- `structured_ocr_service_not_configured`：2
