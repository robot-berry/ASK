# DeepSeek 六图流程补充测试报告

测试日期：2026年10月15日。新增预算上限 7 元，独立于上轮 10 元预算。仅比较解析呈现与时延，不做标准答案对比。

## 型号与范围

新增模型：`deepseek-ai/DeepSeek-V4-Pro`，通过硅基流动 API 调用。账号模型列表已确认该 ID 可用。采用 Non-think 配置，enable_thinking=false，max_tokens=4096；保留原工程提示词、JSON Schema、90 秒预算、学弱层后 15 秒预算及原图输入策略。未启用提供商修复重试。

模型价格按输入 12 元、输出 24 元／百万 tokens 计算，未计缓存折扣，已知用量增加 10% 余量，未知用量保留请求前上限预留。价格核实来源：[硅基流动价格页](https://www.siliconflow.cn/pricing)。

测试使用上轮固定的数学 10 题、物理 10 题。每次从原题执行真实 OCR、模型调用、校验、修正与六图发布；不使用缓存 OCR 替代真实预处理。预热 OCR 的本地调用不计入模型时延。与 10 月 12 日保存的基线比较，未新增基线付费调用，日期及网络负载差异可能影响时延。

## 覆盖与时延

| 模型 | 评估题数 | 实际调用 | 六图文件 | 完整响应 | 严格校验通过 | 首层解析均秒 | 六图齐全均秒 | 最终完成均秒 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek-V4-Pro | 20 | 11 | 11 | 9 | 8/9 | 46.836 | 47.594 | 47.63 |
| qwen_max | 18 | 18 | 18 | 15 | 11/15 | 5.211 | 18.215 | 21.34 |
| glm_flash | 17 | 17 | 17 | 17 | 6/17 | 16.485 | 16.874 | 16.89 |
| kimi | 17 | 17 | 17 | 17 | 6/17 | 40.201 | 40.546 | 40.559 |
| qwen_27b | 18 | 18 | 18 | 15 | 8/15 | 44.698 | 45.013 | 45.026 |

均值限完整响应，不含通用兜底、提前截流复用和不兼容调用。样本集不同，不应直接将非配对均值之差解释为模型性能差距。六图文件存在及尺寸合规不等于解析正确或信息完整。首层解析为第 2 张图，非本地 OCR 的第 1 张图。后端发布计时不含浏览器显示。

## 同题配对

| 条件 | 题数 | 首层时延差均秒 | 六图齐全时延差均秒 | 最终完成时延差均秒 |
|---|---:|---:|---:|---:|
| 双方完整响应 | 9 | 41.697 | 29.473 | 26.724 |
| 另要求 OCR 文本及附图数量相同 | 5 | 29.343 | 19.039 | 16.453 |

差值为 DeepSeek 减历史基线。附图数量相同不代表图像字节完全相同，未据此宣称统计等效。

## 费用

本轮保守计费与预留：0.657360 / 7 元；已报告用量估算：0.597600 元；待返回调用：0 次。最终实际扣费以供应商账单为准。

## 逐题记录

| 题目 | 状态 | 是否调用 | 附图数 | 图数 | 完整响应 | 图片不兼容 | 首层解析秒 | 六图齐全秒 |
|---|---|---|---:|---:|---|---|---:|---:|
| TZIP-MATH-T01-Q01 | completed | True | 0 | 6 | True | False | 27.9999 | 28.6307 |
| TZIP-MATH-T01-Q02 | completed | True | 0 | 6 | True | False | 57.0974 | 57.8399 |
| TZIP-MATH-T01-Q03 | completed | True | 0 | 6 | True | False | 75.5436 | 76.3145 |
| TZIP-MATH-T01-Q05 | completed | True | 0 | 6 | True | False | 27.5584 | 28.1905 |
| TZIP-MATH-T01-Q07 | completed | True | 1 | 6 | False | True | 5.6001 | 6.0329 |
| TZIP-MATH-T01-Q09 | completed | True | 0 | 6 | True | False | 66.8599 | 67.4576 |
| TZIP-MATH-T01-Q10 | completed | True | 0 | 6 | True | False | 29.2932 | 29.9869 |
| TZIP-MATH-T01-Q12 | completed | True | 0 | 6 | True | False | 62.5037 | 63.4842 |
| TZIP-MATH-T01-Q13 | completed | True | 0 | 6 | True | False | 47.3537 | 48.3985 |
| TZIP-MATH-T01-Q15 | failed | False | 2 | 1 | False | True | None | None |
| TZIP-PHYS-T01-Q01 | completed | True | 0 | 6 | False | False | 84.6059 | 85.0908 |
| TZIP-PHYS-T01-Q03 | failed | False | 2 | 1 | False | True | None | None |
| TZIP-PHYS-T01-Q05 | failed | False | 1 | 1 | False | True | None | None |
| TZIP-PHYS-T01-Q07 | completed | True | 0 | 6 | True | False | 27.3159 | 28.0389 |
| TZIP-PHYS-T01-Q08 | failed | False | 3 | 1 | False | True | None | None |
| TZIP-PHYS-T01-Q09 | failed | False | 1 | 1 | False | True | None | None |
| TZIP-PHYS-T01-Q10 | failed | False | 2 | 1 | False | True | None | None |
| TZIP-PHYS-T01-Q11 | failed | False | 1 | 1 | False | True | None | None |
| TZIP-PHYS-T01-Q14 | failed | False | 1 | 1 | False | True | None | None |
| TZIP-PHYS-T01-Q15 | failed | False | 1 | 1 | False | True | None | None |

若图片请求被服务端明确拒绝，后续相同能力需求的请求在发送前跳过，仍保留 OCR 和不兼容状态记录。没有删除图像以伪造同条件测试成功。

原始资料：results_math.json、results_physics.json、ledger.json、manifest.json；每题 jobs 目录内保留六图和 provider 审计；图片文件核查结果见 summary.json。

结论：本次硅基流动 DeepSeek-V4-Pro 不适合作为当前六图工程的直接等效替换。20 题中，9 题获得完整响应，1 题达到输出上限后使用兜底，10 题因图片输入不兼容而无法按原条件完成。9 题完整响应的六图齐全平均为 47.59 秒；其中 5 个 OCR 文本和附图数量一致的配对，六图齐全比历史 Qwen3.8-Max 平均慢 19.04 秒。预算有剩余，但固定样本评估已经结束，继续重复不支持的图片请求不会改善结论。本次不新增付费调用。


图片能力已实测确认：数学 Q07 带 1 张图请求时，服务端返回 HTTP 400、错误码 20041，明确说明模型不是视觉语言模型，要求纯文本输入；后续 9 个需要附图的请求在发送前跳过。首次拒绝也触发了工程兜底，因此原始 fallback 标志共 2 例，其中 1 例与图片不兼容重叠，不能重复算成额外失败。
物理 Q01 返回 finish_reason=length，输出 4096 tokens 后被截断，未形成完整五层解析。其六图由本地兜底生成，不能计作有效模型输出。
人工抽查数学 Q01、Q13 和物理 Q07 的第六图：部分公式呈现为未排版的上标、下标或不完整的线性表达；数学 Q13 有表达式在根号附近结束的现象；物理 Q07 明确提示无法从 OCR 获取完整示意图结构。该题原流程未附图，因此接口可以返回文本，但完整响应不等于题目信息完整或解析可直接使用。报告没有将格式校验通过率当作答案正确率。

