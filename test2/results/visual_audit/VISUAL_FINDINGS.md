# test2 全量批处理视觉抽检结论

- 批量结果：284/284 成功，失败 0；本次分层抽检 21 张。
- 抽样：兜底 3/3、透视待复核 1/4、较大倾斜 6/30、最慢 6/284、普通 5/252。
- 电子参考匹配分支（抽到的较大倾斜、最慢和普通样本）均保持题目主体完整，方向正确，公式和图形可辨。
- 全批次没有一张实际应用透视变换；4 张进入照片保真分支，其中 3 张走兜底，均存在明显视觉问题。

## 明显异常

| 样本 | 严重度 | 目视结论 | 元数据线索 |
|---|---:|---|---|
| `photo_080_0176.jpg` | 高 | 蓝色污迹被二值化为大块黑色噪声；题干和三角形仍在，但电子版质量不可接受。 | `fallback=1`, `perspective=insufficient-coverage` |
| `photo_191_0706.jpg` | 高 | 输出错误旋转 90°，文字竖置；内容基本完整。 | `rotation=90`, `deskew=not-enough-text`, `fallback=1` |
| `photo_241_1284.jpg` | 高 | 强透视/斜切未纠正，左上保留黑色屏幕边缘；题目可辨但不符合电子版方向。 | `deskew=-8.35/out-of-range`, `perspective=insufficient-coverage`, `fallback=1` |
| `photo_141_0354.jpg` | 中 | 方向和题目完整，但二值化后纸纹、边缘和小斑点噪声很多，明显弱于配对干净截图。 | `perspective=low-confidence`, `cleanup=photographic` |

## 结论

当前主干参考匹配输出表现稳定；风险集中在“未匹配参考 → 照片保真/兜底”分支。建议把 `cleanup_method=photographic`、`fallback_count>0` 或 `perspective_status != not-needed` 作为自动质检触发条件，优先人工复核，不应仅凭任务成功状态放行。
