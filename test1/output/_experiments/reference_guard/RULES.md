# 本地电子原图匹配：二阶段安全门槛

## 发现

### photo_070_0149

当前第一阶段的最高候选是错误的 `screenshot_015_0032.png`：

- 71 个 Lowe 过滤匹配、43 个 RANSAC 内点、内点率 60.6%，会通过当前 `>=14` 与 `>=55%` 条件。
- 但参考图内点凸包面积比为 0，参考图只覆盖 1 个象限和 1/16 网格。
- 正向重投影 P90 仅 0.15 px，反向重投影 P90 却为 544.9 px。
- 原因是大量照片特征塌缩匹配到参考图的同一个/同一小片区域；单向 RANSAC 误把退化单应性当成高质量匹配。
- 正确 `screenshot_070_0149.png` 因蓝色污渍只得到少量可靠特征，不应替换；安全结果应回退到照片清理。

### photo_201_0773

- 正确 `screenshot_201_0773.png`：40 个过滤匹配、26 内点、内点率 65%；查询/参考凸包面积比 8.70%/30.98%，均覆盖 4 个象限；4×4 网格覆盖 37.5%/56.25%；正向/反向 P90 为 3.15/2.31 px，几何一致。
- `screenshot_224_1050.png` 具有完全相同的 26 内点和几何指标。人工查看确认两张截图的题干和五幅绳结图完全相同，但 metadata 答案不同。因此视觉特征无法区分二者，候选不唯一时必须拒绝替换。

## 可直接实现的二阶段规则

第一阶段仍保留当前候选生成：SIFT、Lowe 0.72、RANSAC 5 px、`inliers >= 14`、`inlier_ratio >= 0.55`。

对第一阶段候选增加以下全部条件：

```text
query_inlier_hull_area / query_image_area >= 0.012
reference_inlier_hull_area / reference_image_area >= 0.012
query_inliers occupy >= 2 of 4 quadrants
reference_inliers occupy >= 2 of 4 quadrants
query_inliers occupy >= 2 cells of a 4x4 grid       # coverage >= 0.125
reference_inliers occupy >= 2 cells of a 4x4 grid   # coverage >= 0.125
forward reprojection error P90 <= 4.5 px
inverse reprojection error P90 <= 6.0 px
```

然后仅在通过二阶段的候选中比较第一、第二名：

```text
best_inliers >= second_valid_inliers + 6
and best_inliers >= floor(second_valid_inliers * 1.35)
```

第二名与第一名同分时无条件拒绝，即使两张图像视觉上近乎重复。若业务需要处理重复截图，应先离线按图像哈希/感知哈希聚类，并要求同一聚类的题目 ID 和答案元数据一致；不能在在线匹配阶段任选一张。

## 验证

- 重点错误 `photo_070_0149` 的所有第一阶段高分错候选均被二阶段拒绝。
- `photo_201_0773` 的正确几何匹配通过，但因 `201/224` 并列而最终拒绝替换，避免答案不一致。
- 另测 14 个困难样本：除 `photo_008_0016` 外，其余 13 个正确配对都通过二阶段；`008` 本身的参考支持退化，安全回退，不会误替换。
- 所有测试中通过二阶段的错误候选，仍被最终领先幅度门槛拦截；没有发现会被最终接受的错误替换。

## 不应采用的规则

不要用“整张照片四角投影必须落入电子截图”或固定全图投影面积比。照片通常含大量页边/屏幕边，而电子截图紧裁题目，正确匹配的照片四角经常落在参考图外。门槛应只基于实际内点支持区和双向误差。
