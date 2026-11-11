# photo_201_0773 透视矫正实验

## 判断

该图片的问题不是单一 deskew 旋转，而是明显的平面透视变形。只旋转会使局部文字稍微水平，但题目左右尺度、上下尺度和剪切仍不一致。

## 量化结果

- SIFT Lowe 过滤匹配：111。
- Homography：33 内点，内点重投影误差中位数 0.81 px、P90 2.04 px。
- Full affine：16 内点；在 homography 的同一组内点上误差 P90 为 81.61 px。
- Rotation + uniform scale：10 内点；在 homography 同组内点上误差 P90 为 166.18 px。
- Affine 分解显示约 4.03° 旋转、约 -7.47° 剪切、1.39 倍各向异性；但 full affine 仍无法解释跨页面变化。
- Homography 的 projective terms 为 `(7.94e-4, 5.88e-5)`，不是可忽略的纯仿射项。

因此本例应使用完整 homography；affine/deskew 均不足。实验矫正图与干净 screenshot 的文字和五个图形已基本重合。

## 安全使用规则

透视矫正只能在以下证据全部成立时启用：

```text
homography inliers >= 24
homography inlier ratio >= 0.25
query/reference inlier hull ratio >= 0.04 / 0.12
query/reference inliers cover >= 3 quadrants
query/reference 4x4 grid coverage >= 0.25
forward reprojection P90 <= 3.5 px
inverse reprojection P90 <= 4.0 px
```

再判断是否真的需要透视：使用 homography 内点作为同一评价集，拟合 similarity 和 full affine：

```text
if similarity P90 <= 4 px:
    仅旋转/等比缩放，不做透视
elif full_affine P90 <= 4 px:
    使用 affine 去剪切
elif homography P90 <= 3.5 px
     and full_affine P90 >= max(8 px, homography P90 * 2.5):
    使用完整 homography
else:
    不自动矫正
```

本例：homography P90 2.04 px，而 full affine 在同一内点上的 P90 81.61 px，明确进入完整透视分支。

## 输出边界保护

- 变换后只裁剪“参考侧内点凸包 + 8% 安全边距”，不能按检测文本紧裁。
- 必须将原题的全部 homography 内点投影到输出画布内；任一点距边界小于短边 2% 时扩大画布。
- 输出区域面积若小于原图内点凸包外接矩形的 85%，拒绝，以免裁题。
- 没有可信电子参考时，不建议从文字基线单独估计透视；只能做保守旋转并保留原图。

## 注意

此处 clean screenshot 只用于确定真实几何。若实际匹配出现多个视觉重复参考（本题的 201/224），参考替换仍应按独立的“并列拒绝”规则处理；几何矫正可使用一致的 homography，但不能据此任选带不同答案元数据的电子版本。
