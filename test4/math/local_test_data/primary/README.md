# 小学数学本地合成测试样本

这里的 3 张题图是由 `generate_primary_fixtures.py` 在本机确定性生成的干净 PNG，用于补充 Test4 小学阶段的基础覆盖：

- 多步计算与应用题；
- 分数与长度单位；
- 带实际示意图的组合图形面积、周长题。

这些图片是**合成的清晰题面**，不是教材扫描件，也不是手机拍照样本。题面均为小学课程范围，不是选择题或填空题，要求展示列式、计算或推理过程，并且不含答案。

重新生成：

```powershell
python .\generate_primary_fixtures.py
```

只验证清单和图片、保证 0 次网络请求：

```powershell
python ..\..\run_local_dataset_api.py .\primary_synthetic_3.json --dry-run
```

清单中的 `group` 与 `notes` 会明确保留“synthetic-pristine / not a phone photo”属性，避免把这 3 张图误计入真实手机拍照覆盖率。
