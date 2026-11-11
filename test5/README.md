# 拍照题目 OCR 重排预处理

本工程用于把手机拍摄的物理题图片整理成适合交给大模型继续识别的 PNG。处理重点不是简单地把整张照片二值化，而是先恢复页面方向和正视图，再通过一次 OCR 分离印刷文字、公式、插图和手写痕迹，最后重新排版，只保留题目范围。

## 实现目标

1. 自动修正拍照方向，并对倾斜页面做透视矫正和小角度校正。
2. OCR 识别印刷文字，同时保存文字内容、位置和置信度。
3. 在检测副本中遮掉文字，再使用连通域寻找题图、表格、电路图、曲线图和照片。
4. 裁图时从矫正后的原图取像，并保留约 10% 外边距，优先保证题图完整。
5. 对高置信印刷文字重新排版；低置信但具有公式特征的内容保留为图片块，避免公式被 OCR 改错。
6. 手写解题过程不重新插入最终文档；图内的短标签、数字和单位不会因为低置信度被轻易挖掉。
7. 最终同时输出“重排题目 PNG”和“增强后的正视原图”，便于检查和回退。

## 目录说明

- `preprocess.py`：主程序，包含方向判断、透视矫正、OCR 分析、插图检测和重排输出，代码内有详细中文注释。
- `run_test.ps1`：使用 `D:\robot\Documents\ASK\P_DATA\physic` 数据集运行测试。
- `requirements.txt`：Python 依赖。
- `test_output_physic`：默认测试结果目录。

结果目录结构：

```text
test_output_physic/
|-- final_reflow/          # 最终交给大模型的题目重排 PNG
|-- enhanced_original/     # 方向和透视已修正的增强原图
|-- illustrations/        # 从正视原图裁出的候选题图
|-- stages/                # 每一步的调试图
|-- comparison_overview.jpg
|-- report.json
`-- report.csv
```

## 安装依赖

在本目录打开 PowerShell：

```powershell
python -m pip install -r requirements.txt
```

OCR 使用 RapidOCR 和 ONNX Runtime，不需要安装 PyTorch，也不依赖额外的版面定位大模型。

## 测试 physic 数据集

```powershell
.\run_test.ps1
```

测试脚本读取：

```text
D:\robot\Documents\ASK\P_DATA\physic
```

并把结果写到：

```text
D:\robot\Documents\ASK\picture\test5\test_output_physic
```

如果需要保留旧结果，可以直接指定新的输出目录：

```powershell
python .\preprocess.py `
  "D:\robot\Documents\ASK\P_DATA\physic" `
  ".\test_output_physic_new"
```

## 常用参数

```text
input                   第一个位置参数，输入图片或图片目录
output                  第二个位置参数，输出目录
--rotation auto         自动判断方向
--rotation 0/90/180/270 手动指定顺时针旋转角度
--ink-strength 1.0      彩色笔迹弱化强度
--no-illustrations      不检测和插入题图
```

## 处理流程

```text
读取照片和 EXIF 方向
        |
        v
自动旋转 -> 页面四边形检测 -> 透视矫正 -> 小角度校正
        |
        v
图像增强 -> 单次 OCR（文字、坐标、置信度）
        |
        +--> 高置信印刷文字 -> 按阅读顺序重新排版
        |
        +--> 低置信公式 -> 从原图保留为公式图片块
        |
        `--> 生成文字遮罩 -> 连通域检测题图 -> 从正视原图裁图
                                      |
                                      v
                         白底画布合成最终题目 PNG
```

## 调试图含义

- `_00_document_quad`：检测到的页面四边形。
- `_01_rectified`：旋转、透视和倾斜校正后的正视图。
- `_02_enhanced`：光照均衡后的增强图。
- `_03_ocr_structure`：OCR 框及其分类结果。
- `_04_text_mask`：用于移除文字的遮罩。
- `_05_detection_copy`：遮掉文字后用于寻找插图的副本。
- `_06_figure_candidates`：最终保留的插图候选框。
- `_07_removed_mask`：被视为彩色笔迹并弱化的区域。

## 结果使用建议

优先把 `final_reflow` 中的 PNG 交给大模型。若个别题目的公式或复杂图形在重排版中不完整，可以改用同名的 `enhanced_original` PNG；它保留完整题面，只做方向、透视和画质修正。

纯传统图像方法无法百分之百区分“黑色手绘示意图”和“黑色印刷示意图”。本工程通过 OCR 置信度、版面稳定性、图形连通性、贴边特征和候选区域中的疑似手写数量进行联合判断，目标是在不引入额外定位模型的前提下兼顾速度与题图完整性。
