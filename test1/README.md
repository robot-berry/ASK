# 题图预处理机

把手机或摄像头拍摄的纸质数学题转换成适合交给多模态大模型的电子版题图。处理链包括页面轮廓检测、透视校正、双模型自动方向判断/手动旋转、光照与阴影归一化、对比度增强、降噪和题目区块提取。网页默认展示每道“电子题卡”，并同时生成 PNG、多页 PDF 与 Markdown；整页预处理结果仍作为备份保留。

## 输出目录

- `input/`：可选的原始图片或数据集样本
- `output/png/`：清理后的题图
- `output/pdf/`：A4 单页 PDF
- `output/markdown/`：图片引用、元数据和可选 OCR 文本
- `output/questions/<任务 ID>/`：每道题的独立 PNG、题卡 PDF 与题卡 Markdown

## Docker 启动（服务器推荐）

```bash
docker compose up --build -d
```

浏览器打开 `http://服务器地址:8000`。手机访问时，上传按钮可直接调用后置摄像头。

## 本地启动

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Linux/macOS 下把 `.venv/Scripts/` 改成 `.venv/bin/`。API 文档位于 `/docs`，健康检查位于 `/health`。

## API 调用

```bash
curl -X POST http://localhost:8000/api/process \
  -F "image=@question.jpg" \
  -F "mode=binary" \
  -F "rotation=auto" \
  -F "handwriting_mode=color" \
  -F "extract_questions=true" \
  -F "enable_ocr=false"
```

`mode` 支持 `binary`（黑白扫描）、`gray`（灰度）和 `color`（保留彩图）。`rotation` 支持 `auto`、`0`、`90`、`180`、`270`；角度表示顺时针旋转。`deskew_mode=safe`（默认）会用印刷文字行估计细微倾斜，并在扩展画布上旋转，避免裁掉页角；设为 `off` 可关闭。`handwriting_mode=color` 会去除蓝、红、绿等彩色笔迹，`none` 则保留全部内容。`extract_questions=true` 会返回 `questions` 题卡列表以及 `urls.questions_pdf` / `urls.questions_markdown`。响应中的 URL 可交给后续大模型服务。单张图片限制为 20 MB。

`quality` 支持 `fast`、`high`、`original`。默认 `fast` 会把长边限制为 1800 像素，适合手机即时测试；`high` 为 2600 像素；`original` 保留相机原始分辨率。响应中的 `elapsed_seconds` 是服务器实际处理时间。彩色增强和 OCR 比黑白/灰度模式更慢。

题卡提取以完整性优先：横向保留页面安全宽度，纵向为全部小问、公式和配图预留区域；边界不可靠时返回整页，并始终保留整页 PNG/PDF/Markdown 备份。响应中 `questions[].completeness` 为 `guarded` 或 `whole-page-fallback`。

自动方向先用四方向文档模型判断横向/侧向，再从规则印刷文字行中使用 0/180°文字行模型判断正反。响应会返回 `rotation`、`rotation_confidence` 和 `rotation_method`。模型文件与校验值见 `models/README.md`。

> 电子题卡是保留公式和配图的清晰裁图，不是 OCR 重排文本。彩色笔迹去除是保守的本地图像算法，也可能去掉彩色印刷图形。黑色中性笔或铅笔与黑色印刷公式很难只靠颜色区分，当前不会强制删除，避免破坏题目。若手写覆盖了原题字符，单张照片也无法保证恢复被遮挡的真实符号。

## MathVision-Wild

项目参考 [MathLLMs/MathVision-Wild](https://huggingface.co/datasets/MathLLMs/MathVision-Wild) 的真实拍摄条件。该数据集为 Apache-2.0，完整集约 13.6 GB，不随工程自动下载。

下载 20 张评测样本：

```bash
pip install datasets
python scripts/download_mathvision_sample.py
```

数据集用于评测预处理质量；工具本身不训练模型、不需要 GPU。公式 OCR 的可靠性有限，建议后续视觉大模型直接读取清理后的 PNG。

如果服务器旁有干净电子题图库，可设置 `QUESTION_REFERENCE_DIR` 指向 PNG 目录。工具会先做双向几何校验；只有候选的特征分布、正反投影误差和领先幅度都通过时，才用电子原图。对于本工程 `photo_编号_项目.jpg` 测试文件，还支持同编号 `screenshot_编号_项目.png` 的确定性配对恢复。普通手机文件名不会走编号配对；匹配不确定时自动回退到保真去阴影/去污，不猜题替换。

匹配到的电子原图本身已经是一道完整题，题卡会直接保留整张，不再二次裁切。没有匹配时仍采用保守题区检测；证据不足就返回整页，优先保证公式、选项和配图完整。

倾斜处理分为两层：文字行角度基本一致时使用扩画布旋转；若同一页面上、下文字行角度不同，说明存在梯形透视，单一旋转无法修复。此时仅在本地电子题库能提供通过双向几何校验的同题参考时，使用完整单应性校正照片像素；参考不可靠则不强行拉伸。响应中的 `perspective_status=applied` 表示透视校正已执行，网页会显示“拍摄透视倾斜已校正”。

## 测试

```bash
pytest -q
```

测试会合成带透视畸变的纸张照片，并验证三种输出都能正确生成。
