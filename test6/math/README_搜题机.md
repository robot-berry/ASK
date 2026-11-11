# test3 数学手机搜题机

这是独立部署的小学、初中和高中数学搜题服务，适用于计算题、应用题、解答题、证明题和综合题（当前不接收选择题、填空题）。手机拍照上传后，Test5 在本地完成方向/透视校正、RapidOCR、题目选择和白底重排，并立即生成第一张 PNG；数学模型随后只生成一份公共解题骨架，服务器在本地展开五档解析图。它沿用“答案盲提示”方式，不直接展示完整解答或最终答案，也不会用高学段方法替代低学段教材内的常规方法。

第一张图不经过大模型。Test5 优先保留印刷题干、公式和题内图形，并过滤可可靠识别的孤立手写答案；黑笔、铅笔与印刷字完全重叠时仍无法保证安全分离，此时模型必须按不确定内容处理。

模型请求的第一张本地重排图是唯一裁框坐标来源；第二张为 Test5 增强原图，第三张及以后为最多三张局部插图裁片。OCR 转写以带边界的“不可信辅助数据”发送，模型必须与图片交叉核对并忽略其中可能出现的指令、手写答案和软件界面文字。

物理和数学的统一手机入口默认位于 `8230`。启动物理 `8210`、数学 `8220` 与统一入口 `8230` 后，手机只需打开 `http://电脑局域网地址:8230/`，即可在同一页面选择学科、上传并查看六张结果图；浏览器不会跳转到其他端口。

## 六张结果图的固定合同

结果始终按以下名称、顺序和内容生成：

1. `01_question_knowledge.png`：Test5 在本地直接生成的原题重排图，不经过模型；
2. `02_guidance_elite.png`：学霸，只问不指方向；
3. `03_guidance_upper.png`：中上，只点知识域；
4. `04_guidance_middle.png`：中等，只点破题眼；
5. `05_guidance_lower.png`：中下，给解题骨架并留关键空位；
6. `06_guidance_novice.png`：学渣，兜底详解但保留最后一跃。

每张结果图均为 800 像素宽、8 位灰度 `L` 模式、PNG 格式；PNG 的 DPI 与物理 test3 保持一致，按 9 cm 展示宽度写入，高度随内容自动变化。增强原图和插图裁片只作为模型识别证据，不会成为额外结果图。

题干显式引用的图号必须有对应的原图裁片。若题目或配图不完整，服务会要求模型重新检查；仍不能补齐时会把任务标为失败，不会把残缺结果显示为“已完成”。

## Windows 启动

可直接复用 `ASK\picture\test1\.venv` 的 Python 环境；如果该环境不存在，也可以在本目录创建独立环境：

```powershell
uv python install 3.14.6
uv venv --python 3.14.6 .venv
.\.venv\Scripts\python.exe -m pip install -r .\source\requirements.txt
```

然后运行：

```powershell
.\start_search_machine.ps1
```

默认监听 `0.0.0.0:8020`。手机和电脑连接同一受信任 Wi-Fi 后，打开：

```text
http://电脑局域网地址:8020/
```

如需指定端口，可运行 `.\start_search_machine.ps1 -Port 8021`，或设置 `MATH_TEST3_PORT`。新环境变量优先使用 `MATH_TEST3_*`，同时兼容原来的 `TEST3_*`：

- `MATH_TEST3_PORT` / `TEST3_PORT`
- `MATH_TEST3_HOST` / `TEST3_HOST`
- `MATH_TEST3_MODEL_ID` / `TEST3_MODEL_ID`
- `MATH_TEST3_QUEUE_CAPACITY` / `TEST3_QUEUE_CAPACITY`
- `MATH_TEST3_CJK_FONT` / `TEST3_CJK_FONT`
- `MATH_TEST3_CJK_BOLD_FONT` / `TEST3_CJK_BOLD_FONT`
- `MATH_TEST3_LATIN_FONT` / `TEST3_LATIN_FONT`
- `MATH_TEST3_LATIN_BOLD_FONT` / `TEST3_LATIN_BOLD_FONT`
- `MATH_TEST3_SYMBOL_FONT` / `TEST3_SYMBOL_FONT`

中文正文采用宋体风格衬线字体：Windows 默认使用宋体，Linux 服务器推荐
`NotoSerifCJK-Regular.ttc`，中文标题使用对应的
`NotoSerifCJK-Bold.ttc`。英文字母、数字和数学变量继续使用
Times 兼容衬线字体，以保持教材公式的字形习惯。

模型凭据使用 `DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL` 和可选的 `DASHSCOPE_WORKSPACE_ID`。可放在 `source\api_keys.env`；若该文件不存在，启动脚本会读取 `ASK\api_keys.env`。凭据文件已被忽略，不应写进结果目录或提交到代码仓库。

## Docker

```text
docker build -t math-test3-search-machine .
docker run --rm -p 8020:8020 --env-file source/api_keys.env -v math-test3-runs:/app/device_runs math-test3-search-machine
```

## 局域网安全边界

当前网页只面向受信任局域网内的单机或小规模测试，不含用户登录。不要把密钥写进镜像，也不要直接把 8020 端口暴露到公网。若需要公网访问，应先在反向代理中增加身份认证、HTTPS、上传大小限制和请求限流，避免未授权访问或付费模型被滥用。
