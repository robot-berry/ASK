# test3 数学手机搜题机

这是独立部署的小学、初中和高中数学搜题服务，适用于计算题、应用题、解答题、证明题和综合题（当前不接收选择题、填空题）。手机拍照上传后，服务会先进行自动方向判断、页面外框校正和安全倾斜校正，再由数学模型识别题干，按题目所在学段使用相应课程知识，生成固定四张分层提示图。它沿用 test3 的“答案盲提示”方式，不直接展示完整解答或最终答案，也不会用高学段方法替代低学段教材内的常规方法。

为优先保证题目完整，当前搜题流程固定保留照片中的全部笔迹。黑笔、铅笔与印刷字重叠时无法安全区分，本模块不承诺去除；底层已有的彩色笔迹实验模式也未在搜题页面开放。

物理和数学的统一手机入口位于 `unified_portal`。启动物理 `8010`、数学 `8020` 与统一入口 `8030` 后，手机只需打开 `http://电脑局域网地址:8030/`，即可在同一页面选择学科、上传并查看四张结果图；浏览器不会跳转到其他端口。

## 四张结果图的固定合同

结果始终按以下名称、顺序和内容生成：

1. `01_question_knowledge.png`：忠实重排完整原题，保留题目需要的原图配图，并列出知识点和基础公式；
2. `02_guidance_advanced.png`：学霸版，以最少提示给出解题入口；
3. `03_guidance_standard.png`：普通版，给出标准分步提示和易错点；
4. `04_guidance_foundation.png`：基础版，给出更细的分步提示和通用套路。

每张结果图均为 800 像素宽、8 位灰度 `L` 模式、PNG 格式；PNG 的 DPI 与物理 test3 保持一致，按 9 cm 展示宽度写入，高度随内容自动变化。拍照预处理图只作为模型识别和裁取原题配图的共同来源，不会成为第五张结果图，也不会替换以上四张图。

题干显式引用的图号必须有对应的原图裁片。若题目或配图不完整，服务会要求模型重新检查；仍不能补齐时会把任务标为失败，不会把残缺结果显示为“已完成”。

## Windows 启动

可直接复用 `ASK\picture\test1\.venv` 的 Python 环境；如果该环境不存在，也可以在本目录创建独立环境：

```powershell
py -3.11 -m venv .venv
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

模型凭据使用 `DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL` 和可选的 `DASHSCOPE_WORKSPACE_ID`。可放在 `source\api_keys.env`；若该文件不存在，启动脚本会读取 `ASK\api_keys.env`。凭据文件已被忽略，不应写进结果目录或提交到代码仓库。

## Docker

```text
docker build -t math-test3-search-machine .
docker run --rm -p 8020:8020 --env-file source/api_keys.env -v math-test3-runs:/app/device_runs math-test3-search-machine
```

## 局域网安全边界

当前网页只面向受信任局域网内的单机或小规模测试，不含用户登录。不要把密钥写进镜像，也不要直接把 8020 端口暴露到公网。若需要公网访问，应先在反向代理中增加身份认证、HTTPS、上传大小限制和请求限流，避免未授权访问或付费模型被滥用。
