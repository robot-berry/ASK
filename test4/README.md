# Test4 服务器版物理数学搜题机

Test4 是从已验收的 Test3 代码整理出的独立部署工程。它同时服务：

- 手机浏览器：选择物理或数学，拍照/上传后直接查看四张结果图；
- 联网硬件：通过稳定的 `v1` HTTP API 上传 JPEG/PNG/WebP、轮询任务并下载结果。

数学覆盖小学、初中和高中；物理覆盖小学科学中的物理内容、初中和高中。目前主要保证完整的非选择题（计算、解答、证明、实验探究、简答和综合应用）；选择、填空、判断题暂不承诺答案隔离。

> 本仓库只保存程序源码、测试和部署模板。真实题图、模型响应、运行任务、访问令牌、
> 服务器证书、付费测试结果和发布压缩包不会提交到 GitHub。

## 工程做什么

一张题目照片进入系统后，主要经过以下流程：

```text
拍照或上传
  → Python 图像预处理（方向、透视、清晰度和安全边界）
  → 数学/物理专用模型提示词与结构化输出
  → 结构、题目覆盖和“不得直接给最终答案”校验
  → 中文宋体风格、英文与数字 Times 风格、公式专用字体渲染
  → 输出四张适合手机和电子纸设备查看的灰度 PNG
```

数学和物理共用服务框架、图片输入、任务队列、四图合同、渲染接口和统一入口；
两者主要在模型提示词、学科结构校验和公式/知识点规则上分开。

## 目录说明

```text
test4/
├─ math/                 数学识题、校验、渲染、预处理和测试
├─ physics/              物理识题、校验、渲染、预处理和测试
├─ portal/               数学/物理统一网页入口和反向代理层
├─ deploy/               nginx、systemd、安装、清理和冒烟测试脚本
├─ tests/                统一入口测试
├─ tools/                离线图片集预处理与检查工具
├─ requirements.txt      服务器运行依赖
├─ requirements-dev.txt  本地测试依赖
└─ api_keys.env.example  环境变量示例，不含真实密钥
```

`math/source/models/` 和 `physics/source/models/` 下的 ONNX 预处理权重属于大体积运行资源，
不会提交到源码仓库。部署或本地运行前，需要将项目使用的三个模型权重放入两个目录：

- `PP-LCNet_x1_0_doc_ori.onnx`
- `PP-LCNet_x1_0_textline_ori.onnx`
- `PP-OCRv5_mobile_det.onnx`

不要把第三方模型权重、数据集或字体文件重新分发到公开仓库，除非已经确认相应许可证允许。

## 固定四图结果

1. `01_question_knowledge.png`：原题知识图
2. `02_guidance_advanced.png`：学霸版（最少提示）
3. `03_guidance_standard.png`：普通版（标准提示）
4. `04_guidance_foundation.png`：基础版（详细提示）

四张图均保持 Test3 合同：PNG、宽 800 px、灰度 `L`、物理宽度 9 cm、高度随内容自动变化。

## 本地运行与测试

Windows 本地数学服务与统一入口可使用：

```powershell
.\start_local_math.ps1
```

停止本地服务：

```powershell
.\stop_local_math.ps1
```

手机局域网测试使用 `start_local_mobile.ps1`，脚本会明确绑定局域网地址；不要把本地调试端口
直接暴露到公网。

测试文件位于数学、物理 `source` 目录和根目录 `tests` 中。示例：

```powershell
python math/source/test_search_machine.py
python math/source/test_math_guidance_pipeline.py
python physics/source/test_search_machine.py
python physics/source/test_physics_guidance_pipeline.py
python tests/test_portal.py
```

测试默认不应调用付费模型。凡是会访问阿里云百炼的批量测试，都必须使用独立的显式授权参数、
限制最大新任务数，并把运行目录放在 Git 忽略范围内。

## 服务结构

```text
公网 HTTPS 443
       │
       ▼
    nginx
       │ 127.0.0.1:8030
       ▼
Test4 统一入口
   ├── 127.0.0.1:8010 物理
   └── 127.0.0.1:8020 数学
```

服务器只应公开 80/443。8010、8020、8030 均只监听回环地址，不对外开放。

## Linux 服务器安装

推荐 Ubuntu/Debian，Python 3.10 或更高。把整个 `test4` 上传到服务器临时目录后：

```bash
sudo bash deploy/scripts/install_server.sh
sudoedit /etc/solver-test4/test4.env
```

必须填写：

- `DASHSCOPE_API_KEY`
- `TEST4_ACCESS_TOKEN`（建议 `openssl rand -hex 32`）

若已有解析到服务器的正式域名（脚本会申请或复用 Let's Encrypt 证书）：

```bash
sudo bash /opt/solver-test4/current/deploy/scripts/configure_nginx.sh --domain solver.example.com admin@example.com
```

仅用于短期公网 IP 测试（HTTP 不适合正式使用）：

```bash
sudo bash /opt/solver-test4/current/deploy/scripts/configure_nginx.sh --http-ip 203.0.113.10
```

然后启动并检查：

```bash
sudo systemctl restart solver-test4.target
sudo systemctl enable --now solver-test4-cleanup.timer
curl -fsS http://127.0.0.1:8010/health
curl -fsS http://127.0.0.1:8020/health
curl -fsS http://127.0.0.1:8030/health
```

正式地址可用后，以交互方式读取令牌并运行端到端验收；脚本会自动生成 `Idempotency-Key`，重发同一图片确认没有重复建任务。不要把真实令牌写在 `--token` 参数或命令历史中。

```bash
read -rsp 'Test4 token: ' TEST4_ACCESS_TOKEN; echo
export TEST4_ACCESS_TOKEN
python /opt/solver-test4/current/deploy/scripts/smoke_test.py \
  --base https://solver.example.com/ --subject physics \
  --upload-mode raw --image /path/to/physics.jpg
python /opt/solver-test4/current/deploy/scripts/smoke_test.py \
  --base https://solver.example.com/ --subject math \
  --upload-mode multipart --image /path/to/math.jpg
unset TEST4_ACCESS_TOKEN
```

## 数据与隐私

- 原始上传、预处理图和结果保存在 `/var/lib/solver-test4`。
- 断线重试映射保存在 `/var/lib/solver-test4/portal/gateway.sqlite3`（SQLite/WAL）；只保存鉴权主体和 key 的不可逆标识，不保存原始令牌。
- 定时器默认只删除超过 7 天的 `completed`/`failed` 任务；排队或处理中任务不会删除。
- 题图会发送给已配置的 DashScope 模型，请不要上传与题目无关的个人照片。
- 黑笔/铅笔与印刷内容重叠时无法可靠删除；当前服务器流程默认保留照片笔迹，避免误删题干、公式或图形。
- `api_keys.env.example` 只列出变量名和安全示例；真实 `api_keys.env` 永远不应提交。
- 公开仓库不包含真实学生题图、手写答案、模型返回正文或服务器任务数据库。

## 运维命令

```bash
sudo systemctl status solver-test4.target
sudo journalctl -u solver-test4-physics -u solver-test4-math -u solver-test4-portal -f
sudo systemctl restart solver-test4.target
sudo systemctl stop solver-test4.target
```

更新前先确认两个后端 `/health` 的 `queue_size` 都为 0，避免中断正在处理的任务。

硬件接口见 [HARDWARE_API.md](HARDWARE_API.md)。
