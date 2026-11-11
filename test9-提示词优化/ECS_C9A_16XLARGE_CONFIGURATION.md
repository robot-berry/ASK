# Test6：`ecs.c9a.16xlarge` 单机配置清单

适用目标：一台 ECS 承载初期约 100 台已登记搜题设备的网关、数学服务、物理服务与本地 OCR；DashScope 模型仍通过公网调用，不在 ECS 上部署大模型。

## 1. 阿里云实例与网络

| 项目 | 建议值 | 说明 |
|---|---:|---|
| 实例规格 | `ecs.c9a.16xlarge` | 64 vCPU、128 GiB 内存，计算型 AMD 实例。 |
| 系统镜像 | Alibaba Cloud Linux 3 x86_64 | 使用受支持的长期维护镜像，开启自动安全更新。 |
| 系统盘 | ESSD PL1，100 GiB | 操作系统、Python 虚拟环境、日志和 nginx。 |
| 数据盘 | ESSD PL1，500 GiB，ext4/xfs | 挂载到 `/var/lib/solver-test4`，保存七天任务、OCR 审计与 SQLite 状态。 |
| 公网访问 | EIP，建议 100 Mbps 起 | 对外只开放 HTTPS；高峰上传较多时可升到 200 Mbps 或按流量计费。 |
| 安全组入站 | TCP 443；TCP 80（仅证书跳转）；TCP 22（仅固定管理 IP） | 不开放 8010、8020、8030、8240。 |
| 安全组出站 | TCP 443 到 DashScope 及系统更新源 | 模型调用需要 HTTPS 出网。 |
| 域名与证书 | 已备案域名 + nginx TLS | 设备应访问域名，不直接访问后端端口。 |

截至 2026-09，阿里云对该规格标注的基础网络为 32 Gbit/s、云盘基础 IOPS 为 120,000、云盘基础带宽为 16 Gbit/s；这些是实例能力，不等于 EIP 的公网带宽。

## 2. 购买页面字段逐项填写

下表与 ECS 下单页一一对应。截图中的规格和 CPU 选项是正确的；其中“镜像为空、40 GiB PL0 系统盘、1 Mbps 带宽、默认安全组”不应按截图直接创建。

| 下单页字段 | 建议填写 | 对截图的结论 |
|---|---|---|
| 付费类型 | 包年包月（业务已稳定时）；验证期可按量付费 | 可用。按量更灵活，包年包月通常成本更低。 |
| 地域 | 华东 1（杭州）；若设备、现有 EIP 和域名服务均在杭州则保持不变 | 可用。不要跨地域迁移现有服务。 |
| 可用区 | 随机分配，或选择有库存的单一可用区 | 可用；单机不需要跨可用区。 |
| 专有网络 / 交换机 | 使用独立的生产 VPC 和交换机；没有则先使用默认 VPC/默认交换机 | 截图可用，但生产环境建议不要混用测试网络。 |
| 实例规格 | `ecs.c9a.16xlarge`（64 vCPU / 128 GiB） | 正确。 |
| CPU 选项 | 每核线程数 2、核心计数 32 | 正确，合计正好是 64 vCPU；不要把核心数改小。 |
| 镜像 | Alibaba Cloud Linux 3 x86_64（公共镜像） | **必填**；当前为空，不能创建。 |
| 系统盘 | ESSD PL1，100 GiB | 截图的 ESSD PL0 40 GiB 仅够最小系统，无法稳妥容纳运行环境、日志和升级缓存。 |
| 数据盘 | 新增 ESSD PL1，500 GiB；挂载到 `/var/lib/solver-test4` | 截图未配置；用于题图、结果图、审计和 SQLite。 |
| 文件备份 | 启用云盘备份：每日一次，保留 7 天（按隐私策略调整） | 截图“激活备份”正确，但还需在备份策略中设保留期。 |
| 公网带宽 | EIP，按固定带宽 100 Mbps 起；上传峰值高可改 200 Mbps 或按流量 | **1 Mbps 不可用**，会显著拖慢图片上传和模型 HTTPS 往返。 |
| 安全组 | 新建 `solver-test6-prod-sg`：443 对外、80 仅证书跳转、22 仅固定管理 IP；8010/8020/8030/8240 不对外开放 | 不要使用不加规则审计的默认安全组。 |
| 登录凭证 | 创建并保存 SSH 密钥对；禁用密码登录 | 可用。私钥只保存在管理员设备，不上传到分享包。 |
| 实例名称 | `solver-test6-prod-01` | 建议更名，避免新服务器继续出现 Test4 的历史名称。 |
| 标签 | `project=test6`、`env=prod`、`service=solver`、`owner=<负责人>` | 建议补充，便于成本、权限和资产管理。 |
| 元数据访问模式 | 优先仅 IMDSv2；若部署工具兼容性未确认，先“普通模式和加固模式”再切换 | 截图可暂用；正式上线应收紧到 IMDSv2。 |

创建后先在安全组中确认公网只可访问 443，再部署服务。公网带宽、系统盘、数据盘和安全组属于实际性能与安全边界，不能只依赖实例的 64 vCPU / 128 GiB 标称规格。

## 3. `/etc/solver-test4/test4.env` 推荐值

以下是“详细五层解析”下的保守起始值。数学与物理最多同时执行 6 个 OCR，每个 OCR 最多 8 个 ONNX Runtime 线程，约占 48 个 vCPU；余量供重排、插图裁剪、PNG 渲染、网关和系统波动使用。

```dotenv
# DashScope、门户令牌、字体、路径等沿用 api_keys.env.example 的必填项。

# 详细五层：不给输出 token 强行截断，给模型足够完成时间。
TEST6_MATH_MODEL_BUDGET_SECONDS=28
TEST6_PHYSICS_MODEL_BUDGET_SECONDS=40
TEST6_MATH_POST_LOWER_BUDGET_SECONDS=15
TEST6_PHYSICS_POST_LOWER_BUDGET_SECONDS=15
TEST6_MATH_MAX_OUTPUT_TOKENS=
TEST6_PHYSICS_MAX_OUTPUT_TOKENS=

# 本地 OCR：6 个槽 × 8 线程 = 48 个 vCPU 的理论上限。
TEST6_RAPIDOCR_THREADS=8

# 数学通常占比更高。
TEST6_MATH_QUEUE_CAPACITY=200
TEST6_MATH_PER_DEVICE_PENDING=4
TEST6_MATH_PER_DEVICE_ACTIVE=1
TEST6_MATH_JOB_WORKERS=8
TEST6_MATH_OCR_CONCURRENCY=4
TEST6_MATH_MODEL_CONCURRENCY=6

TEST6_PHYSICS_QUEUE_CAPACITY=200
TEST6_PHYSICS_PER_DEVICE_PENDING=4
TEST6_PHYSICS_PER_DEVICE_ACTIVE=1
TEST6_PHYSICS_JOB_WORKERS=4
TEST6_PHYSICS_OCR_CONCURRENCY=2
TEST6_PHYSICS_MODEL_CONCURRENCY=3

TEST6_PHYSICS_RUNS_ROOT=/var/lib/solver-test4/physics/device_runs
TEST6_MATH_RUNS_ROOT=/var/lib/solver-test4/math/device_runs
TEST6_PORTAL_STATE_DB=/var/lib/solver-test4/portal/gateway.sqlite3
TEST6_PHYSICS_BACKEND_URL=http://127.0.0.1:8010
TEST6_MATH_BACKEND_URL=http://127.0.0.1:8020
```

模型槽合计从 9 开始，是为了先观察 DashScope 的 429、RPM、TPM 和 P95 模型时延。连续一周无 429、CPU 峰值低于 75%、内存可用高于 25 GiB 时，可逐步提高到数学 8、物理 4；不要一次把 OCR 槽也同时提高。

## 4. 服务与系统设置

- 保持三个 uvicorn 服务都使用 `--workers 1`。Test6 已在服务内部创建按设备公平的工作线程；启动多个 uvicorn 进程会形成多个独立内存队列，破坏单机公平调度。
- nginx 只反向代理门户 8030；数学 8020、物理 8010 和可选公式 OCR 8240 只绑定 `127.0.0.1`。
- 为服务账户设置 `LimitNOFILE=65536`，并确认数据盘挂载在服务启动前完成。
- 配置每日任务清理和数据盘快照；原题、结果图和 SQLite 状态要一起备份。默认保留策略为 7 天可按隐私策略缩短。
- PP-DocLayoutV2/PP-FormulaNet 先保持可选关闭；它们改善极复杂表格/公式，但会额外占用 CPU 和首图时间。先以 RapidOCR 主路径压测，再单独评估是否启用。

## 5. 上线顺序

1. 安装新版 Test6 后，先使用上述环境变量并执行 `systemctl daemon-reload`、重启 `solver-test4.target`。
2. 先用 1、4、8、12、20 台虚拟设备依次压测，记录队列等待、OCR 槽等待、模型槽等待、首图和六图完成时间。
3. 若队列随测试结束能持续下降、无 429、无内存交换，再提高模型并发；若 OCR 槽等待高且 CPU 长期低于 75%，再提高 OCR 槽到数学 5、物理 3。
4. 每台设备必须使用独立 Bearer Token，或至少附带唯一 `X-Device-SN` 和 `Idempotency-Key`。

不要直接把本配置用于当前 2 vCPU 服务器；该服务器应继续保持两学科各 1 个工作、OCR、模型槽。
