# Test6 单服务器多设备部署

本文面向初期约 100 台搜题设备共同访问一台 ECS 的场景。容量按峰值同时提交数计算，而不是按登记设备总数计算。

针对 64 vCPU、128 GiB 的 `ecs.c9a.16xlarge` 的实例、磁盘、网络、安全组和可直接粘贴的环境变量，见 [ECS_C9A_16XLARGE_CONFIGURATION.md](ECS_C9A_16XLARGE_CONFIGURATION.md)。

## 已实现的并发策略

- 网关支持每台设备独立 Bearer Token、`X-Device-SN` 和 `Idempotency-Key`。
- 相同设备、学科、幂等键和图片只创建一个任务；断线重传不会重复调用模型。
- 学科服务按设备轮转取任务，同一设备默认最多同时处理一题，避免单台设备占满全部工作槽。
- 每个工作线程持有独立流水线和独立 RapidOCR 引擎，因此不再受单个 `SearchMachinePipeline._solve_lock` 串行化影响。
- OCR 槽和 DashScope 模型槽分别限流。工作线程等待模型时，其他线程仍可使用空闲 OCR 槽处理后续设备。
- `/health` 返回活动任务、等待任务、活动设备、等待设备以及存活工作线程数。
- 任务响应返回提交时队列深度，以及任务排队、OCR槽等待、OCR计算、模型槽等待和模型调用时延。

## 64 vCPU / 128 GiB 初始配置

数学请求通常多于物理时，可在 `/etc/solver-test4/test4.env` 使用：

```dotenv
TEST6_RAPIDOCR_THREADS=8

TEST6_MATH_QUEUE_CAPACITY=200
TEST6_MATH_PER_DEVICE_PENDING=4
TEST6_MATH_PER_DEVICE_ACTIVE=1
TEST6_MATH_JOB_WORKERS=8
TEST6_MATH_OCR_CONCURRENCY=4
TEST6_MATH_MODEL_CONCURRENCY=8
TEST6_MATH_MODEL_BUDGET_SECONDS=28
TEST6_MATH_POST_LOWER_BUDGET_SECONDS=15

TEST6_PHYSICS_QUEUE_CAPACITY=200
TEST6_PHYSICS_PER_DEVICE_PENDING=4
TEST6_PHYSICS_PER_DEVICE_ACTIVE=1
TEST6_PHYSICS_JOB_WORKERS=4
TEST6_PHYSICS_OCR_CONCURRENCY=2
TEST6_PHYSICS_MODEL_CONCURRENCY=4
TEST6_PHYSICS_MODEL_BUDGET_SECONDS=40
TEST6_PHYSICS_POST_LOWER_BUDGET_SECONDS=15
```

这组参数最多同时运行约 6 个本地 OCR，每个 RapidOCR 会使用至多 8 个 ONNX Runtime 线程，理论上占用约 48 个 vCPU。剩余资源用于上传、排版、插图裁剪、五层 PNG 渲染、网关和系统波动。数学与物理模型调用合计最多 12 个；必须结合百炼账号实际 RPM/TPM 和 429 记录调整。

如果数学和物理负载接近，可改成两边各 6 个工作线程、3 个 OCR 槽和 6 个模型槽。不要把两个学科都设置成 8 个 OCR 槽，否则峰值时最多会有 16 个 OCR 同时争抢 64 vCPU。

## 32 vCPU / 64 GiB 节省配置

```dotenv
TEST6_RAPIDOCR_THREADS=6
TEST6_MATH_JOB_WORKERS=5
TEST6_MATH_OCR_CONCURRENCY=3
TEST6_MATH_MODEL_CONCURRENCY=5
TEST6_PHYSICS_JOB_WORKERS=3
TEST6_PHYSICS_OCR_CONCURRENCY=1
TEST6_PHYSICS_MODEL_CONCURRENCY=3
```

这一配置适合 100 台登记设备但峰值约 5～10 台同时提交。队列可以接收更多任务，但集中上传时后续设备会看到更长的排队时间。

## 设备请求要求

每台正式硬件应拥有唯一令牌，并在每次拍题时发送：

```http
Authorization: Bearer <该设备独立令牌>
X-Device-SN: ASK-CAMERA-0007
Idempotency-Key: ASK-CAMERA-0007-0000001234
```

如果暂时共用一个令牌，服务仍会用合法的 `X-Device-SN` 分离公平调度和幂等范围，但这只适合过渡阶段。正式上线应使用独立令牌，以便单独撤销丢失或异常设备。

可在安全的管理机上一次生成 100 台设备的独立凭据：

```bash
python provision_device_tokens.py \
  --count 100 \
  --sn-prefix ASK-CAMERA \
  --output-dir /secure/test6-device-credentials
```

`device_credentials.csv` 含各设备要烧录的明文令牌，只能保存在受控位置；不要上传到服务器或发进聊天。服务器只需要把 `device_token_hashes.env` 中的一行合并到 `/etc/solver-test4/test4.env`，然后重启 portal。工具默认拒绝覆盖已有凭据，避免误操作导致所有设备令牌失效。

## 上线前压测门槛

至少分别进行 1、4、8、12、20 台虚拟设备同时提交的测试，并记录：

- HTTP 202 接收成功率及重复请求是否返回同一 `job_id`；
- 提交时队列深度和服务器排队时间；
- OCR槽等待、OCR计算和首张原题图总等待；
- 模型槽等待、模型调用、六图全部完成时间；
- CPU、内存、磁盘剩余空间以及 DashScope 429/5xx；
- 数学任务持续提交时，物理设备是否仍能及时获得本学科槽位。

建议验收目标：无任务丢失、同一幂等键无重复任务、单设备不会占满所有工作线程、工作线程异常后健康接口能显示降级、峰值后队列可以持续下降。单题 5 秒 OCR 目标必须在目标 ECS 上实测，不能由核数线性推算。

工程根目录的 `benchmark_multi_device.py` 会使用硬件原始字节接口并发提交，输出 `summary.json` 和逐请求 `requests.csv`。它会真实调用 DashScope，因此必须显式设置临时测试令牌并添加确认参数，例如：

```bash
export TEST6_BENCHMARK_TOKEN='临时压测设备令牌'
python benchmark_multi_device.py \
  --base-url https://example.com \
  --subject mixed \
  --math-image /secure/test/math.png \
  --physics-image /secure/test/physics.png \
  --devices 20 --concurrency 20 \
  --output-dir /secure/results/load-20 \
  --confirm-live-calls
```

令牌只从环境变量读取，不应写进命令参数、报告或发布包。

## 单机边界

单服务器可以解决初期容量，但仍是单点：重启、磁盘故障或系统升级会同时影响全部设备。任务原图和结果应定时备份；当峰值长期超过单机能力时，再把本机队列和结果状态迁移到共享存储并增加第二台计算节点。
