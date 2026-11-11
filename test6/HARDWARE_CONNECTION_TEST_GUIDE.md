# Test4 搜题服务——硬件连接测试说明

## 1. 统一测试入口

手机和硬件只连接下面这一个公网入口：

```text
https://47.114.76.156
```

物理和数学不是两个网站，也不需要连接不同端口。设备在同一个入口下通过 `subject` 选择学科：

| 学科 | `subject` 值 | 创建任务路径 |
| --- | --- | --- |
| 数学 | `math` | `/api/v1/math/jobs` 或 `/api/v1/math/jobs/raw` |
| 物理 | `physics` | `/api/v1/physics/jobs` 或 `/api/v1/physics/jobs/raw` |

服务器内部虽然分别运行数学处理、物理处理和统一网关，但这些内部服务不会直接暴露给设备。硬件只访问 HTTPS `443` 端口上的统一网关。

> 当前连接前置条件：需要先在阿里云 ECS 安全组中放行入方向 TCP `443`。未放行时，手机和硬件都无法访问上述地址。

## 2. 认证与安全要求

部署方会通过安全渠道单独交付设备 Bearer 令牌。本文不包含真实令牌或模型密钥。

每个 API 请求都要发送：

```http
Authorization: Bearer <DEVICE_TOKEN>
```

安全要求：

- 使用部署方单独提供的设备令牌，不要使用示例占位符。
- 不得把令牌放进 URL、文件名、截图、普通聊天消息或运行日志。
- 正式设备必须校验服务器 HTTPS 证书以及证书中的 IP 身份，不得关闭证书校验，也不得使用 `curl -k`、`verify=False` 等绕过方式。
- 设备系统时间必须准确，否则可能无法验证 HTTPS 证书。
- 如果证书校验失败，停止发送题图和令牌，并联系部署方检查证书；不要降级成 HTTP。

命令行测试时可以交互读取令牌，避免把真实值写入命令历史：

```bash
read -rsp 'Test4 device token: ' TOKEN; echo
BASE='https://47.114.76.156'
```

## 3. 检查服务能力

```http
GET https://47.114.76.156/api/v1/capabilities
Authorization: Bearer <DEVICE_TOKEN>
```

命令行示例：

```bash
curl --fail-with-body -sS \
  -H "Authorization: Bearer $TOKEN" \
  "$BASE/api/v1/capabilities"
```

成功时返回 HTTP `200`，内容应包括：

- `api_version: "v1"`
- `subjects` 中同时存在 `math` 和 `physics`
- 支持的图片类型、最大上传大小和固定六张结果图信息
- `idempotency` 中的 `Idempotency-Key` 要求

## 4. 上传题图

支持 JPEG、PNG、WebP，编码后单张图片最大 `20 MiB`。暂不支持 HEIC/HEIF。

拍照时请保证题干、全部小问、公式和配图都在画面内。服务会处理常见的旋转、倾斜和背景干扰，但裁出画面的内容无法恢复。

### 4.1 原始图片字节上传（硬件推荐）

设备先选择 `subject=math` 或 `subject=physics`，再拼出对应路径。数学示例：

```http
POST https://47.114.76.156/api/v1/math/jobs/raw
Authorization: Bearer <DEVICE_TOKEN>
Idempotency-Key: device01-20260815-000001
Content-Type: image/jpeg

<JPEG 图片原始字节>
```

物理只需把路径中的 `math` 改成 `physics`：

```http
POST https://47.114.76.156/api/v1/physics/jobs/raw
```

命令行示例：

```bash
SUBJECT='math'  # 物理改为 physics
IDEMPOTENCY_KEY='device01-20260815-000001'

curl --fail-with-body -sS \
  -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Idempotency-Key: $IDEMPOTENCY_KEY" \
  -H "Content-Type: image/jpeg" \
  --data-binary @question.jpg \
  "$BASE/api/v1/$SUBJECT/jobs/raw"
```

### 4.2 multipart 上传

上传字段名固定为 `image`：

```bash
curl --fail-with-body -sS \
  -H "Authorization: Bearer $TOKEN" \
  -H "Idempotency-Key: $IDEMPOTENCY_KEY" \
  -F "image=@question.jpg;type=image/jpeg" \
  "$BASE/api/v1/$SUBJECT/jobs"
```

## 5. `Idempotency-Key` 与断线重试

每道新题生成一个新的 `Idempotency-Key`，并在本地保留该 Key 和完全相同的编码图片字节，直到确认收到 HTTP `202`。

- 长度严格为 `8–128` 位。
- 首字符必须是 ASCII 字母或数字；其余字符只能使用 ASCII 字母、数字、`.`、`_`、`~`、`-`。
- 推荐使用 UUID，或“设备短名 + 日期 + 单调递增序号”。
- 网络断开或响应不确定时，必须向原学科路径重发同一个 Key 和完全相同的图片字节。
- 重试时不要重新拍摄或重新压缩图片，否则图片字节会变化。
- 同一设备、同一学科、同一 Key 和同一图片只会创建一个任务；重试仍返回同一 `job_id`。
- 同一 Key 换成另一张图片会返回 `409`，此时应为新图片生成新 Key。
- 遇到 `429` 或 `503` 时，按响应中的 `Retry-After` 等待，再使用原 Key 和原图片重试。
- 幂等记录保留 7 天，设备不要依赖超过该期限的去重结果。

首次创建的响应头：

```http
Idempotency-Replayed: false
```

命中已有任务时：

```http
Idempotency-Replayed: true
```

## 6. 创建响应和状态轮询

任务创建成功返回 HTTP `202`。示例：

```json
{
  "api_version": "v1",
  "subject": "math",
  "job_id": "20260815T000000_xxx",
  "status": "queued",
  "retry_after_seconds": 2,
  "poll_url": "/api/v1/math/jobs/20260815T000000_xxx",
  "images": []
}
```

响应头还会包含：

```http
Location: /api/v1/math/jobs/20260815T000000_xxx
Retry-After: 2
```

之后用返回的 `poll_url` 查询，不要自行猜测任务地址：

```http
GET https://47.114.76.156<poll_url>
Authorization: Bearer <DEVICE_TOKEN>
```

命令行示例：

```bash
curl --fail-with-body -sS \
  -H "Authorization: Bearer $TOKEN" \
  "$BASE$POLL_URL"
```

按 `Retry-After` 指定的秒数轮询，常见状态为：

```text
queued -> preprocessing -> solving -> rendering -> completed
任一处理阶段也可能进入 failed
```

不要高频连续请求。建议为单个任务预留最长约 10 分钟等待时间。

## 7. 下载固定六张 PNG

状态变为 `completed` 后，响应中的 `images` 固定按以下顺序提供一张原题图和五档解析图：

1. `01_question_knowledge.png`：Test5 本地原题图（兼容保留旧文件名）
2. `02_guidance_elite.png`：学霸，只问不指方向
3. `03_guidance_upper.png`：中上，只点知识域
4. `04_guidance_middle.png`：中等，点破题眼
5. `05_guidance_lower.png`：中下，给解题骨架
6. `06_guidance_novice.png`：学渣，兜底详解但保留最后一跃

下载地址应使用状态响应里每张图片给出的 URL；每次下载仍需携带相同的 Bearer 令牌。例如：

```http
GET https://47.114.76.156/api/v1/math/jobs/<job_id>/images/01_question_knowledge.png
Authorization: Bearer <DEVICE_TOKEN>
```

命令行示例：

```bash
curl --fail-with-body -sS \
  -H "Authorization: Bearer $TOKEN" \
  "$BASE/api/v1/$SUBJECT/jobs/$JOB_ID/images/01_question_knowledge.png" \
  -o 01_question_knowledge.png
```

硬件应逐张校验：

- 文件名和 `sequence` 顺序固定；
- `Content-Type` 为 `image/png`；
- PNG 宽度为 `800 px`；
- 色彩模式为 8 位灰度 `L`；
- 下载字节数等于状态响应中的 `bytes`；
- 下载文件 SHA-256 等于状态响应中的 `sha256`。

内存较小的设备应流式上传题图，并逐张流式下载到存储介质，不要一次把六张结果图全部载入内存。

## 8. 常见状态码

- `200`：查询成功或图片下载成功
- `202`：任务创建成功
- `400`：图片无法解码或请求格式错误
- `401`：Bearer 令牌缺失或错误
- `404`：学科、任务或图片不存在
- `409`：同一 `Idempotency-Key` 已用于另一张图片
- `413`：图片超过 `20 MiB`
- `415`：图片不是 JPEG、PNG 或 WebP
- `429`：请求过于频繁或任务队列已满，按 `Retry-After` 重试
- `503`：对应学科服务暂不可用，按 `Retry-After` 重试

## 9. 最小测试流程

1. 确认阿里云安全组已放行入方向 TCP `443`。
2. 使用系统信任库正常验证 `https://47.114.76.156` 的 HTTPS 证书。
3. 通过安全渠道取得设备 Bearer 令牌。
4. 请求 `/api/v1/capabilities`，确认数学和物理都存在。
5. 选择 `subject=math` 或 `subject=physics`。
6. 生成新的 `Idempotency-Key`，上传一张完整题图。
7. 保存返回的 `job_id` 和 `poll_url`，按 `Retry-After` 轮询。
8. 状态为 `completed` 后，按顺序下载并校验六张 PNG。
9. 用相同 Key 和相同图片重传一次，确认 `job_id` 不变且 `Idempotency-Replayed: true`。

## 10. 测试反馈内容

请记录并反馈：

- 设备型号、系统版本和网络类型
- `subject` 值（`math` 或 `physics`）
- 原图格式和字节数
- HTTP 状态码、`job_id` 和各阶段耗时
- 是否完整得到六张 PNG
- 六张 PNG 的字节数和 SHA-256 是否匹配
- 失败时的公开错误码和提示

反馈中不得包含 Bearer 令牌、模型密钥、完整请求头或其他秘密信息。
