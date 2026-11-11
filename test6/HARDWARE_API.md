# Test6 流式硬件接口 v1

## 认证

硬件在每个请求中发送：

```http
Authorization: Bearer <TEST6_ACCESS_TOKEN>
```

可为每台硬件分配不同随机令牌，并只把各令牌的 SHA-256 写入服务器的 `TEST6_DEVICE_TOKEN_HASHES`。这样可以单独撤销设备，而不影响手机登录令牌。令牌不得放在 URL、文件名或日志中。正式设备必须校验服务器 HTTPS 证书。

约 100 台设备共同接入时，每台设备还应发送稳定且唯一的 SN：

```http
X-Device-SN: ASK-CAMERA-0007
```

SN 只允许 1～128 位 ASCII 字母、数字、`.`、`_`、`:`、`-`。服务按设备公平调度，并把 SN 写入后台任务证据。正式环境仍应为每台硬件分配独立令牌；共用令牌加不同 SN 只作为过渡兼容方式。

命令行测试时建议交互读取令牌，避免把真实令牌写进命令历史：

```bash
read -rsp 'Test6 token: ' TOKEN; echo
export TOKEN
```

## 能力查询

```http
GET /api/v1/capabilities
```

返回学科、最大上传字节数、支持的媒体类型及六张结果图合同（一张原题图 + 五档解析图）。

## 方式一：multipart 上传

```bash
curl --fail-with-body -sS \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Device-SN: ASK-CAMERA-0007" \
  -H "Idempotency-Key: $IDEMPOTENCY_KEY" \
  -F "image=@question.jpg;type=image/jpeg" \
  "$BASE/api/v1/math/jobs"
```

物理把 `math` 换成 `physics`。上传字段固定为 `image`。

## 方式二：原始图片字节上传

适合不方便生成 multipart 的相机或嵌入式设备：

```bash
curl --fail-with-body -sS \
  -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Device-SN: ASK-CAMERA-0007" \
  -H "Idempotency-Key: $IDEMPOTENCY_KEY" \
  -H "Content-Type: image/jpeg" \
  --data-binary @question.jpg \
  "$BASE/api/v1/physics/jobs/raw"
```

支持 `image/jpeg`、`image/png`、`image/webp`，编码后图片最大 20 MiB。

## 断线重试与 Idempotency-Key

硬件每拍一题应生成一个新的 `Idempotency-Key`，并在本地保留该 key 和**完全相同的编码图片字节**，直到收到 `202`。网络断开或响应不确定时，向原学科接口重发同一个 key 和同一份字节；不要重新压缩图片，也不要换 key。

- 格式严格为 8-128 位 ASCII 字符：字母、数字、`.`、`_`、`~`、`-`；建议使用 UUID 或“设备短名 + 单调序号”。
- 相同鉴权主体、学科、key 和图片只创建一个上游任务；重复请求仍返回 `202`、相同 `job_id` 和相同 `poll_url`。
- 首次响应含 `Idempotency-Replayed: false`，命中已有任务时为 `Idempotency-Replayed: true`。
- 同一鉴权主体和学科下，同 key 换成不同图片会返回 `409`，此时必须为新图片生成新 key。
- `429` 或 `503` 时按 `Retry-After` 等待，并继续使用原 key。上游在创建前明确失败时，该 key 会被释放，可安全重试。
- key 按鉴权主体隔离。网关只持久化令牌和 key 的不可逆标识，不把原始令牌放入数据库、URL 或响应。

幂等记录保留 7 天。设备不得依赖超过该期限的去重结果。

## 异步任务

创建成功返回 HTTP `202`，响应头含：

- `Location`：状态查询路径
- `Retry-After: 2`：建议轮询间隔（秒）

JSON 主要字段：

```json
{
  "api_version": "v1",
  "subject": "math",
  "job_id": "20260815T000000_...",
  "status": "queued",
  "retry_after_seconds": 2,
  "poll_url": "/api/v1/math/jobs/任务编号",
  "scheduler": {
    "queue_depth_observed": 3,
    "worker_count": 8,
    "ocr_concurrency": 4,
    "model_concurrency": 8
  },
  "images": []
}
```

`queue_depth_observed` 是提交瞬间的队列观测值，不是固定排位。任务开始后，`timing` 会逐步增加 `queue_wait_seconds`、`ocr_slot_wait_seconds`、`ocr_processing_seconds`、`model_slot_wait_seconds` 和 `model_call_seconds`，用于区分多设备排队、OCR算力和模型服务时延。

状态顺序为：`queued` → `preprocessing` → `solving` → `rendering` → `completed`。Test6 的 `images` 是固定六图序列的已生成前缀：在 `solving` 阶段可能先出现第一张，在 `rendering` 阶段依次增加学霸、中上、中等、中下、学渣五档解析图。失败时为 `failed`，可读取安全的用户提示，但服务器内部路径和模型审计不会公开。

```bash
curl --fail-with-body -sS \
  -H "Authorization: Bearer $TOKEN" \
  "$BASE/api/v1/math/jobs/$JOB_ID"
```

## 逐张下载 PNG

只要某一项出现在 `images` 中即可立即下载，不必等待 `completed`。始终按 `images[].sequence` 处理；每次下载都要继续发送 Bearer Token：

```bash
curl --fail-with-body -sS \
  -H "Authorization: Bearer $TOKEN" \
  "$BASE/api/v1/math/jobs/$JOB_ID/images/01_question_knowledge.png" \
  -o 01_question_knowledge.png
```

每个已发布图片项都会提供 `width`、`height`、`mode`、`bytes`、`sha256`。硬件应校验：

- 文件名和顺序固定；
- PNG 宽 800 px；
- 模式为 `L`；
- 下载字节数和 SHA-256 与状态响应一致。

内存较小的硬件应流式上传 JPEG，并逐张下载到存储介质，不要一次把六张图全部载入内存。

## 常见状态码

- `202`：任务创建成功
- `400`：图片无法解码或请求格式错误
- `401`：访问令牌缺失或错误
- `409`：同一 Idempotency-Key 已用于另一张图片
- `404`：学科、任务或结果不存在
- `413`：图片超过 20 MiB
- `415`：不是 JPEG/PNG/WebP
- `429`：任务队列或限流已满，按 `Retry-After` 重试
- `503`：对应学科服务暂不可用
