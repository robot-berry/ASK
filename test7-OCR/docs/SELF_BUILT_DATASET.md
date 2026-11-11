# test7-OCR 自建数据实施说明

## 1. 唯一主 OCR Gold

本项目当前唯一主 OCR Gold 来自：

```text
D:\robot\Documents\ASK\datasets\gaokao_physics_300_v1_0_1
```

配置中的稳定数据集标识为 `gaokao_physics_300_v1`。目录包含 300 道高中物理题、300 张确定性生成的 `1280×720` PNG，以及与渲染使用同一事实源的题干、选项、公式和图示布局契约。全部样本固定为 `split=test`、`training_allowed=false`。

`testzip_math_physics_300_v1` 已停止作为正式 OCR Gold。它只保留为历史版面、真实扫描、手写污染和安全门禁回归，不得重新参与当前正式准确率评分，也不能替代下面的真实自建采集。

## 2. 这套 Gold 能证明什么

高考物理 300 题的 OCR Gold 不是从 OCR 预测、答案 PDF或人工猜测生成，而是从出题事实源和确定性渲染契约派生。生成器会逐项核对题干、选项、公式、题型、原图哈希、图片尺寸、图示类型和图示框，并明确排除答案、解析、评分点和分类真值，因此当前 300/300 已打开正式的“来源派生合成物理基准”评分。

它能证明同一分布上的文字、公式、选项、800 px 排版、题内图片直裁、LaTeX 防泄漏和时延门禁；不能外推到手机斜拍、阴影、折痕、模糊、真实手写覆盖、中文数学和多题整页。后者必须由第 10 节的 410 张真实自建数据补齐。

## 3. 当前 Gold 与自建数据资产

| 文件 | 作用 |
|---|---|
| `configs/self_built_data.yaml` | 指定唯一主 Gold、真实自建目标、隔离和冻结策略 |
| `data/splits/gaokao_physics_ocr_gold300_manifest.jsonl` | 300 条来源、图片哈希和测试划分索引 |
| `data/splits/gaokao_physics_ocr_gold300_summary.json` | 正式评分状态、题型分布和视觉资产统计 |
| `data/gold/gaokao_physics_300/ocr_gold.jsonl` | 300 条来源派生 OCR Gold；不包含答案或解析 |
| `data/splits/gaokao_physics_visual_bbox_candidates.jsonl` | 190 个由生成布局契约验证的原图裁框 |
| `scripts/prepare_gaokao_physics_ocr_gold.py` | 生成或只读验证主 Gold、来源图片和哈希 |
| `templates/self_built_collection_plan.csv` | 410 行真实采集计划 |
| `scripts/prepare_self_built_collection_workspace.py` | 建立并检查私有真实采集工作区 |
| `scripts/freeze_self_built_dataset.py` | 在数量、双审、隐私和泄漏检查全部通过后冻结自建版本 |

Gold 和真实采集原图均只按路径与 SHA-256 引用，不复制进 Git。真实采集与 Gold 是两个不同角色：Gold 用于当前来源派生准确率评分；自建数据用于验证真实拍照、题界和手写泛化。

## 4. 还需要人工完成什么

当前主 OCR Gold 的 300 条转写已经由事实源和渲染契约验证，不要求人工把同一批题重新抄写一遍。“金标制作”页仍保留为可选来源审计入口，但它不是当前 v65 OCR 处理链路，也不是 300 条来源派生评分的阻塞项。

真正需要人工完成的是三类不能由模型替代的证据：

1. 对当前候选版本的最终 800 px PNG 做哈希绑定验收，确认文字语义、公式显示、选项、裁图、手写排除、相邻题噪声、截断和换行；
2. 对 410 张真实自建图片完成隐私与来源声明，并双审 200 对普通手机照片、100 对带手写照片和 60 页题界；
3. 为至少 100 张真实手写样本绘制像素级 Mask，并由另一名复核员核验。

填空题仍必须使用连续下划线 `______` 保留答案槽。无法同时保留印刷内容和移除手写时必须标成 `overlap_unrecoverable`，不能补写或猜测原文。

## 5. 视觉资产只能裁原图

所有视觉资产必须满足：

```json
{
  "preserve_source_pixels": true,
  "redraw_allowed": false,
  "edge_complete": true
}
```

允许的处理只有方向纠正、灰度化、保守白边整理和等比缩放。禁止：

- 按识别到的点、线、箭头重新绘制；
- 使用 SVG、Canvas 或生成式模型复原原图；
- 为了排版好看裁掉图中文字或伸出主体的标注；
- 手写覆盖印刷图时臆测被覆盖内容。

若不能同时做到完整裁图和排除手写，该题必须标为 `REVIEW` 或 `overlap_unrecoverable`。

## 6. 当前复核优先顺序

1. 先处理实际 OCR 产品页中被用户登记为错误的哈希绑定回归样本；
2. 再核对 v65 的 254 个自动 PASS 成品，46 个自动 REVIEW 先回到工程侧处理，不能靠人工 PASS 放宽门禁；
3. 真实采集时先完成 50 份干净原稿，再拍与其严格配对的普通照片和带手写照片；
4. 带手写照片登记后立即进入 Mask 和拍照配对队列，整页图片进入题界队列；
5. 只有双审及冻结完成的数据才参与真实分布验收。

任何页面显示的 OCR 候选、LaTeX 源码或自动框都只能作为辅助证据，不能直接变成自建 Gold。

## 7. 主 Gold 与自建冻结门禁

主 OCR Gold 的生成器必须同时满足：300 个唯一样本；事实源、题目项和渲染清单一致；全部原图为 `1280×720` PNG 且 SHA-256 匹配；题干、选项与公式非空并一致；190 个视觉资产由生成布局契约定位；答案、解析和评分字段没有进入 OCR Gold；训练权限关闭。当前这组门禁已经 300/300 通过。

真实自建数据使用另一套更严格的冻结门禁：410 张图片齐全、逐图隐私和来源声明完成、拍照配对与题界双审完成、至少 100 张手写 Mask 双审完成、文档组隔离且与主 Gold 没有 SHA-256 泄漏。任何一项未满足，都不得生成正式冻结版本。

## 8. 运行方式

```powershell
.\scripts\run_self_built.ps1
```

该命令会检查 410 行采集工作区和标准计划，重新生成并验证高考物理 300 题来源派生 Gold，同步手写 Mask、拍照配对和题界队列，再执行自建数据与冻结就绪审计。它不会把缺失图片、合成图片或旧 Testzip 样本计入真实采集数量。

冻结验收：

```powershell
.\scripts\run_self_built.ps1 -RequireVerified
```

当前真实采集和双审未完成时，严格冻结验收按设计失败，并明确显示每一类缺口；普通结构检查可以通过。

## 9. 本机人工复核页面

运行：

```powershell
.\scripts\start_gold_review.ps1
```

页面固定入口为 `http://127.0.0.1:8765/review/`，只监听本机，并明确拆成六个不能互相替代的入口。日常上传新题应使用 `http://127.0.0.1:8765/ocr/`，不要把验收台误认为 OCR 产品。

- **最终 PNG 验收**：默认入口，选择具体 `run_id`，并排检查原图和该次流水线实际生成的 800 px PNG。主队列默认排除自动 `REJECT/FAIL`，让人工先验收具备候选资格的成图；被自动拒绝的结果仍可通过专项筛选做失败原因审计。十项人工检查按文字与语义、公式、原题裁图、手写/杂质/版面四组执行，人工结论由清单自动得出，不能另外选择一个互相矛盾的 PASS/FAIL。人工记录绑定原图哈希、成品哈希和运行版本，单独写入 `data/annotations/output_quality_reviews/<run_id>/`。只有十项均被明确判为正常时才允许人工 PASS；候选转写、结构化 JSON 或自动回读不能替代这一步。同题旧版本的有效人工记录会显示为只读复查提示，并标明成品哈希是否相同；它不会预填或继承当前结论，当前 PNG 必须重新验收。
- **金标制作**：列出高考物理 300 条来源派生任务，供人工做额外来源审计；当前正式 OCR Gold 已由生成事实源验证，因此这里的草稿状态不影响 300 条来源派生评分。规范转写使用本地 KaTeX 实时显示视觉效果；含 LaTeX 的原始候选源码仅在主动展开“高级”区域后可见，且绝不能作为最终图片验收对象。

在“金标制作”中，标注人员可校正题型、题号、题干、选项、规范转写和公式状态，并直接在原图上拖框登记：

- 印刷文字区；
- 必须从原题裁切的插图区；
- 学生手写区；
- 页码、水印等忽略区。

两个环节的保存接口都会核对哈希并使用版本冲突保护。金标接口禁止修改来源、候选预标注和既有审计证据；“已验证”只能由三名互不相同的标注/复核人员按顺序签署，而且六项检查必须全为真；发生分歧的任务还要求第四名仲裁员。任何已签内容被修改后，活动签名自动失效并退回待标注，状态历史由服务端生成，浏览器不能覆盖。最终 PNG 页面另提供单图放大、全屏并排对照、联动滚动、全屏内逐项判定、分组检查进度、自动派生人工结论、固定保存区、明确的服务器保存回执、逐项提交阻断原因、图片加载门禁、状态筛选、样本搜索、上一题/下一题、保存当前进度、确认结论并下一题、`Ctrl+S` 和 `Ctrl+Enter`；当前运行与题号同步到网址的 `run`、`sample` 参数，便于收藏和跨复核员定位；金标入口还提供优先级筛选及进度 CSV 导出。

其余四个入口分别用于手写 Mask 标注、自建数据安全登记、拍照配对 Gold 和整页题界 Gold。拍照配对入口要求同时对照干净原稿与对应的普通手机照片或带手写照片，确认同页、内容完整、质量、条件、隐私及手写状态，再由另一人复核；上传时选择的关系不能直接成为 Gold。整页题界入口只接收已登记为 `boundary_page` 的真实隔离图片，标注至少两道完整题的外接框、题号、阅读顺序、单双栏、父题及跨页关系，再由不同复核员完成五项检查并签核。当前没有真实拍照或整页登记，所以两类进度均为 0；系统不会把现有单题样本、自动框或合成照片伪装成 Gold。

任何阶段都可运行 `scripts/run_self_built.ps1` 查看真实进度；它不会覆盖已填写的人工内容，也不会把 Gold 页的可选审计草稿错误地计入 410 张真实采集。

## 10. 仍需另外自建的数据

高考物理 300 题属于确定性来源派生基准，不包含真实手机拍照和手写分布，因此仍需另外采集：

- 60 页中文多题整页及双栏题界 Gold；
- 50 个干净原稿对应的至少 200 张真实手机照片；
- 100 张手写、印刷内容、教师批改和不可恢复重叠的像素级 mask；
- 同一原稿—带手写照片的严格成对数据。

这些数据用于题目切分、拍照鲁棒性和去手写评测，不替换本文件指定的 300 条 OCR Gold。

当前已先建立像素 Mask 的冻结评测入口：程序从已退役 Testzip 的人工失败证据中筛出 22 张冻结候选，并在 `data/annotations/handwriting_masks/tasks` 建立待标注任务，在 `data/splits/testzip_handwriting_mask_eval_manifest.jsonl` 记录不可变来源和 SHA-256。这些旧样本只用于独立安全评测，不是当前主 OCR Gold，也不计入 410 张真实自建采集；任务不复制原图，保持 `frozen_test=true`、`training_allowed=false`。Mask 必须与原图同尺寸并使用以下单通道像素值：`0` 背景、`1` 印刷内容、`2` 学生手写、`3` 教师批改、`4` 不可恢复重叠、`255` 不确定。

运行 `scripts/start_gold_review.ps1` 后，可在网页第三个入口“手写 Mask 标注”完成绘制。该入口同时显示人工证据筛选的冻结测试候选，以及每张从自建登记自动同步进来的 `handwriting_capture` 隔离任务，并在列表标明来源。网页支持六类笔刷、逐笔撤销、透明度、像素统计、保存草稿以及进度 CSV；服务端负责尺寸、灰度类别、SHA-256、来源登记和并发版本校验。提交需要一名标注员，最终核验需要另一名复核员完成五项检查；签核后的像素发生任何变化都会退回草稿。Mask PNG 保存在 `data/annotations/handwriting_masks/masks`，任务和签核记录保存在相邻的 `tasks` 目录。

目前为 `22` 张冻结候选待标注、`0` 张双人核验，距离 100 张最低目标还缺 `78` 张。现有 JSON 只是空任务模板，不能用于计算去手写准确率；后续需要继续采集真实含手写样本、绘制 Mask，并由两名不同人员签核。冻结测试图片及其 Mask 均不得进入训练、提示词、检索或阈值校准。

## 11. 新采集数据如何登记

复核工作台的第四个入口“自建数据登记”用于导入后续真实采集图片。必须先登记干净原稿，再登记与其对应的手机照片或带手写照片；系统会自动继承原稿组、学科和学段，要求匿名设备编号及至少一个拍摄条件。整页题界图片必须明确选择单栏、双栏或混合版式。

所有新图片初始状态固定为 `quarantine`，训练和评测权限均为关闭。登记前必须确认已经完成隐私检查、图片不含姓名/手机号/人脸等个人信息，并明确是否允许公开分发；不确定时应保持“不允许分发”。已许可的外部原稿还必须填写许可来源。相同字节按 SHA-256 只保存一份，原始图片位于 `D:\robot\Documents\ASK\datasets\test7_ocr_self_built_v1\sources`，不进入项目 Git；私有登记记录位于 `data/annotations/self_built_intake/records`。

### 11.1 批量登记

当采集数量增加后，使用 `templates/self_built_intake_batch.csv`，不需要在网页中逐张填写。CSV 必须与图片放在同一批次目录内，`image_path` 只能填写相对 CSV 的子路径。每批最多 1000 行，列名和布尔值必须严格使用模板；`true/false` 不能写成“是/否”。

正式采集不要直接在工程模板目录存放图片。先运行 `scripts/prepare_self_built_collection_workspace.py`，程序会在 `D:\robot\Documents\ASK\datasets\test7_ocr_self_built_v1\collection` 创建私有工作清单和 `images` 目录树。工作清单一旦建立便只读式保留人工修改；后续运行仅更新 `collection_readiness.json`、`collection_missing_files.csv` 和 `collection_ready_batch.csv`，不会覆盖文件名、隐私声明或其他采集记录。就绪小批次只包含真实图片有效、隐私声明完成且原稿依赖齐全的行，可以直接交给 `import_self_built_batch.py` 做增量只读预检。加入 `--check` 可完全只读检查，加入 `--require-ready` 可把 410 张图片和逐图隐私声明作为自动化门禁。

工程另提供已经排好配额的 `templates/self_built_collection_plan.csv`，网页“自建数据登记”也可直接下载同一份内容。该表有 410 行并可直接作为批量登记 CSV：50 份干净原稿、200 张普通手机照片、100 张带手写照片、60 张多题整页；数学和物理各占一半，单双栏各 30 页，300 张配对照片均衡分给三个匿名设备编号，并覆盖必需拍摄条件。每份干净原稿已经用 `clean_row_id` 关联四张普通照片和两张带手写照片。

计划中的 `image_path` 是建议的相对文件名。采集时可以按该目录命名真实文件，也可以修改为实际文件名，但不能填写绝对路径或跳出批次目录。`privacy_checked` 与 `contains_personal_data` 故意留空，防止模板替采集者作出隐私声明；检查每张真实图片后必须显式填写。公开分发默认是 `false`，只有确实拥有分发权时才能改为 `true`。`phone-a/b/c` 只是匿名槽位，采集前应在本机私下记录它们对应的三台实际设备，不要在 CSV 中写姓名、手机号或设备所有者。

```powershell
# 验证工程内的 410 行标准模板没有损坏
.\.venv\Scripts\python.exe .\scripts\create_self_built_collection_plan.py --check

# 把模板复制到真实采集批次目录后再填写；不要覆盖工程标准模板
Copy-Item .\templates\self_built_collection_plan.csv D:\待采集批次\batch.csv
```

四种 `role` 的用途如下：

| role | 图片 | 特殊字段 |
|---|---|---|
| `clean_page` | 无学生作答的干净原稿 | 不填配对、设备、条件，`page_layout=not_applicable` |
| `photo_capture` | 普通手机实拍 | 用 `clean_row_id` 指向本 CSV 的原稿，或用 `clean_asset_id` 指向既有原稿；填写设备和条件 |
| `handwriting_capture` | 同页带真实手写/批改的实拍 | 配对规则同普通照片；登记后自动进入配对与 Mask 队列 |
| `boundary_page` | 含多道题的真实整页 | `page_layout` 明确为 `single_column`、`double_column` 或 `mixed` |

`conditions` 用 `|` 分隔，可用值为 `frontal`、`perspective`、`shadow`、`fold_or_curve`、`blur`、`compression`、`uneven_light`、`screen_moire`。同一批次允许先写照片行、后写原稿行；系统会做两遍解析并强制先登记原稿。

```powershell
# 第一步始终做只读预检，不创建登记、源图或报告文件
.\.venv\Scripts\python.exe .\scripts\import_self_built_batch.py D:\待采集批次\batch.csv

# 第二步人工阅读预检结果后，显式写入隔离区
.\.venv\Scripts\python.exe .\scripts\import_self_built_batch.py D:\待采集批次\batch.csv --apply
```

正式导入前会先验证整批；只要一行的路径、格式、尺寸、配对、来源或隐私字段失败，整批都不开始登记。写入前会再次检查所有图片的 SHA-256，防止预检后替换文件。正式报告只保存结构化结果和哈希，位于 `data/annotations/self_built_intake/batches`，不复制图片。导入成功后会自动同步手写 Mask、拍照配对和整页题界三个复核队列，但仍然只是隔离候选，不是 Gold。

登记完成只表示原始数据已安全入库，不表示已形成 Gold。带手写照片会立即生成隔离 Mask 任务，但仍要完成像素标注、双人复核、文档组隔离、授权审计和冻结版本生成。

### 11.2 已核验干净原稿如何进入处理链

`configs/pipeline.json` 默认开启保守的干净来源解析，但它只接受角色为 `handwriting_capture` 且状态已经是 `verified` 的配对任务。处理一张图片时，程序先计算输入 SHA-256；只有它与登记照片逐字节一致，配对评估全部为真、手写状态为 `present`、复核检查全部为真，并且标注员与复核员是两名不同人员时，才把实际识别源切换为对应 `clean_page`。

该切换不是去手写算法，也不会改写照片像素：文字、公式和题内图片均重新从核验过的干净原稿直接识别或裁切。每次结果同时保存输入路径/哈希、干净原稿路径/哈希、配对任务路径/哈希、资产 ID 和两名签核者。延期回读、网页验收和发布阶段都会重新验证这些证据；任何源文件、任务内容或签核发生变化都会封闭失败。没有核验配对时继续处理原输入，并按照原有手写规则 `REJECT`，不会静默修复。

目前真实核验配对仍为 0，因此已有全量运行没有任何样本实际使用该替换。像 `TZIP-MATH-T09-Q10` 这类手写覆盖必要九宫格的题，在采集并双人核验同页干净原稿前仍保持自动拒绝。自建干净原稿和照片仍属于隔离数据，不会成为本项目的 OCR Gold，也不会因此获得训练或正式评测权限。

题界任务可独立执行 `scripts/check_question_boundary_gold.py` 检查；加入 `--require-targets` 后要求 60 页、300 题以及单栏/双栏各 30 页全部通过双人签核。拍照配对可执行 `scripts/check_capture_pair_gold.py`，严格模式要求 50 份原稿、200 对普通照片、100 对带手写照片、3 台设备和斜拍、阴影、折痕/曲面、模糊或压缩场景全部经过双人核验。总入口 `scripts/run_self_built.ps1 -RequireVerified` 会继续执行完 OCR Gold、手写 Mask、题界 Gold、拍照配对 Gold、原始采集和 D0-E 冻结就绪六组门禁，再汇总全部缺口，便于按真实进度迭代。当前普通模式通过结构校验，严格模式按设计失败：题界为 `0/60` 页、`0/300` 题，拍照配对和自建 Mask 均为 0，不能冻结或用于模型评测。

## 12. D0-E 数据冻结

`scripts/freeze_self_built_dataset.py` 是唯一冻结入口。它会验证 300 条 OCR Gold、100 张 Mask、60 页题界、200 对普通照片、100 对带手写照片、3 台设备和必要拍摄条件，并重新检查双人签核、原图/Mask 哈希、隐私、许可、文档组一致性以及本机 OCR Gold 与自建来源之间的 SHA-256 重叠。

```powershell
# 日常结构检查并更新就绪报告
.\.venv\Scripts\python.exe .\scripts\freeze_self_built_dataset.py

# 全部目标完成后先执行严格就绪检查
.\.venv\Scripts\python.exe .\scripts\freeze_self_built_dataset.py --require-targets

# 创建一次性、不可覆盖的私有冻结版本
.\.venv\Scripts\python.exe .\scripts\freeze_self_built_dataset.py --version test7-self-built-v2026.09.06-01
```

冻结版本写入 `data/frozen/<version>`，不进入 Git。它保存 OCR Gold、配对/题界/Mask 任务、来源登记清单和必要 Mask PNG，但不复制干净原稿或手机照片；源图只通过数据集名、相对路径和 SHA-256 引用。`freeze_manifest.json` 记录所有载荷文件的哈希、大小和整体内容指纹，已有版本名绝不允许覆盖。可用 `--verify-version <version>` 做恢复后校验。
