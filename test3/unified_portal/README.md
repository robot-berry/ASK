# test3 物理数学统一入口

这个入口把现有物理服务（本机 `8010`）和数学服务（本机 `8020`）放在同一个手机页面中。用户先选择物理或数学，再拍照上传；上传、状态查询和四张结果图都保持在统一网址下，不会跳转到其他端口。

## 启动

先确认原来的两个服务正在运行，然后在 PowerShell 中执行：

```powershell
cd D:\robot\Documents\ASK\picture\test3\unified_portal
.\start_unified_portal.ps1
```

需要作为隐藏后台服务运行时，可使用同一 Python 环境执行：

```powershell
D:\robot\Documents\ASK\picture\test1\.venv\Scripts\python.exe .\source\run_portal_detached.py
```

默认监听 `0.0.0.0:8030`。手机和电脑连接同一局域网后，访问：

```text
http://电脑的局域网IP:8030/
```

若不复用现有环境，可先安装 `requirements.txt` 中的依赖。

统一入口只负责转发，不会启动、停止或修改 `8010`、`8020` 上的服务。若只启动了其中一个后端，健康状态会显示另一个题目类型暂不可用。

## 当前处理边界

- 数学适用于小学、初中、高中；物理适用于小学科学中的物理内容、初中、高中。
- 两学科当前主要面向完整的非选择题，包括计算题、解答题、证明题、实验探究题、简答题和综合应用题。选择题、填空题、判断题暂不承诺答案隔离，不应理解为已经覆盖所有题型。
- 物理和数学模块默认保留照片中的全部笔迹。黑笔、铅笔与印刷字重叠时，不能保证安全去除。

## 统一接口

- `POST /api/{subject}/jobs`：上传题图，`subject` 只能是 `physics` 或 `math`。
- `GET /api/{subject}/jobs/{job_id}`：查询对应学科任务。
- `GET /api/{subject}/jobs/{job_id}/images/{filename}`：读取对应学科结果图。
- `GET /health`：聚合物理、数学两个后端的可用状态。

浏览器只访问当前 `8030` 来源。服务端仅允许转发到固定的本机地址 `127.0.0.1:8010` 和 `127.0.0.1:8020`，不会接受用户提供的目标地址。

## 四张结果图约束

完成状态必须同时具备并只展示以下四个 PNG，统一入口会按固定顺序重写为同源地址：

1. `01_question_knowledge.png`：原题知识图
2. `02_guidance_advanced.png`：学霸版（最少提示）
3. `03_guidance_standard.png`：普通版（标准提示）
4. `04_guidance_foundation.png`：基础版（详细提示）

其他文件名不会被代理。上传图片上限为 20 MB，结果响应也有大小和 PNG 文件签名检查。该页面没有登录功能，仅建议在可信局域网中使用。

## 测试

```powershell
D:\robot\Documents\ASK\picture\test1\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
