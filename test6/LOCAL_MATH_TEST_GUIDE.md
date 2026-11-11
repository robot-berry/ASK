# Test4 数学模块电脑本地验收说明

本轮只在当前电脑验证数学模块。确认通过前，不构建或部署新的服务器版本。

## 启动后的访问地址

- 运行 `start_local_math.ps1` 后，本机统一入口：<http://127.0.0.1:8130/>
- 数学后端健康检查：<http://127.0.0.1:8120/health>
- 20 题、80 张结果图离线总览：
  [打开本地排版总览](deployment_acceptance/local_math_layout_final_20260816T_final3/index.html)

本机入口只监听 `127.0.0.1`，其他电脑和手机无法访问。本脚本只新启动 Test4 数学后端和本地入口；物理是否可用取决于当前电脑原有的 Test3 物理服务是否正在运行，本轮不会启动、重启或改动它。

## 启动与停止

在 PowerShell 中进入 `D:\robot\Documents\ASK\picture\test4` 后运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\start_local_math.ps1
```

停止：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\stop_local_math.ps1
```

运行记录和日志只写入 `picture\test4\.local\`。

## 两种测试方式

### 1. 离线排版检查（不调用模型、不产生模型费用）

打开上面的“本地排版总览”，逐题检查四张图。重点查看：

- 下标是否真正显示为下标，例如 `x_B`、`y_A`；
- `k/x`、`y_A/2` 是否显示为上下分式；
- `m≠0`、`k>0`、`x_D<x_B` 是否保持完整，不在关系符处断行；
- 坐标、根式、向量、求和、统计表及题图是否清楚且无裁断；
- 每题仍严格是本地原题图、学霸版、普通版、基础版四张；后三张均包含知识点、基础公式和对应详细度的步骤。

重新生成完整离线结果：

```powershell
& '..\test1\.venv\Scripts\python.exe' `
  '.\math\render_local_dataset.py'
```

### 2. 电脑页面上传新题（会调用真实模型）

打开 <http://127.0.0.1:8130/>，选择“数学”，再上传 JPEG、PNG 或 WebP 题图。每次新上传都会调用真实模型，可能产生模型费用；重复刷新页面不会重复提交任务。

## 本轮自动验收结果

- 数学回归测试：99/99 通过；
- 离线数据集：20/20 题成功；
- 四图结果：80/80 张通过 PNG、800px、灰度 L、9cm 合同；
- 显示正文裸下划线、裸关系式和裸斜杠公式扫描：0；
- 80/80 张结果图的四边墨迹触边扫描：0；
- 已保存的真实题图通过本机预处理、强校验和四图渲染全流程，未调用模型，状态为 completed。

### 截图对应题目的同题前后对照

- [旧版第三张图](deployment_acceptance/math/screenshot_20260815T134530_server/03_guidance_standard.png)
- [本地修复后的第一张图](deployment_acceptance/screenshot_layout_final_20260816T_final2/20260815T162016_aef7d06ee04c4347817dd8ccb0dbd3d7/01_question_knowledge.png)
- [本地修复后的第三张图](deployment_acceptance/screenshot_layout_final_20260816T_final2/20260815T162016_aef7d06ee04c4347817dd8ccb0dbd3d7/03_guidance_standard.png)

本地修复后的同题任务为 `completed`，强校验 0 错误，四张图合同有效，正文裸公式扫描和四边墨迹触边扫描均为 0，并且没有再次调用模型。

## 服务器发布门槛

只有在电脑页面上传实测通过，并人工确认公式排版与四张图内容后，才重新构建并部署服务器版本。本地当前修改尚未发布到服务器。
