# Test10 提示词工程接入

Test9 通过 `test10_prompts.py` 只读加载 `prompt_project/prompts/统一数学提示词.md`，它与封印提示词处于同一 `prompts/` 层级。运行时优先使用
`TEST10_PROMPT_ROOT` 指向的维护目录；未配置时使用 `prompt_project/test10/` 的专题审计树。统一文件内部嵌入六个年级、十二个学期和 77 个专题段：

```text
提示词/数学/
├── 初中/七年级/上学期、下学期
├── 初中/八年级/上学期、下学期
├── 初中/九年级/上学期、下学期
├── 高中/高一/上学期、下学期
├── 高中/高二/上学期、下学期
└── 高中/高三/上学期、下学期
```

统一文件的前半部分是路由闸门、教材范围和 Test9 五层 JSON 契约，后半部分是带稳定元数据的专题知识点、方法索引、公式卡、易错点和禁止事项。运行时只抽取当前学期及之前、且与题面匹配的专题段，不把约 38 万字的全部专题同时发送给模型；因此模型收到的仍是一段完整的单字符串提示词。验证结果写在 `搜题工程验证/覆盖验证`，不会混入提示词目录。

`test10_prompts.py` 负责年级学期过滤、关键词检索、统一文件 SHA256 和上下文组合。`math/source/search_machine_pipeline.py` 将实际正文、所选主题和统一文件来源哈希写入任务状态 `preprocess.test10_prompt`。检索是本地关键词近似匹配，不声称语义分类完全准确。

仍遵守 Test9 的五档 JSON、教学范围和不泄露最终答案规则。高年级题目需要低年级工具时，路由器只追加匹配的初中专题文件；不因命中专题放宽所选学期范围。

验证命令：

```powershell
python -m unittest discover -s picture/test9-提示词优化/tests -p test_test10_integration.py -v
python 搜题工程验证/覆盖验证/coverage_verify.py --prompt-root picture/test10-提示词工程 --output-dir 搜题工程验证/覆盖验证
```

覆盖报告会明确区分当前提示词树覆盖和题库/服务器历史证据；本地未提供题库 PNG 或历史结果时记录为 0，不伪造通过数量。
