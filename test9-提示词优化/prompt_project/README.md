# Test9 集中提示词工程

Test9 的运行时提示词统一保存在本目录。各服务仍可通过环境变量覆盖默认来源，集中目录提供默认基线。

## 目录

- `base/math/`、`base/physics/`：学科基础提示词；
- `curriculum/math/`、`curriculum/physics/`：按年级和学期加载的范围提示词；
- `test10/提示词/`：Test10 六学段专题审计树；
- `prompts/统一数学提示词.md`：与封印提示词同级的 Test9 数学统一提示词；内部包含 77 个专题段，运行时按学期和题面抽取命中段；
- `prompts/seal_generation_v1.md`、`schemas/`、`registry.json`、`loader.py`：封印题主提示词及其响应 Schema；
- `tests/`：封印题提示词加载回归测试。

`prompt_store.py` 读取 `base/`，`semester_curriculum.py` 读取 `curriculum/`，`test10_prompts.py` 默认读取 `prompts/统一数学提示词.md` 并使用 `test10/提示词/` 的专题审计树，封印题桥接读取 `prompts/seal_generation_v1.md`。两套提示词同处 `prompts/`，但各自保留独立的运行契约和 Schema。`TEST9_PROMPT_DB`、`TEST10_PROMPT_ROOT` 和 `TEST11_PROMPT_ROOT` 仍可覆盖默认来源。
