"""Build a local comparison from saved responses; no network or model calls."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
case_id = 'TZIP-MATH-T01-Q01'
models = [('当前 Qwen3.8-Max', 'qwen_max'), ('Kimi-K2.6', 'kimi')]
data = [(name, json.loads((root / f'{case_id}__{label}.json').read_text(encoding='utf-8'))) for name, label in models]
lines = ['# 同题五层解析原文对照', '', f'题目：{case_id}（集合交集）。', '',
         '仅比较解析呈现与时延，不对标准答案评分。此处是模型原始文本，未经生产系统脱敏、校验和渲染。', '',
         '| 指标 | 当前 Qwen3.8-Max | Kimi-K2.6 |', '|---|---:|---:|']
for label, key in [('首段内容（秒）', 'first_content_seconds'), ('完整响应（秒）', 'elapsed_seconds')]:
    lines.append(f'| {label} | {data[0][1][key]} | {data[1][1][key]} |')
lines += ['', '观察：两者均有五层且通过结构检查。Kimi的L1出现具体公式，L2也包含公式，L3已展开中间推演；与工程“浅层仅启发、深层逐步展开”的要求存在差异。Max在本题的浅层更克制。', '',
          '单题单次响应无法证明长期时延或整体解析能力；不能据此宣布任何候选等效替代。']
for tier in ('l1', 'l2', 'l3', 'l4', 'l5'):
    lines += ['', f'## {tier.upper()}', '']
    for name, response in data:
        lines += [f'### {name}', '']
        for part in response['parsed'][tier]:
            if isinstance(part, str):
                lines.append('- ' + part)
            else:
                lines.append('小问：' + part['n'])
                lines.append('')
                lines.extend(f'{i}. {s}' for i, s in enumerate(part['s'], 1))
                if part.get('pit'):
                    lines += ['', '易错点：' + part['pit']]
        lines.append('')
(root / 'PAIR_REVIEW.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('Saved PAIR_REVIEW.md; no provider calls')
