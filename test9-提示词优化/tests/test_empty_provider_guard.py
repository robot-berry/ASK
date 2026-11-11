import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

class EmptyProviderGuard(unittest.TestCase):
    def run_guard(self, output, stream):
        source = Path(__file__).resolve().parents[1] / 'math/source/search_machine_pipeline.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and any(isinstance(v, ast.Constant) and v.value == 'empty_provider_output' for v in ast.walk(n)) and len(n.body) == 2)
        state = {}
        env = dict(attempt={'output_text': output}, collector=SimpleNamespace(latest=stream), state=state)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), env)
        return state

    def test_empty_and_whitespace_are_rejected(self):
        for output in (None, '', '   '):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, '未返回解析正文'):
                self.run_guard(output, '')

    def test_content_and_stream_are_preserved(self):
        for output, stream in [('{}', ''), ('', '{"l1": []}')]:
            self.assertEqual(self.run_guard(output, stream), {})
