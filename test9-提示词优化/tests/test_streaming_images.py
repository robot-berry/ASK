from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "math" / "source"
sys.path.insert(0, str(SOURCE))

from streaming_images import (  # noqa: E402
    EarlyKnowledgePublisher,
    KNOWLEDGE_PREFIX_KEYS,
    completed_ordered_prefix,
    validation_candidate,
)
import full_benchmark as benchmark  # noqa: E402


def prefix_payload() -> dict:
    return {
        "subject": "mathematics",
        "problem_scope": "primary_to_high_school",
        "recognition_status": "clear",
        "uncertain_items": [],
        "question_blocks": [
            {"type": "text", "text": "求 $x$ 的取值范围。", "figure_id": ""}
        ],
        "source_figures": [],
        "visual_evidence": [],
        "knowledge_points": ["函数定义域", "不等式"],
        "base_formulas": [
            {"formula": "f(x)>0", "condition": "研究函数正值区间时"}
        ],
    }


class StreamingImagesTests(unittest.TestCase):
    def test_completed_prefix_decodes_before_root_object_finishes(self) -> None:
        payload = prefix_payload()
        fields = ",".join(
            json.dumps(key, ensure_ascii=False)
            + ":"
            + json.dumps(payload[key], ensure_ascii=False)
            for key in KNOWLEDGE_PREFIX_KEYS
        )
        partial = "{" + fields + ',"guidance":{"advanced":'
        self.assertEqual(completed_ordered_prefix(partial), payload)
        self.assertIsNone(completed_ordered_prefix(partial[: partial.rfind("]")]))

    def test_out_of_order_prefix_is_not_published(self) -> None:
        payload = prefix_payload()
        payload = {"problem_scope": payload["problem_scope"], **payload}
        self.assertIsNone(
            completed_ordered_prefix(json.dumps(payload, ensure_ascii=False))
        )

    def test_validation_candidate_preserves_prefix_and_covers_parts(self) -> None:
        prefix = prefix_payload()
        candidate = validation_candidate(
            prefix,
            subject="mathematics",
            part_labels=("（1）", "（2）"),
        )
        self.assertEqual(candidate["question_blocks"], prefix["question_blocks"])
        self.assertEqual(
            [item["part"] for item in candidate["guidance"]["advanced"]["parts"]],
            ["（1）", "（2）"],
        )

    def test_early_publisher_atomically_publishes_only_once(self) -> None:
        payload = prefix_payload()
        stream = json.dumps({**payload, "guidance": {}}, ensure_ascii=False)
        published: list[tuple[dict, float | None]] = []
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "01_question_knowledge.png"

            def render(_prefix: dict, temporary: Path) -> None:
                temporary.write_bytes(b"validated-png")

            publisher = EarlyKnowledgePublisher(
                destination=destination,
                validate=lambda _prefix: [],
                render=render,
                inspect=lambda path: {
                    "path": str(path),
                    "filename": path.name,
                    "valid": path.read_bytes() == b"validated-png",
                },
                publish=lambda metadata, elapsed: published.append(
                    (dict(metadata), elapsed)
                ),
            )
            publisher(stream, 1234.5)
            publisher(stream, 1500.0)
            self.assertTrue(publisher.published)
            self.assertEqual(destination.read_bytes(), b"validated-png")
            self.assertEqual(len(published), 1)
            self.assertEqual(published[0][1], 1234.5)

    def test_provider_sse_callback_reaches_publisher_before_done(self) -> None:
        payload = {**prefix_payload(), "guidance": {}}
        content = json.dumps(payload, ensure_ascii=False)
        split = content.index(', "guidance"')
        chunks = (content[:split], content[split:])
        lines: list[bytes] = []
        for chunk in chunks:
            event = {
                "choices": [{"delta": {"content": chunk}, "finish_reason": None}]
            }
            lines.extend(
                [
                    (
                        "data: "
                        + json.dumps(event, ensure_ascii=False)
                        + "\n"
                    ).encode("utf-8"),
                    b"\n",
                ]
            )
        lines.extend([b"data: [DONE]\n", b"\n"])

        class Response:
            def readline(self) -> bytes:
                return lines.pop(0) if lines else b""

        published: list[dict] = []
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "01_question_knowledge.png"
            publisher = EarlyKnowledgePublisher(
                destination=destination,
                validate=lambda _prefix: [],
                render=lambda _prefix, path: path.write_bytes(b"validated-png"),
                inspect=lambda path: {
                    "path": str(path),
                    "filename": path.name,
                    "valid": path.read_bytes() == b"validated-png",
                },
                publish=lambda metadata, _elapsed: published.append(dict(metadata)),
            )
            with benchmark.stream_content_callback(publisher):
                parsed = benchmark.parse_sse_response(
                    Response(),
                    benchmark.monotonic_ms(),
                    2.0,
                    benchmark.ROOT_KEYS,
                )
        self.assertTrue(parsed["stream_complete"])
        self.assertTrue(publisher.published)
        self.assertEqual(len(published), 1)


if __name__ == "__main__":
    unittest.main()
