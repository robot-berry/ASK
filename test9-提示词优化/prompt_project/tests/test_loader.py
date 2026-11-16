from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent))

from prompt_project.loader import PromptProjectError, load_prompt_package, render_prompt  # noqa: E402


def main() -> None:
    variables = {
        "request_id": "req-test",
        "question_text": "求方程的根",
        "knowledge": "一元二次方程",
        "l2": "关注未知量和条件",
        "l3": "识别方程与根的关系",
        "l4": "整理并代入",
    }
    package = load_prompt_package("seal_generation_v1", variables)
    assert package["version"] == "seal_generation_v1"
    assert "req-test" in package["prompt"]
    assert "{{" not in package["prompt"]
    try:
        render_prompt("seal_generation_v1", {"request_id": "only"})
    except PromptProjectError:
        pass
    else:
        raise AssertionError("missing variables must be rejected")
    schema = json.loads((PROJECT / "schemas" / "seal_generation_v1.schema.json").read_text(encoding="utf-8"))
    assert schema["properties"]["questions"]["required"] == ["q1", "q2", "q3"]
    print({"ok": True, "checks": ["version_registry", "variable_render", "required_variables", "schema_q1_q2_q3"]})


if __name__ == "__main__":
    main()
